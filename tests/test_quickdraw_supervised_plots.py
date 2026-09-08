from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import numpy as np
from PIL import Image
import pytest

from icil_jax_rlbench.quickdraw import supervised_plots as plots
from icil_jax_rlbench.quickdraw.supervised_models import SupervisedModelConfig, init_model


def _dataset():
    # The two development categories each have two eligible query rows; supports
    # are separate reserved reference rows. Query tokens are poison deliberately.
    drawing = np.asarray([[-.8, -.8, 0, 0], [-.3, .1, 1, 0], [.3, -.1, 0, 0],
                          [.8, .8, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    tokens = np.stack([drawing.copy() for _ in range(8)])
    tokens[:4] = np.nan
    return SimpleNamespace(
        identifier='fixed-dataset', top_m=2, max_steps=5,
        categories=('cat', 'dog'), category_ids=np.asarray([0, 0, 1, 1, 0, 0, 1, 1]),
        base_ids=np.asarray([f'drawing-{row}' for row in range(8)]),
        tokens=tokens, lengths=np.full(8, 4),
        neighbors=np.asarray([[4, 5], [5, 4], [6, 7], [7, 6]] + [[-1, -1]] * 4),
        rows=lambda split: np.arange(4) if split == 'development' else np.arange(4, 8),
    )


def _batch(**overrides):
    return plots.prepare_plot_batch(_dataset(), **{
        'count': 2, 'support_count': 2, 'selection_mode': 'sample_top_m',
        'seed': 413, **overrides})


def _config(architecture):
    return SupervisedModelConfig(architecture=architecture, hidden_dim=8, num_heads=2,
        context_layers=1, decoder_layers=1, mlp_ratio=2, max_steps=5,
        mixture_components=2, dropout=.1, diffusion_steps=3, dtype='float32')


def test_plot_context_selection_is_fixed_held_out_diverse_and_query_free():
    before = np.random.get_state()
    first, second = _batch(), _batch()
    after = np.random.get_state()
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    assert first['metadata'] == second['metadata']
    assert set(first['metadata']['category_names']) == {'cat', 'dog'}
    assert set(first['metadata']['target_rows']) <= {0, 1, 2, 3}
    assert set(np.asarray(first['metadata']['support_rows']).ravel()) <= {4, 5, 6, 7}
    assert first['metadata']['split'] == 'development'
    json.dumps(first['metadata'])
    assert set(first) == {'support_tokens', 'support_mask', 'metadata'}
    assert np.isfinite(first['support_tokens']).all()
    np.testing.assert_array_equal(first['support_tokens'], second['support_tokens'])
    larger = _batch(count=8)
    assert len(larger['metadata']['target_rows']) == 4
    assert len(set(larger['metadata']['target_rows'])) == 4
    unconditional = _batch(condition_on_support=False)
    assert not unconditional['support_mask'].any()
    assert not unconditional['support_tokens'].any()
    assert not unconditional['metadata']['condition_on_support']
    zero = _batch(support_count=0)
    assert zero['support_tokens'].shape == (2, 0, 5, 4)


@pytest.mark.parametrize('architecture', ['autoregressive', 'diffusion'])
def test_actual_plot_generation_is_fixed_and_does_not_touch_training_rng(tmp_path, architecture, monkeypatch):
    cfg, batch = _config(architecture), _batch()
    training_key = jax.random.PRNGKey(32)
    original_key = np.asarray(training_key).copy()
    params = init_model(training_key, cfg, support_count=2)
    rendered = []
    original_draw = plots._draw_sketch

    def capture(ax, tokens, point_mask, **kwargs):
        if 'raw_actions' in kwargs:
            rendered.append(np.asarray(kwargs['raw_actions']).copy())
        return original_draw(ax, tokens, point_mask, **kwargs)

    monkeypatch.setattr(plots, '_draw_sketch', capture)
    first = plots.generate_plot(params, cfg, batch, tmp_path / 'first.png', step=10)
    second = plots.generate_plot(params, cfg, batch, tmp_path / 'second.png', step=20)
    assert first.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    for path in (first, second):
        with Image.open(path) as image:
            assert max(image.size) <= 2400
            pixels = np.asarray(image.convert('RGB'))
            assert pixels.std() > 5  # Figure contains real ink/text, not a blank canvas.
        assert path.stat().st_size < 500_000
    assert len(rendered) == 4
    np.testing.assert_array_equal(rendered[0], rendered[2])
    np.testing.assert_array_equal(rendered[1], rendered[3])
    np.testing.assert_array_equal(training_key, original_key)
    assert plots._generator(cfg) is plots._generator(cfg)


def test_plot_pen_gaps_invalid_status_and_no_stop_are_not_hidden():
    from matplotlib.figure import Figure

    figure = Figure()
    ax = figure.subplots()
    tokens = _dataset().tokens[4]
    status = plots._draw_sketch(ax, tokens, np.asarray([True] * 4 + [False]), stopped=True)
    assert status == []
    segments = ax.collections[0].get_segments()
    assert len(segments) == 2  # No false connection across the pen-up third point.
    np.testing.assert_array_equal(segments[0], tokens[:2, :2])
    np.testing.assert_array_equal(segments[1], tokens[2:4, :2])
    assert ax.get_ylim()[0] > ax.get_ylim()[1]
    bad = tokens.copy()
    bad[1, 0] = np.nan
    bad[3, 0] = 100
    second = figure.add_subplot(122)
    status = plots._draw_sketch(second, bad, np.ones(5, bool), stopped=False)
    assert set(status) == {'NO STOP', 'NONFINITE', 'OUTSIDE CANVAS: 1'}
    assert 'NONFINITE' in second.texts[0].get_text()
    assert len(second.collections[0].get_segments()) == 1
    status = plots._draw_sketch(ax, tokens, np.zeros(5, bool), stopped=True)
    assert status == ['EMPTY']
    figure.clear()


def test_unconditional_plot_explicitly_labels_hidden_context(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    cfg, batch = _config('autoregressive'), _batch(condition_on_support=False)
    params = init_model(jax.random.PRNGKey(0), cfg, support_count=2)
    captured = []
    original_clear = Figure.clear

    def inspect_clear(figure, *args, **kwargs):
        captured.extend(text.get_text() for ax in figure.axes for text in ax.texts)
        return original_clear(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, 'clear', inspect_clear)
    plots.generate_plot(params, cfg, batch, tmp_path / 'unconditional.png', step=1)
    assert captured.count('No context\n(unconditional)') == 2
