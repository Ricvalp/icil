from __future__ import annotations

import json

import jax
import numpy as np
from PIL import Image
import pytest

from icil_jax_rlbench.quickdraw import supervised_train as training
from icil_jax_rlbench.quickdraw import supervised_visualize as figures
from icil_jax_rlbench.quickdraw.metrics import file_sha256
from icil_jax_rlbench.quickdraw.supervised_models import SupervisedModelConfig, init_model
from test_quickdraw_supervised_evaluate import _dataset


def test_diverse_gallery_selection_is_fixed_held_out_and_prefix_stable(tmp_path, monkeypatch):
    dataset = _dataset(monkeypatch, tmp_path)
    first = figures.select_gallery_targets(dataset, count=10, split='development', seed=37)
    larger = figures.select_gallery_targets(dataset, count=20, split='development', seed=37)
    np.testing.assert_array_equal(first, larger[:10])
    assert len(set(first)) == 10
    assert set(first) <= set(dataset.rows('development'))
    np.testing.assert_array_equal(np.bincount(dataset.category_ids[first]), [5, 5])
    assert len(set(dataset.category_ids[first[:2]])) == 2
    with pytest.raises(ValueError, match='only'):
        figures.select_gallery_targets(dataset, count=1000, split='development', seed=37)
    with pytest.raises(ValueError, match='development or test'):
        figures.select_gallery_targets(dataset, count=2, split='train', seed=37)


@pytest.mark.parametrize('architecture', ['autoregressive', 'diffusion'])
def test_checkpoint_figures_use_actual_contexts_and_save_all_samples(tmp_path, monkeypatch, architecture):
    dataset = _dataset(monkeypatch, tmp_path)
    config = SupervisedModelConfig(architecture=architecture, hidden_dim=8, num_heads=2,
        context_layers=1, decoder_layers=1, mlp_ratio=2, max_steps=dataset.max_steps,
        mixture_components=2, dropout=0., diffusion_steps=3, dtype='float32')
    params = init_model(jax.random.PRNGKey(19), config, support_count=2)
    checkpoint = tmp_path / 'best.pkl'
    checkpoint.write_bytes(b'synthetic-checkpoint-with-in-memory-parameters')
    loaded = []

    def load_run(path, *, dataset_root):
        loaded.append((path, dataset_root))
        return ({'params': params, 'step': 1234},
                {'support_count': 2, 'selection_mode': 'sample_top_m', 'condition_on_support': True},
                config, dataset)

    monkeypatch.setattr(training, 'load_run', load_run)
    original_draw = figures._draw
    drawn = []

    def draw(ax, tokens, point_mask, **kwargs):
        drawn.append((np.asarray(tokens).copy(), kwargs.get('raw_actions') is not None))
        return original_draw(ax, tokens, point_mask, **kwargs)

    monkeypatch.setattr(figures, '_draw', draw)
    output = figures.visualize_checkpoint(checkpoint, tmp_path / 'figures',
        dataset_root='/override/on/hpc', context_examples=2, grid_rows=2, grid_columns=2,
        formats=('png', 'pdf', 'svg'), dpi=70, batch_size=2, seed=37, progress=False)
    assert loaded == [(checkpoint, '/override/on/hpc')]
    metadata = json.loads((output / 'samples.json').read_text())
    assert metadata['checkpoint_sha256'] == file_sha256(checkpoint)
    assert metadata['optimizer_step'] == 1234
    assert metadata['architecture'] == architecture
    assert metadata['model_config']['max_steps'] == dataset.max_steps
    assert metadata['generation_batch_size'] == 2
    assert metadata['execution']['backend'] == jax.default_backend()
    assert metadata['execution']['source_hashes']['supervised_visualize.py'] == file_sha256(figures.__file__)
    assert metadata['dataset_id'] == dataset.identifier
    assert metadata['selection_mode'] == 'sample_top_m'
    assert metadata['context_indices'] == [0, 1]
    assert metadata['gallery_indices'] == [0, 1, 2, 3]
    assert metadata['generated_count'] == len(metadata['records']) == 4
    with np.load(output / 'generated_samples.npz') as archive:
        assert archive['tokens'].shape == (4, dataset.max_steps, 4)
        assert archive['raw_actions'].shape == archive['tokens'].shape
        assert archive['stopped'].shape == (4,)
    contexts = [tokens for tokens, generated in drawn if not generated]
    expected_contexts = [dataset.tokens[row] for record in metadata['records'][:2]
                         for row in record['support_rows']]
    np.testing.assert_array_equal(contexts, expected_contexts)
    assert sum(generated for _, generated in drawn) == 6  # Two panels plus all four grid entries.
    for record in metadata['records']:
        assert record['target_row'] in dataset.rows('development')
        assert record['target_row'] not in record['support_rows']
        assert record['support_ids'] == np.asarray(dataset.base_ids[record['support_rows']]).tolist()
        assert isinstance(record['generation_status'], list)
    for stem in ('context_samples', 'sample_gallery'):
        with Image.open(output / f'{stem}.png') as image:
            assert np.asarray(image.convert('RGB')).std() > 5
        assert (output / f'{stem}.pdf').read_bytes().startswith(b'%PDF-')
        assert '<svg' in (output / f'{stem}.svg').read_text()
    with pytest.raises(FileExistsError, match='new directory'):
        figures.visualize_checkpoint(checkpoint, output)
    with pytest.raises(ValueError, match='allow-test'):
        figures.visualize_checkpoint(checkpoint, tmp_path / 'test', split='test')


def test_clean_panels_keep_pen_gaps_and_failure_labels():
    from matplotlib.figure import Figure

    tokens = np.asarray([[-.8, -.8, 0, 0], [-.3, .1, 1, 0], [.3, -.1, 0, 0],
                         [.8, .8, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    events = np.ones(5, bool)
    points = events & (tokens[:, 3] < .5)
    figure = Figure()
    ax = figure.subplots()
    status = figures._draw(ax, tokens, points, stopped=True, event_mask=events)
    assert status == [] and not ax.texts  # Successful gallery entries have no debug footer.
    segments = ax.collections[0].get_segments()
    assert len(segments) == 2
    np.testing.assert_array_equal(segments[0], tokens[:2, :2])
    np.testing.assert_array_equal(segments[1], tokens[2:4, :2])
    assert ax.get_ylim()[0] > ax.get_ylim()[1]
    second = figure.add_subplot(122)
    tokens[1, 0] = np.nan
    tokens[3, 0] = 10.
    status = figures._draw(second, tokens, points, stopped=False, event_mask=events,
                           raw_actions=tokens)
    assert set(status) == {'NO STOP', 'NONFINITE', 'OUTSIDE CANVAS: 1'}
    assert 'NONFINITE' in second.texts[0].get_text()
    assert len(second.collections[0].get_segments()) == 1
    third = figure.add_subplot(133)
    status = figures._draw(third, tokens, np.zeros(5, bool), stopped=True, event_mask=events)
    assert 'EMPTY' in status
    assert 'EMPTY' in third.texts[0].get_text()
    figure.clear()


def test_unconditional_context_panels_do_not_display_retrieved_demos(tmp_path, monkeypatch):
    class Dataset:
        @property
        def tokens(self):
            raise AssertionError('Unconditional figures must not load or display retrieved contexts.')

    tokens = np.asarray([[[-.5, -.5, 0, 0], [.5, .5, 1, 0], [0, 0, 0, 1]]], np.float32)
    arrays = {'tokens': tokens, 'raw_actions': tokens, 'point_mask': np.asarray([[1, 1, 0]], bool),
              'event_mask': np.ones((1, 3), bool), 'stopped': np.ones(1, bool)}
    records = [{'support_rows': [4, 5], 'intended_category': 'category'}]
    labels = []
    original_save = figures._save_figure

    def save(figure, *args, **kwargs):
        labels.extend(text.get_text() for ax in figure.axes for text in ax.texts)
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(figures, '_save_figure', save)
    figures.render_contexts(Dataset(), arrays, records, tmp_path, count=1,
                            condition_on_support=False, formats=('png',), dpi=70)
    assert 'No context\n(unconditional)' in labels
