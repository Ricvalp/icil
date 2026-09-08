"""Small, reproducible context/generation panels for supervised policy training."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from .supervised_models import SupervisedModelConfig, generate


def prepare_plot_batch(dataset, *, count: int, support_count: int,
                       selection_mode: str, seed: int,
                       condition_on_support: bool = True) -> dict:
    """Choose fixed development contexts without loading demonstrated query actions.

    Category IDs and drawing IDs are selection/provenance metadata only. Neither
    category labels nor query tokens enter generation. Separate local RNGs keep
    plotting independent of training sampling, dropout, and diffusion noise.
    """
    if not 1 <= count <= 8:
        raise ValueError('Plot count must be between 1 and 8.')
    if not 0 <= support_count <= dataset.top_m:
        raise ValueError('Plot support count must fit the dataset top_m.')
    if selection_mode not in ('exact_top_k', 'sample_top_m'):
        raise ValueError('Unknown plot neighbor selection mode.')
    if not 0 <= seed < 2 ** 32:
        raise ValueError('Plot seed must fit an unsigned 32-bit integer.')
    development_rows = np.asarray(dataset.rows('development'))
    if not len(development_rows):
        raise ValueError('Training plots require development examples.')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8321]))
    categories = np.asarray(dataset.category_ids[development_rows])
    category_order = rng.permutation(np.unique(categories))
    pools = [rng.permutation(development_rows[categories == category]) for category in category_order]
    targets = []
    # Round robin covers distinct categories before repeating one on small data.
    offset = 0
    while len(targets) < min(count, len(development_rows)):
        for pool in pools:
            if offset < len(pool):
                targets.append(int(pool[offset]))
                if len(targets) == min(count, len(development_rows)):
                    break
        offset += 1
    target_rows = np.asarray(targets, dtype=np.int32)
    candidates = np.asarray(dataset.neighbors[target_rows])
    if selection_mode == 'sample_top_m':
        choices = np.stack([rng.choice(dataset.top_m, support_count, replace=False) for _ in targets])
        support_rows = np.take_along_axis(candidates, choices, axis=1)
    else:
        support_rows = candidates[:, :support_count]
    tokens = np.asarray(dataset.tokens[support_rows])
    mask = np.arange(dataset.max_steps) <= dataset.lengths[support_rows, None]
    if not condition_on_support:
        tokens, mask = np.zeros_like(tokens), np.zeros_like(mask)
    metadata = {
        'version': 1, 'dataset_id': dataset.identifier, 'split': 'development',
        'seed': int(seed), 'selection_mode': selection_mode,
        'condition_on_support': bool(condition_on_support),
        'support_count': int(support_count), 'target_rows': target_rows.tolist(),
        'support_rows': support_rows.tolist(),
        'target_ids': np.asarray(dataset.base_ids[target_rows]).tolist(),
        'support_ids': np.asarray(dataset.base_ids[support_rows]).tolist(),
        'category_names': [dataset.categories[int(dataset.category_ids[row])] for row in targets],
    }
    return {'support_tokens': tokens, 'support_mask': mask, 'metadata': metadata}


@lru_cache(maxsize=8)
def _generator(model_cfg: SupervisedModelConfig):
    # A batch of one bounds generation memory independently of the contact sheet
    # size, and makes each fixed example's random draw independent of chunking.
    return jax.jit(lambda params, tokens, mask, key:
                   generate(params, tokens, mask, model_cfg, key, deterministic=False))


def _draw_sketch(ax, tokens, point_mask, *, stopped: bool,
                 event_mask=None, raw_actions=None) -> list[str]:
    """Draw incoming-pen segments on the fixed normalized canvas, flag failures."""
    from matplotlib.collections import LineCollection

    tokens = np.asarray(tokens)
    point_mask = np.asarray(point_mask, dtype=bool)
    event_mask = point_mask if event_mask is None else np.asarray(event_mask, dtype=bool)
    raw_actions = tokens if raw_actions is None else np.asarray(raw_actions)
    finite = np.isfinite(tokens[:, :3]).all(axis=-1)
    points = tokens[point_mask]
    outside = np.isfinite(points[:, :2]).all(axis=-1) & np.any(np.abs(points[:, :2]) > 1., axis=-1)
    invalid = not np.isfinite(tokens[event_mask]).all() or not np.isfinite(raw_actions[event_mask]).all()
    status = []
    if not point_mask.any():
        status.append('EMPTY')
    if not stopped:
        status.append('NO STOP')
    if invalid:
        status.append('NONFINITE')
    if outside.any():
        status.append(f'OUTSIDE CANVAS: {int(outside.sum())}')
    # Never join across masked/nonfinite points or a destination pen-up event.
    joins = (point_mask[:-1] & point_mask[1:] & finite[:-1] & finite[1:]
             & (tokens[1:, 2] >= .5))
    segments = np.stack((tokens[:-1, :2][joins], tokens[1:, :2][joins]), axis=1)
    ax.add_collection(LineCollection(segments, colors='#20242b', linewidths=1.5, capstyle='round'))
    starts = point_mask & finite & ((tokens[:, 2] < .5) | ~np.r_[False, point_mask[:-1] & finite[:-1]])
    if starts.any():
        ax.scatter(tokens[starts, 0], tokens[starts, 1], s=3, color='#20242b', linewidths=0)
    ax.set(xlim=(-1.04, 1.04), ylim=(1.04, -1.04), aspect='equal', xticks=[], yticks=[])
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#cfd4da')
    status_text = '\n'.join(status) if status else f'{int(point_mask.sum())} points | STOP'
    ax.text(.5, -.06, status_text, transform=ax.transAxes, ha='center', va='top',
            fontsize=6, color='#b42318' if status else '#657080', linespacing=1.05)
    return status


def generate_plot(params, model_cfg: SupervisedModelConfig, plot_batch: dict,
                  output_path: str | Path, *, step: int) -> Path:
    """Save one PNG with the K actual contexts and a free-running generated sketch.

    The random draw stays fixed across optimizer steps and resumed runs. Failed
    outputs remain in their original slots and are labeled, including missing
    STOP, empty sketches, nonfinite values, and geometry outside the canvas.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    tokens = np.asarray(plot_batch['support_tokens'])
    mask = np.asarray(plot_batch['support_mask'])
    metadata = plot_batch['metadata']
    count, support_count = tokens.shape[:2]
    if not count or count != len(metadata['target_rows']):
        raise ValueError('Plot metadata and support examples do not match.')
    generator = _generator(model_cfg)
    key = jax.random.fold_in(jax.random.PRNGKey(metadata['seed']), 0x504C4F54)
    generated = []
    for index, row in enumerate(metadata['target_rows']):
        output = generator(params, jnp.asarray(tokens[index:index + 1]),
                           jnp.asarray(mask[index:index + 1]), jax.random.fold_in(key, row))
        generated.append(jax.tree.map(lambda value: np.asarray(value)[0], jax.device_get(output)))

    show_context = bool(metadata['condition_on_support']) and support_count > 0
    panel_count = support_count + 1 if show_context else 2
    columns = min(panel_count, 6)
    rows_per_example = math.ceil(panel_count / columns)
    rows = count * rows_per_example
    width, height = columns * 1.65 + .25, rows * 1.95 + .85
    dpi = min(110, 2400 / max(width, height))
    figure = Figure(figsize=(width, height), dpi=dpi, facecolor='white')
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    figure.subplots_adjust(left=.035, right=.985, bottom=.045, top=1 - .75 / height,
                           wspace=.18, hspace=.65)
    figure.suptitle(f'{model_cfg.architecture} | step {step:,}\nFixed validation contexts and sampling seeds',
                   fontsize=9, y=1 - .1 / height)
    for index, output in enumerate(generated):
        example_axes = axes[index * rows_per_example:(index + 1) * rows_per_example].ravel()
        category = metadata['category_names'][index]
        for panel, ax in enumerate(example_axes):
            if panel >= panel_count:
                ax.set_visible(False)
                continue
            if panel == panel_count - 1:
                _draw_sketch(ax, output['tokens'], output['point_mask'], stopped=bool(output['stopped']),
                             event_mask=output['event_mask'], raw_actions=output['raw_actions'])
                ax.set_title(f'{category}\nGenerated', fontsize=7, color='#17613a')
            elif show_context:
                support = tokens[index, panel]
                events = mask[index, panel]
                _draw_sketch(ax, support, events & (support[:, 3] < .5),
                             stopped=bool(np.any(events & (support[:, 3] >= .5))), event_mask=events)
                ax.set_title(f'{category}\nContext {panel + 1}', fontsize=7)
            else:
                ax.set_axis_off()
                ax.text(.5, .5, 'No context\n(unconditional)', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='#657080')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix='.png', delete=False) as stream:
            temporary = Path(stream.name)
            canvas.print_png(stream)
        temporary.replace(output_path)
    finally:
        figure.clear()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output_path
