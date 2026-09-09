"""Reproducible checkpoint context panels and publication-sized sample galleries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import math
from pathlib import Path
import textwrap

import numpy as np

from .metrics import file_sha256
from .supervised_plots import _draw_sketch


def select_gallery_targets(dataset, *, count: int, split: str, seed: int) -> np.ndarray:
    """Round robin across shuffled categories, without selecting on model outputs."""
    if split not in ('development', 'test'):
        raise ValueError('Figures require the development or test split.')
    if count < 1 or not 0 <= seed < 2 ** 32:
        raise ValueError('A positive count and unsigned 32-bit seed are required.')
    available = np.asarray(dataset.rows(split), dtype=np.int32)
    if count > len(available):
        raise ValueError(f'Requested {count} examples but {split} has only {len(available)}.')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x46494753]))
    categories = np.asarray(dataset.category_ids[available])
    pools = [rng.permutation(available[categories == category])
             for category in rng.permutation(np.unique(categories))]
    selected, offset = [], 0
    while len(selected) < count:
        for pool in pools:
            if offset < len(pool):
                selected.append(int(pool[offset]))
                if len(selected) == count:
                    break
        offset += 1
    return np.asarray(selected, dtype=np.int32)


def _draw(ax, tokens, point_mask, *, stopped, event_mask, raw_actions=None):
    """Reuse the training renderer's pen semantics, omitting successful footers."""
    status = _draw_sketch(ax, tokens, point_mask, stopped=stopped,
                          event_mask=event_mask, raw_actions=raw_actions)
    for artist in list(ax.texts):
        artist.remove()
    for spine in ax.spines.values():
        spine.set_visible(False)
    if status:
        # Failed samples stay in their selected slots, including empty outputs.
        ax.text(.5, .015, '\n'.join(status), transform=ax.transAxes,
                ha='center', va='bottom', fontsize=5, color='#b42318',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8, 'pad': 1})
    return status


def _draw_generated(ax, arrays, index):
    return _draw(ax, arrays['tokens'][index], arrays['point_mask'][index],
                 stopped=bool(arrays['stopped'][index]), event_mask=arrays['event_mask'][index],
                 raw_actions=arrays['raw_actions'][index])


def _category(record):
    return '\n'.join(textwrap.wrap(str(record['intended_category']).replace('_', ' '), width=21))


def _save_figure(figure, output: Path, stem: str, formats, dpi: int) -> list[Path]:
    files = []
    for extension in formats:
        path = output / f'{stem}.{extension}'
        # Vector PDF/SVG preserve the actual sketch polylines for publication.
        figure.savefig(path, format=extension, dpi=dpi, facecolor='white')
        files.append(path)
    figure.clear()
    return files


def render_contexts(dataset, arrays, records, output, *, count: int,
                    condition_on_support: bool, formats=('png', 'pdf'), dpi=300):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    support_count = len(records[0]['support_rows']) if condition_on_support else 0
    panels = support_count + 1 if support_count else 2
    columns = min(panels, 6)
    per_example = math.ceil(panels / columns)
    rows = count * per_example
    figure = Figure(figsize=(columns * 1.65 + .65, rows * 1.65 + .55), facecolor='white')
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    figure.subplots_adjust(left=.07, right=.99, bottom=.025, top=.985,
                           wspace=.12, hspace=.28)
    statuses = {}
    for index, record in enumerate(records[:count]):
        group = axes[index * per_example:(index + 1) * per_example].ravel()
        group[0].set_ylabel(_category(record), fontsize=8, labelpad=8)
        for panel, ax in enumerate(group):
            if panel >= panels:
                ax.set_visible(False)
            elif panel == panels - 1:
                statuses[index] = _draw_generated(ax, arrays, index)
                ax.set_title('Generated', fontsize=9, color='#17613a', pad=2)
            elif support_count:
                support_row = record['support_rows'][panel]
                tokens = np.asarray(dataset.tokens[support_row])
                events = np.arange(dataset.max_steps) <= dataset.lengths[support_row]
                _draw(ax, tokens, events & (tokens[:, 3] < .5),
                      stopped=bool(np.any(events & (tokens[:, 3] >= .5))), event_mask=events)
                ax.set_title(f'Context {panel + 1}', fontsize=9, pad=2)
            else:
                ax.set(xticks=[], yticks=[])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.text(.5, .5, 'No context\n(unconditional)', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color='#657080')
    return _save_figure(figure, Path(output), 'context_samples', formats, dpi), statuses


def render_gallery(arrays, records, output, *, rows: int, columns: int,
                   formats=('png', 'pdf'), dpi=300, category_labels=True):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(columns * 1.25, rows * 1.25), facecolor='white')
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    figure.subplots_adjust(left=.015, right=.985, bottom=.015, top=.97,
                           wspace=.10, hspace=.20 if category_labels else .08)
    statuses = {}
    for index, ax in enumerate(axes.ravel()):
        statuses[index] = _draw_generated(ax, arrays, index)
        if category_labels:
            ax.set_title(_category(records[index]), fontsize=7, pad=2)
    return _save_figure(figure, Path(output), 'sample_gallery', formats, dpi), statuses


def visualize_checkpoint(checkpoint, output, *, dataset_root=None, split='development',
                         context_examples=10, grid_rows=10, grid_columns=10,
                         seed=2027, batch_size=8, formats=('png', 'pdf'), dpi=300,
                         category_labels=True, allow_test=False, progress=True):
    """Generate once; the first context examples also appear in the unfiltered grid."""
    import jax

    from .supervised_fid import generate_samples
    from .supervised_train import load_run

    if split == 'test' and not allow_test:
        raise ValueError('Test figures require explicit --allow-test.')
    if split not in ('development', 'test'):
        raise ValueError('Figures require the development or test split.')
    if min(context_examples, grid_rows, grid_columns, batch_size, dpi) < 1:
        raise ValueError('Example counts, grid dimensions, batch size, and dpi must be positive.')
    if not formats or set(formats) - {'png', 'pdf', 'svg'} or len(set(formats)) != len(formats):
        raise ValueError('Use distinct formats from png, pdf, and svg.')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Figure output already exists; choose a new directory: {output}')
    if progress:
        print('Verifying checkpoint and dataset for sample figures ...', flush=True)
    payload, cfg, model_cfg, dataset = load_run(checkpoint, dataset_root=dataset_root)
    count = max(context_examples, grid_rows * grid_columns)
    targets = select_gallery_targets(dataset, count=count, split=split, seed=seed)
    params = jax.device_put(payload['params'])
    condition_on_support = cfg.get('condition_on_support', True)
    arrays, records = generate_samples(params, model_cfg, dataset, targets,
        support_count=cfg['support_count'], selection_mode=cfg['selection_mode'],
        condition_on_support=condition_on_support, seed=seed, batch_size=batch_size, progress=progress)
    output.mkdir(parents=True, exist_ok=False)
    context_files, context_status = render_contexts(dataset, arrays, records, output,
        count=context_examples, condition_on_support=condition_on_support, formats=formats, dpi=dpi)
    gallery_files, gallery_status = render_gallery(arrays, records, output,
        rows=grid_rows, columns=grid_columns, formats=formats, dpi=dpi, category_labels=category_labels)
    statuses = {**context_status, **gallery_status}
    for index, record in enumerate(records):
        record['generation_status'] = statuses[index]
    sample_path = output / 'generated_samples.npz'
    np.savez_compressed(sample_path, **arrays)
    metadata = {
        'schema_version': 1, 'checkpoint_path': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': file_sha256(checkpoint), 'optimizer_step': int(payload['step']),
        'architecture': model_cfg.architecture, 'dataset_id': dataset.identifier,
        'model_config': asdict(model_cfg), 'generation_batch_size': batch_size,
        'execution': {
            'versions': {name: version(name) for name in ('jax', 'jaxlib', 'flax', 'numpy', 'matplotlib')},
            'backend': jax.default_backend(),
            'device_kinds': [device.device_kind for device in jax.local_devices()],
            'source_hashes': {name: file_sha256(Path(__file__).with_name(name)) for name in
                              ('supervised_visualize.py', 'supervised_plots.py', 'supervised_fid.py',
                               'supervised_models.py', 'full_data.py')},
        },
        'dataset_root': str(dataset.root.resolve()), 'split': split, 'seed': int(seed),
        'support_count': cfg['support_count'], 'selection_mode': cfg['selection_mode'],
        'condition_on_support': bool(condition_on_support), 'generated_count': count,
        'context_examples': context_examples, 'grid_rows': grid_rows, 'grid_columns': grid_columns,
        'category_labels': bool(category_labels), 'dpi': dpi,
        'selection_protocol': 'Shuffled category round robin and random distinct held-out query rows; no output filtering.',
        'generation_protocol': 'Retrieval-centroid query is excluded from its K nearest-neighbor contexts and model input.',
        'coordinate_mode': 'absolute', 'pen_semantics': 'incoming',
        'canvas': [-1.04, 1.04, -1.04, 1.04],
        'failure_policy': 'All outputs retained; empty, missing STOP, nonfinite, or out-of-canvas outputs labeled.',
        'context_indices': list(range(context_examples)),
        'gallery_indices': list(range(grid_rows * grid_columns)),
        'records': records,
        'files': {path.name: file_sha256(path) for path in [*context_files, *gallery_files, sample_path]},
    }
    (output / 'samples.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    if progress:
        print(f'Saved {count} generated samples, context panels, and gallery to {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-root')
    parser.add_argument('--split', choices=('development', 'test'), default='development')
    parser.add_argument('--allow-test', action='store_true')
    parser.add_argument('--context-examples', type=int, default=10)
    parser.add_argument('--grid-rows', type=int, default=10)
    parser.add_argument('--grid-columns', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf', 'svg'), default=('png', 'pdf'))
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--no-category-labels', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs['category_labels'] = not kwargs.pop('no_category_labels')
    kwargs['progress'] = not kwargs.pop('quiet')
    visualize_checkpoint(**kwargs)


if __name__ == '__main__':
    main()
