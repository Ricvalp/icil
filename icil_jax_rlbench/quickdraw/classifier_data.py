"""Prepare disjoint fresh-evaluator and policy data using the donor renderer.

This offline preparation imports no PyTorch, FAISS, or donor training loaders.
Raster arrays are streamed to memory-mapped files in the original float32 scale.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import monotonic

import numpy as np

from .data import DUPLICATE_RULE, SketchStore
from .metrics import RENDERER_CONFIG, file_sha256


SPLIT_NAMES = ('train', 'validation', 'test', 'policy')


def _status(message: str, enabled: bool) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _progress(items, description: str, *, total: int | None = None,
              unit: str = 'items', enabled: bool = False):
    """Report completed work to stderr without changing iteration order."""
    if not enabled:
        yield from items
        return
    total = len(items) if total is None else total
    stream = sys.stderr
    terminal = stream.isatty()
    interval = 0.25 if terminal else 5.0
    started = last_update = monotonic()

    def report(completed, now):
        elapsed = now - started
        fraction = completed / total if total else 1.0
        filled = int(20 * fraction)
        bar = '#' * filled + '-' * (20 - filled)
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = f'{(total - completed) / rate:.0f}s' if rate > 0 else '--'
        line = (f'{description}: [{bar}] {fraction:6.1%} '
                f'{completed:,}/{total:,} {unit} | {rate:.1f}/s | '
                f'elapsed {elapsed:.0f}s | ETA {eta}')
        print(('\r' if terminal else '') + line + ('   ' if terminal else ''),
              end='' if terminal else '\n', file=stream, flush=True)

    report(0, started)
    try:
        for completed, item in enumerate(items, 1):
            yield item
            now = monotonic()
            if completed == total or now - last_update >= interval:
                report(completed, now)
                last_update = now
    finally:
        if terminal:
            print(file=stream, flush=True)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _quotas(available: int, caps: np.ndarray) -> np.ndarray:
    minimum = np.asarray([1, 1, 1, 5], np.int64)
    if available < int(minimum.sum()):
        raise ValueError('Need at least eight distinct clusters per category: 1/1/1 evaluator and 5 policy.')
    if available >= int(caps.sum()):
        return caps.copy()
    # Allocate remaining capacity proportionally after respecting each minimum.
    headroom = caps - minimum
    fractions = (available - minimum.sum()) * headroom / max(1, int(headroom.sum()))
    extra = np.floor(fractions).astype(np.int64)
    remaining = available - int((minimum + extra).sum())
    order = sorted(range(4), key=lambda i: (-(fractions[i] - extra[i]), i))
    for index in order[:remaining]:
        extra[index] += 1
    return minimum + extra


def _load_renderer(path: Path):
    name = '_quickdraw_fresh_classifier_rasterizer'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load the original renderer: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_classifier_cache(
    cache_root: str | Path,
    donor_root: str | Path,
    output: str | Path,
    *,
    train_per_category: int = 2000,
    validation_per_category: int = 200,
    test_per_category: int = 200,
    policy_per_category: int = 5000,
    seed: int = 37,
    allow_subset_fixture: bool = False,
    progress: bool = False,
) -> Path:
    """Split base clusters before rendering or creating the independent policy pool.

    A fresh evaluator is a newly trained measurement resource, not a reproduction
    of an unavailable historical checkpoint. All 345 categories are required
    unless the caller explicitly requests a software fixture.
    """
    requested = [train_per_category, validation_per_category, test_per_category, policy_per_category]
    if any(not isinstance(value, (int, np.integer)) for value in requested):
        raise ValueError('Per-category budgets must be integers.')
    caps = np.asarray(requested, np.int64)
    if np.any(caps < np.asarray([1, 1, 1, 5])):
        raise ValueError('Budgets need at least 1 train/validation/test and 5 policy records per category.')
    output, donor_root = Path(output).resolve(), Path(donor_root).resolve()
    if output.exists():
        raise FileExistsError(f'Classifier preparation outputs are immutable: {output}')
    renderer_path = donor_root / 'dataset' / 'rasterize.py'
    if not renderer_path.is_file():
        raise FileNotFoundError(f'Original renderer is required: {renderer_path}')
    _status(f'Loading and validating vector cache: {cache_root} ...', progress)
    store = SketchStore.open(cache_root)
    _status(f'Loaded {len(store):,} drawings in {len(store.categories):,} categories.', progress)
    categories = sorted(store.categories)
    if len(categories) > 345 or (not allow_subset_fixture and len(categories) != 345):
        raise ValueError('Fresh evaluator preparation requires all 345 categories; use allow_subset_fixture only for tests.')
    categories_path = donor_root / 'categories.txt'
    source_hashes = {'rasterizer': file_sha256(renderer_path), 'classifier_data': file_sha256(__file__)}
    if not allow_subset_fixture:
        official_categories = {line.strip() for line in categories_path.read_text().splitlines() if line.strip()}
        if len(official_categories) != 345 or set(categories) != official_categories:
            raise ValueError('Cache categories must match the donor official 345-category list.')
        source_hashes['category_list'] = file_sha256(categories_path)
    names = list(categories)
    index = 0
    while len(names) < 345:
        name = f'fixture_missing_{index:03d}'
        if name not in names:
            names.append(name)
        index += 1
    label_map = {name: index for index, name in enumerate(sorted(names))}
    clusters = defaultdict(list)
    raw_counts = defaultdict(int)
    for record in _progress(store.records, 'Grouping duplicates', unit='drawings', enabled=progress):
        clusters[record.duplicate_cluster_id].append(record)
        raw_counts[record.category] += 1
    unique = {category: [] for category in categories}
    excluded = []
    _status(f'Sorting {len(clusters):,} duplicate clusters ...', progress)
    for cluster, records in _progress(sorted(clusters.items()), 'Selecting unique drawings',
                                     unit='clusters', enabled=progress):
        if len({record.category for record in records}) > 1:
            excluded.append(cluster)
            continue
        representative = min(records, key=lambda record: record.base_id)
        unique[representative.category].append(representative)
    selected, policy_records, actual = [], [], {}
    for category in _progress(categories, 'Splitting drawings', unit='categories', enabled=progress):
        ordered = sorted(unique[category], key=lambda record: hashlib.sha256(
            _canonical(['fresh_evaluator_split_v1', int(seed), category, record.duplicate_cluster_id])
        ).hexdigest())
        try:
            counts = _quotas(len(ordered), caps)
        except ValueError as error:
            raise ValueError(f'{category}: {error}') from error
        cursor = 0
        for split, count in enumerate(counts):
            records = ordered[cursor:cursor + int(count)]
            if split < 3:
                selected.extend((record, split) for record in records)
            else:
                policy_records.extend(records)
            cursor += int(count)
        actual[category] = {
            'source_records': raw_counts[category], 'unique_eligible_clusters': len(ordered),
            **{name: int(value) for name, value in zip(SPLIT_NAMES, counts)},
            'unused_clusters': len(ordered) - cursor,
            'scaled_below_requested_caps': bool(np.any(counts < caps)),
        }
    renderer = _load_renderer(renderer_path)
    renderer_config = renderer.RasterizerConfig(**RENDERER_CONFIG)
    output.mkdir(parents=True, exist_ok=False)
    _status(f'Writing and compressing policy cache ({len(policy_records):,} drawings) ...', progress)
    policy = SketchStore.write(output / 'policy_cache', policy_records, provenance={
        'kind': 'policy_pool_disjoint_from_new_evaluator_v1', 'input_cache_id': store.identifier,
        'input_provenance': store.provenance, 'seed': int(seed), 'evaluation_only_pool_excluded': True,
        'source_hashes': source_hashes, 'duplicate_rule': DUPLICATE_RULE,
        'split_caps': dict(zip(SPLIT_NAMES, map(int, caps))),
        'actual_policy_count_per_category': {category: row['policy'] for category, row in actual.items()},
    })
    _status('Policy cache saved. Creating classifier image arrays ...', progress)
    rasters = output / 'rasters'
    rasters.mkdir()
    images = np.lib.format.open_memmap(rasters / 'images.npy', mode='w+', dtype=np.float32,
                                      shape=(len(selected), 64, 64))
    labels = np.lib.format.open_memmap(rasters / 'labels.npy', mode='w+', dtype=np.int64,
                                      shape=(len(selected),))
    splits = np.lib.format.open_memmap(rasters / 'splits.npy', mode='w+', dtype=np.uint8,
                                      shape=(len(selected),))
    with (rasters / 'records.jsonl').open('x', encoding='utf-8') as records_file:
        for index, (record, split) in _progress(enumerate(selected), 'Rendering classifier images',
                                               total=len(selected), unit='images', enabled=progress):
            points = np.column_stack((record.absolute, record.incoming_pen))
            pixels = renderer.rasterize_absolute_points(points, config=renderer_config)
            if pixels.shape != (64, 64) or pixels.dtype != np.float32 or not np.all(np.isfinite(pixels)):
                raise ValueError('Original renderer must return finite float32 64x64 images.')
            images[index] = pixels
            labels[index] = label_map[record.category]
            splits[index] = split
            records_file.write(_canonical({
                'row_index': index, 'drawing_id': record.base_id, 'category': record.category,
                'label': label_map[record.category], 'split': SPLIT_NAMES[split],
                'duplicate_cluster_id': record.duplicate_cluster_id,
            }).decode('utf-8') + '\n')
    _status('Flushing classifier image arrays to disk ...', progress)
    for array in (images, labels, splits):
        array.flush()
    del images, labels, splits
    (rasters / 'label_map.json').write_bytes(_canonical(label_map) + b'\n')
    files = {}
    for name in ('images.npy', 'labels.npy', 'splits.npy', 'records.jsonl', 'label_map.json'):
        _status(f'Hashing {name} ...', progress)
        files[name] = file_sha256(rasters / name)
    manifest = {
        'schema_version': 1, 'role': 'new_evaluator_v1',
        'input_cache_id': store.identifier, 'input_provenance': store.provenance,
        'policy_cache_id': policy.identifier, 'policy_cache': '../policy_cache',
        'allow_subset_fixture': bool(allow_subset_fixture),
        'present_category_count': len(categories), 'classifier_output_count': 345,
        'seed': int(seed), 'split_names': list(SPLIT_NAMES[:3]),
        'split_caps': dict(zip(SPLIT_NAMES, map(int, caps))),
        'quota_rule': 'proportional_remaining_capacity_after_minimum_train1_validation1_test1_policy5',
        'actual_counts_per_category': actual,
        'total_counts': {name: sum(row[name] for row in actual.values()) for name in SPLIT_NAMES},
        'excluded_cross_category_clusters': excluded, 'duplicate_rule': DUPLICATE_RULE,
        'renderer_config': dict(RENDERER_CONFIG), 'source_hashes': source_hashes,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pillow': version('Pillow'), 'raster_backend': 'Pillow original donor rasterizer'},
        'image_dtype': 'float32', 'image_shape': [len(selected), 64, 64],
        'feature_role': 'evaluation_only_never_policy_inputs',
        'partition_contract': 'Evaluator train/validation/test and policy duplicate clusters are disjoint.',
        'files': files,
    }
    manifest['cache_id'] = hashlib.sha256(_canonical(manifest)).hexdigest()
    (rasters / 'manifest.json').write_bytes(_canonical(manifest) + b'\n')
    _status('Preparation complete.', progress)
    return rasters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root', 'donor-root', 'output'):
        parser.add_argument('--' + name, required=True)
    for name, default in [('train-per-category', 2000), ('validation-per-category', 200),
                          ('test-per-category', 200), ('policy-per-category', 5000)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--allow-subset-fixture', action='store_true')
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                        help='Disable stage messages and progress bars on stderr')
    print(prepare_classifier_cache(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
