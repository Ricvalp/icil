from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from icil_jax_rlbench.quickdraw.classifier_data import _load_renderer, prepare_classifier_cache
from icil_jax_rlbench.quickdraw.data import SketchRecord, SketchStore, create_fixture_cache
from icil_jax_rlbench.quickdraw.metrics import RENDERER_CONFIG, file_sha256


@pytest.fixture
def donor():
    root = Path(__file__).resolve().parents[2] / 'quick-robot-draw'
    if not (root / 'dataset' / 'rasterize.py').is_file():
        pytest.skip('Original sibling renderer is needed for exact raster parity.')
    pytest.importorskip('PIL')
    return root


def _manifest(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    payload = {key: value for key, value in manifest.items() if key != 'cache_id'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert digest == manifest['cache_id']
    for name, digest in manifest['files'].items():
        assert digest == file_sha256(path / name)
    return manifest


def test_classifier_cache_streams_exact_original_rasters_and_disjoint_ids(tmp_path, donor):
    source = tmp_path / 'source'
    store = create_fixture_cache(source, categories=3, drawings_per_category=64, seed=4)
    output = prepare_classifier_cache(source, donor, tmp_path / 'prepared',
                                      train_per_category=12, validation_per_category=4,
                                      test_per_category=3, policy_per_category=8,
                                      allow_subset_fixture=True)
    manifest = _manifest(output)
    assert manifest['total_counts'] == {'train': 36, 'validation': 12, 'test': 9, 'policy': 24}
    assert manifest['renderer_config'] == RENDERER_CONFIG
    assert manifest['environment']['numpy'] == np.__version__
    assert manifest['environment']['pillow']
    labels = json.loads((output / 'label_map.json').read_text())
    assert len(labels) == 345 and set(labels.values()) == set(range(345))
    assert set(store.categories).issubset(labels)
    assert sum(name.startswith('fixture_missing_') for name in labels) == 342
    images = np.load(output / 'images.npy', mmap_mode='r')
    image_labels = np.load(output / 'labels.npy', mmap_mode='r')
    splits = np.load(output / 'splits.npy', mmap_mode='r')
    assert isinstance(images, np.memmap) and images.shape == (57, 64, 64)
    assert images.dtype == np.float32 and image_labels.dtype == np.int64 and splits.dtype == np.uint8
    records = [json.loads(line) for line in (output / 'records.jsonl').read_text().splitlines()]
    renderer = _load_renderer(donor / 'dataset' / 'rasterize.py')
    config = renderer.RasterizerConfig(**RENDERER_CONFIG)
    clusters = {'train': set(), 'validation': set(), 'test': set()}
    for index, row in enumerate(records):
        record = store.get(row['drawing_id'])
        expected = renderer.rasterize_absolute_points(np.column_stack((record.absolute, record.incoming_pen)), config=config)
        np.testing.assert_array_equal(images[index], expected)
        assert image_labels[index] == labels[record.category]
        assert ('train', 'validation', 'test')[splits[index]] == row['split']
        clusters[row['split']].add(row['duplicate_cluster_id'])
    policy = SketchStore.open(output.parent / 'policy_cache')
    assert policy.identifier == manifest['policy_cache_id']
    assert len(policy) == 24
    clusters['policy'] = {record.duplicate_cluster_id for record in policy.records}
    assert sum(map(len, clusters.values())) == len(set.union(*clusters.values()))
    assert set(record.base_id for record in policy.records).isdisjoint(row['drawing_id'] for row in records)


def test_classifier_allocation_scaled_and_reproducible_with_duplicate_curation(tmp_path, donor):
    initial = create_fixture_cache(tmp_path / 'initial', categories=2, drawings_per_category=12)
    record = initial.records[0]
    duplicate = SketchRecord('same_category_copy', record.category, record.absolute, record.incoming_pen)
    cross = SketchRecord('cross_category_copy', initial.categories[1], record.absolute, record.incoming_pen)
    source = tmp_path / 'source'
    SketchStore.write(source, [*initial.records, duplicate, cross])
    outputs = [prepare_classifier_cache(source, donor, tmp_path / f'prepared_{index}',
                                        allow_subset_fixture=True) for index in range(2)]
    left, right = map(_manifest, outputs)
    assert left == right
    assert record.duplicate_cluster_id in left['excluded_cross_category_clusters']
    for row in left['actual_counts_per_category'].values():
        assert row['scaled_below_requested_caps']
        assert row['train'] >= 1 and row['validation'] >= 1 and row['test'] >= 1 and row['policy'] >= 5
        assert sum(row[name] for name in ('train', 'validation', 'test', 'policy')) == row['unique_eligible_clusters']
    with pytest.raises(FileExistsError):
        prepare_classifier_cache(source, donor, outputs[0].parent, allow_subset_fixture=True)


def test_classifier_cache_requires_explicit_fixture_and_sufficient_clusters(tmp_path, donor):
    source = tmp_path / 'source'
    create_fixture_cache(source, categories=2, drawings_per_category=7)
    with pytest.raises(ValueError, match='345'):
        prepare_classifier_cache(source, donor, tmp_path / 'not_real')
    with pytest.raises(ValueError, match='eight distinct'):
        prepare_classifier_cache(source, donor, tmp_path / 'too_small', allow_subset_fixture=True)
    assert not (tmp_path / 'too_small').exists()
    with pytest.raises(ValueError, match='5 policy'):
        prepare_classifier_cache(source, donor, tmp_path / 'bad_budget', policy_per_category=2,
                                 allow_subset_fixture=True)
