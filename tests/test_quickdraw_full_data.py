from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import embeddings, full_data
from icil_jax_rlbench.quickdraw.data import SketchRecord, SketchStore


def _offline_python():
    python = os.environ.get('QUICKDRAW_METRIC_PYTHON')
    if python is None and importlib.util.find_spec('faiss'):
        python = sys.executable
    if python is None:
        pytest.skip('Full-data index checks require QUICKDRAW_METRIC_PYTHON with FAISS.')
    return python


def _sources(root, *, categories=2, count=48, duplicates=True):
    records = []
    for category in range(categories):
        for index in range(count):
            # Two within-category duplicates test group boundaries and replenishment.
            cluster_index = index - 1 if duplicates and index == 1 else index
            xy = np.asarray([[0., 0.], [.1 + cluster_index * .013, .2 + category * .019], [.2, -.1]], np.float32)
            records.append(SketchRecord(f'category-{category:03d}/{index:04d}', f'category-{category:03d}',
                                        xy, np.asarray([0, 1, 1], np.float32)))
    cache = SketchStore.write(root / 'cache', records, provenance={'synthetic_fixture': True})
    feature_root = root / 'embeddings'
    feature_root.mkdir()
    raw = np.random.default_rng(91).normal(size=(len(records), 512)).astype(np.float32)
    # Exact feature ties exercise stable base-ID ordering, even with duplicate groups.
    raw[:count] = raw[0]
    np.save(feature_root / 'raw.npy', raw)
    np.save(feature_root / 'cosine.npy', raw / np.linalg.norm(raw, axis=1, keepdims=True))
    rows = [{'row_index': row, 'base_id': record.base_id, 'category': record.category,
             'duplicate_cluster_id': record.duplicate_cluster_id} for row, record in enumerate(cache.records)]
    (feature_root / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = {'embedding_version': embeddings.EMBEDDING_VERSION, 'actual_count': len(rows),
                'cache_identifier': cache.identifier, 'raw_normalization': 'none',
                'cosine_normalization': 'l2', 'extractor_sha256': 'fixture-extractor',
                'allow_subset_fixture': True,
                'source_cache_files': {name: embeddings.file_sha256(root / 'cache' / name)
                                       for name in ('index.json', 'records.npz')},
                'files': {name: embeddings.file_sha256(feature_root / name)
                          for name in ('raw.npy', 'cosine.npy', 'records.jsonl')}}
    manifest['identifier'] = embeddings._identifier(manifest)
    (feature_root / 'manifest.json').write_text(json.dumps(manifest))
    return cache, feature_root


def _build(root, *, top_m=2):
    result = subprocess.run([_offline_python(), '-m', 'icil_jax_rlbench.quickdraw.full_data', 'build',
                    '--cache-root', str(root / 'cache'), '--embedding-root', str(root / 'embeddings'),
                    '--output', str(root / 'dataset'), '--top-m', str(top_m), '--max-steps', '5',
                    '--batch-size', '11', '--threads', '1', '--no-progress'],
                   capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return full_data.FullDataset.open(root / 'dataset')


def test_actual_faiss_full_dataset_covers_rows_with_exact_neighbors_and_masks(tmp_path):
    cache, feature_root = _sources(tmp_path)
    dataset = _build(tmp_path)
    assert len(dataset) == len(cache) == 96
    assert isinstance(dataset.tokens, np.memmap)
    assert isinstance(dataset.neighbors, np.memmap)
    assert dataset.categories == ('category-000', 'category-001')
    assert len(dataset.manifest['indexes']) == 6
    assert dataset.manifest['exclusions']['excluded_cross_category_drawings'] == 0
    assert dataset.manifest['target_coverage'].startswith('every_nonreference')
    assert sum(len(dataset.rows(split)) + len(dataset.reference_rows(split)) for split in full_data.SPLITS) == len(dataset)
    assert np.array_equal(dataset.rows('validation'), dataset.rows('development'))
    cosine = np.load(feature_root / 'cosine.npy').astype(np.float64)
    for split in full_data.SPLITS:
        all_rows = dataset.rows(split)
        references = dataset.reference_rows(split)
        assert not set(all_rows) & set(references)
        for target in all_rows:
            candidates = all_rows[dataset.category_ids[all_rows] == dataset.category_ids[target]]
            scores = np.asarray([np.sum(cosine[row] * cosine[target]) for row in candidates])
            order = np.lexsort((candidates, -scores))
            seen, expected = {int(dataset.duplicate_ids[target])}, []
            for position in order:
                row = candidates[position]
                if int(dataset.duplicate_ids[row]) not in seen:
                    expected.append(row)
                    seen.add(int(dataset.duplicate_ids[row]))
                if len(expected) == dataset.top_m:
                    break
            np.testing.assert_array_equal(dataset.neighbors[target], expected)
            np.testing.assert_allclose(dataset.neighbor_scores[target], cosine[expected] @ cosine[target], atol=1e-14)
    targets = dataset.rows('train')[:4]
    batch = dataset.batch(targets, support_count=2)
    assert batch['support_tokens'].shape == (4, 2, 5, 4)
    assert batch['support_mask'].shape == (4, 2, 5)
    assert batch['query_mask'].sum() == 4 * 4
    assert batch['query_point_mask'].sum() == 4 * 3
    np.testing.assert_array_equal(batch['query_tokens'][:, 3, 3], 1)
    np.testing.assert_array_equal(batch['query_tokens'][:, 4], 0)
    assert set(batch['metadata']) == {'target_rows', 'support_rows', 'category_ids'}
    first = dataset.batch(targets, support_count=1, selection_mode='sample_top_m', rng=np.random.default_rng(3))
    second = dataset.batch(targets, support_count=1, selection_mode='sample_top_m', rng=np.random.default_rng(3))
    np.testing.assert_array_equal(first['metadata']['support_rows'], second['metadata']['support_rows'])
    assert dataset.batch(targets, support_count=0)['support_tokens'].shape == (4, 0, 5, 4)
    with pytest.raises(ValueError, match='Generator'):
        dataset.batch(targets, selection_mode='sample_top_m', support_count=1)
    with pytest.raises(ValueError, match='eligible examples'):
        dataset.batch(dataset.reference_rows('train')[:1], support_count=1)


def test_all_345_categories_retain_train_development_test_populations():
    categories = tuple(f'category-{index:03d}' for index in range(345))
    category_ids = np.repeat(np.arange(345, dtype=np.int32), 5000)
    duplicate_ids = np.arange(len(category_ids), dtype=np.int32)
    clusters = tuple(f'duplicate-{index}' for index in duplicate_ids)
    split_codes, roles, counts, exclusions = full_data._assign_pools(
        category_ids, duplicate_ids, categories, clusters, seed=0,
        train_fraction=.7, development_fraction=.15, reference_fraction=.15, top_m=32)
    assert len(counts) == 345 and not any(exclusions.values())
    for code, split in enumerate(full_data.SPLITS):
        rows = np.flatnonzero((split_codes == code) & (roles == full_data.EXAMPLE))
        assert len(set(category_ids[rows])) == 345
        expected = 2975 if split == 'train' else 638
        assert len(rows) == 345 * expected
        assert all(counts[category][split]['examples'] == expected for category in categories)


def test_actual_full_builder_indexes_all_345_categories(tmp_path):
    _sources(tmp_path, categories=345, count=40, duplicates=False)
    dataset = _build(tmp_path)
    assert len(dataset.categories) == 345
    assert len(dataset) == 13_800
    assert len(dataset.manifest['indexes']) == 1_035
    assert dataset.manifest['split_counts'] == {
        'train': {'examples': 8_280, 'references': 1_380},
        'development': {'examples': 1_380, 'references': 690},
        'test': {'examples': 1_380, 'references': 690},
    }
    for split in full_data.SPLITS:
        assert len(set(dataset.category_ids[dataset.rows(split)])) == 345


def test_duplicate_groups_are_split_before_references_and_cross_category_groups_excluded():
    categories = ('a', 'b')
    category_ids = np.repeat(np.arange(2), 80)
    duplicate_ids = np.arange(160)
    duplicate_ids[1] = duplicate_ids[0]
    duplicate_ids[80] = duplicate_ids[79]  # One ambiguous cross-category group.
    args = dict(seed=8, train_fraction=.7, development_fraction=.15, reference_fraction=.15, top_m=2)
    first = full_data._assign_pools(category_ids, duplicate_ids, categories, tuple(map(str, range(160))), **args)
    second = full_data._assign_pools(category_ids, duplicate_ids, categories, tuple(map(str, range(160))), **args)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0][0] == first[0][1] and first[1][0] == first[1][1]
    assert first[0][79] == first[0][80] == -1
    assert first[1][79] == first[1][80] == full_data.EXCLUDED
    assert first[3] == {'cross_category_duplicate_groups': 1, 'excluded_cross_category_drawings': 2}


def test_source_alignment_and_attested_cache_hashes_are_checked(tmp_path):
    _sources(tmp_path)
    feature_root = tmp_path / 'embeddings'
    mapping = feature_root / 'records.jsonl'
    records = [json.loads(line) for line in mapping.read_text().splitlines()]
    records[3]['base_id'] = 'wrong-ID'
    mapping.write_text(''.join(json.dumps(row) + '\n' for row in records))
    manifest = json.loads((feature_root / 'manifest.json').read_text())
    manifest['files']['records.jsonl'] = embeddings.file_sha256(mapping)
    manifest['identifier'] = embeddings._identifier(manifest)
    (feature_root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='alignment mismatch'):
        full_data._aligned_sources(tmp_path / 'cache', feature_root, False)
    with (tmp_path / 'cache' / 'index.json').open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError, match='Source cache file hash mismatch'):
        full_data._aligned_sources(tmp_path / 'cache', feature_root, False)


def test_dataset_hash_and_semantic_boundary_corruption_rejected(tmp_path):
    _sources(tmp_path)
    dataset = _build(tmp_path)
    path = dataset.root / 'neighbors.npy'
    neighbors = np.load(path).copy()
    target = dataset.rows('train')[0]
    neighbors[target, 0] = target
    np.save(path, neighbors)
    with pytest.raises(ValueError, match='file hash mismatch'):
        full_data.FullDataset.open(dataset.root)
    with pytest.raises(ValueError, match='boundaries'):
        full_data.FullDataset.open(dataset.root, verify_hashes=False)


def test_reader_does_not_import_faiss_torch_or_jax():
    subprocess.run([sys.executable, '-c', '''
import sys
from icil_jax_rlbench.quickdraw.full_data import FullDataset
assert not {'torch', 'torchvision', 'faiss', 'jax'} & set(sys.modules)
'''], check=True)


def test_small_category_is_rejected_instead_of_silently_dropped():
    with pytest.raises(ValueError, match='never silently dropped'):
        full_data._assign_pools(np.zeros(10, np.int32), np.arange(10), ('tiny',), tuple(map(str, range(10))),
                               seed=0, train_fraction=.7, development_fraction=.15, reference_fraction=.15, top_m=2)
