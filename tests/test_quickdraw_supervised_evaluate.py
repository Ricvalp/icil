"""Frozen evaluation contracts; synthetic inputs do not validate real retrieval."""
from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import jax
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import embeddings, full_data, supervised_evaluate as evaluation
from icil_jax_rlbench.quickdraw.metrics import FEATURE_VERSION, file_sha256, geometry_descriptor, load_trajectories
from icil_jax_rlbench.quickdraw.supervised_models import SupervisedModelConfig, init_model


def _dataset(monkeypatch, tmp_path):
    count, steps, per_category = 180, 8, 90
    categories = np.repeat(np.arange(2, dtype=np.int32), per_category)
    base_ids = np.asarray([f'category-{category}/{i:03d}' for category in range(2) for i in range(per_category)])
    tokens = np.zeros((count, steps, 4), np.float32)
    lengths = (4 + np.arange(count) % 3).astype(np.int32)
    for row in range(count):
        length = lengths[row]
        tokens[row, :length, 0] = np.linspace(-.7, .7, length) + row * .0002
        tokens[row, :length, 1] = np.sin(np.linspace(0, 3, length) + row) * .6
        tokens[row, 1:length, 2] = 1
        tokens[row, length, 3] = 1
    duplicate_ids = np.arange(count, dtype=np.int32)
    rows, references = {}, {}
    roles = np.zeros(count, np.uint8)
    for split, lower, upper, ref_lower, ref_upper in (
            ('train', 0, 24, 72, 72), ('development', 24, 60, 72, 84), ('test', 60, 72, 84, 90)):
        rows[split] = np.concatenate([np.arange(lower, upper) + offset for offset in (0, 90)]).astype(np.int32)
        references[split] = np.concatenate([np.arange(ref_lower, ref_upper) + offset for offset in (0, 90)]).astype(np.int32)
        roles[references[split]] = full_data.REFERENCE
    # Duplicate references must stay out of both global reference halves.
    for offset in (0, 90):
        duplicate_ids[offset + 73] = duplicate_ids[offset + 72]
        tokens[offset + 73], lengths[offset + 73] = tokens[offset + 72], lengths[offset + 72]
    neighbors = np.full((count, 4), -1, np.int32)
    for split_rows in rows.values():
        for row in split_rows:
            possible = split_rows[(categories[split_rows] == categories[row]) &
                                  (split_rows % 3 == row % 3) & (split_rows != row)]
            # Test split has only four per direction: fill last from another direction.
            if len(possible) < 4:
                extra = split_rows[(categories[split_rows] == categories[row]) &
                                   (split_rows % 3 != row % 3)]
                possible = np.r_[possible, extra]
            neighbors[row] = possible[:4]
    manifest = {'identifier': 'synthetic-full-dataset', 'categories': ['category-0', 'category-1'],
                'embedding_identifier': 'synthetic-embeddings', 'count': count, 'max_steps': steps, 'top_m': 4,
                'allow_subset_fixture': True}
    dataset = full_data.FullDataset(tmp_path, manifest, dict(tokens=tokens, lengths=lengths,
        category_ids=categories, duplicate_ids=duplicate_ids, base_ids=base_ids, neighbors=neighbors, roles=roles), rows, references)
    angles = (np.arange(count) % 3) * 2 * np.pi / 3
    cosine = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    mapping = {str(item): row for row, item in enumerate(base_ids)}
    features = SimpleNamespace(identifier='synthetic-embeddings', cosine=cosine, row=lambda item: mapping[item])
    monkeypatch.setattr(full_data.FullDataset, 'open', lambda *args, **kwargs: dataset)
    monkeypatch.setattr(embeddings.EmbeddingStore, 'open', lambda *args, **kwargs: features)
    return dataset


def _prepare(tmp_path, monkeypatch):
    dataset = _dataset(monkeypatch, tmp_path)
    path = evaluation.prepare_evaluation(tmp_path, tmp_path, tmp_path / 'episodes',
        tasks_per_category=2, support_count=2, reference_count=8, seed=5)
    return dataset, path


def test_frozen_rows_controls_duplicate_safe_references_and_test_opt_in(tmp_path, monkeypatch):
    dataset, path = _prepare(tmp_path, monkeypatch)
    arrays, metadata = evaluation.load_evaluation(path, dataset)
    assert len(arrays['target_rows']) == 4
    assert [record['intended_category'] for record in metadata['records']] == ['category-0', 'category-1'] * 2
    references = []
    for half in ('real_a', 'real_b'):
        reference_rows = arrays[f'reference_rows_{half}'].ravel()
        references.append(set(dataset.duplicate_ids[reference_rows].tolist()))
        sequences, records = load_trajectories(path / half)
        assert len(sequences['tokens']) == 16
        assert {record['intended_neighborhood_id'] for record in records['records']} == {
            record['intended_neighborhood_id'] for record in metadata['records']}
        for role in ('target_rows', 'support_rows', 'same_category_wrong_neighborhood_rows', 'wrong_support_rows'):
            assert not references[-1] & set(dataset.duplicate_ids[arrays[role].ravel()].tolist())
    assert not references[0] & references[1]
    for index, pairing in enumerate(metadata['same_category_wrong_pairings']):
        target = arrays['target_rows'][index]
        negative = arrays['same_category_wrong_neighborhood_rows'][index]
        assert np.all(dataset.category_ids[negative] == dataset.category_ids[target])
        assert pairing['centroid_cosine_distance'] >= .05
        assert pairing['member_jaccard'] <= .1
        assert dataset.duplicate_ids[target] not in dataset.duplicate_ids[negative]
        assert np.all(dataset.category_ids[arrays['wrong_support_rows'][index]] != dataset.category_ids[target])
    with pytest.raises(ValueError, match='allow-test'):
        evaluation.prepare_evaluation(tmp_path, tmp_path, tmp_path / 'test', split='test')
    with pytest.raises(FileExistsError, match='immutable'):
        evaluation.prepare_evaluation(tmp_path, tmp_path, path, support_count=2)
    metadata['seed'] = 123
    (path / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='metadata changed'):
        evaluation.load_evaluation(path)


def test_wrong_neighborhood_requires_real_separation():
    target = {'target': 0, 'target_group': 0, 'groups': {0, 1}, 'members': {0, 1},
              'centroid': np.asarray([1., 0.]), 'complexity': np.asarray([10., 2.])}
    overlap = {**target, 'target': 2, 'centroid': np.asarray([-1., 0.])}
    close = {**target, 'target': 3, 'groups': {3, 4}, 'members': {3, 4}}
    with pytest.raises(ValueError, match='No separated'):
        evaluation._wrong_neighborhood(target, [overlap, close], minimum_distance=.05, maximum_jaccard=.1)


@pytest.mark.parametrize('architecture', ['autoregressive', 'diffusion'])
def test_actual_generation_paired_keys_chunk_stability_and_metadata_isolation(tmp_path, monkeypatch, architecture):
    dataset, episodes = _prepare(tmp_path, monkeypatch)
    cfg = SupervisedModelConfig(architecture=architecture, hidden_dim=8, num_heads=2,
        context_layers=1, decoder_layers=1, mlp_ratio=2, max_steps=8,
        mixture_components=2, dropout=0., diffusion_steps=3, dtype='float32')
    params = init_model(jax.random.PRNGKey(2), cfg, support_count=2)
    module = ModuleType('icil_jax_rlbench.quickdraw.supervised_train')
    module.load_run = lambda *args, **kwargs: ({'params': params},
        {'support_count': 2, 'selection_mode': 'exact_top_k'}, cfg, dataset)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    checkpoint = tmp_path / 'checkpoint.pkl'
    checkpoint.write_bytes(b'synthetic-in-memory-parameters')
    conditions = evaluation.CONDITIONS
    first = evaluation.generate_run(checkpoint, episodes, tmp_path / 'generation',
        conditions=conditions, samples_per_task=2, batch_size=2, seed=9)
    metadata = json.loads((episodes / 'metadata.json').read_text())
    for record in metadata['records']:
        record.update(query_id='metadata-only', query_ids=['metadata-only'], target_row=-999,
                      intended_neighborhood_id='unused-by-generation', task_id='metadata-only', centroid=[99., -99.])
    metadata['identifier'] = evaluation._identifier(metadata)
    (episodes / 'metadata.json').write_text(json.dumps(metadata))
    second = evaluation.generate_run(checkpoint, episodes, tmp_path / 'repeated',
        conditions=('correct_support',), samples_per_task=2, batch_size=3, seed=9)
    original, original_metadata = load_trajectories(first / 'correct_support')
    repeated, _ = load_trajectories(second / 'correct_support')
    for name in original:
        np.testing.assert_array_equal(original[name], repeated[name])
    for condition in conditions:
        arrays, metadata = load_trajectories(first / condition)
        assert len(arrays['tokens']) == 8
        assert [row['query_seed'] for row in metadata['records']] == [row['query_seed'] for row in original_metadata['records']]
        assert [row['query_id'] for row in metadata['records']] == [row['query_id'] for row in original_metadata['records']]
        if condition == 'no_context':
            assert all(not row['support_ids'] for row in metadata['records'])
    summary = json.loads((first / 'summary.json').read_text())['conditions']
    assert np.isfinite(summary['correct_support']['conditional_target_loss']['mean'])
    objective = summary['correct_support']['conditional_target_loss']['objective']
    assert ('not a likelihood' in objective) == (architecture == 'diffusion')
    copy_report = evaluation.geometry_copying(tmp_path, first / 'support_copy', tmp_path / 'copy.json', query_batch_size=3)
    assert copy_report['training_population_count'] == 48
    assert all(item['exact_support_point_sequence'] for item in copy_report['per_generated'])
    assert max(item['nearest_support_descriptor_distance'] for item in copy_report['per_generated']) < 1e-6
    nearest, nearest_metadata = load_trajectories(first / 'nearest_training_example')
    for row, record in enumerate(nearest_metadata['records']):
        index = np.flatnonzero(dataset.base_ids == record['copied_training_id'])[0]
        assert index in dataset.rows('train')
        np.testing.assert_array_equal(nearest['tokens'][row], dataset.tokens[index])
    if architecture == 'autoregressive':
        # Exercise the unchanged frozen-feature scorer and overlap-aware paired
        # summaries. These artificial features assert bookkeeping, not quality.
        feature_root = tmp_path / 'features'
        for index, condition in enumerate(conditions):
            _feature_artifact(first / condition, feature_root / condition, shift=float(index))
        _feature_artifact(episodes / 'real_a', feature_root / 'real_a')
        _feature_artifact(episodes / 'real_b', feature_root / 'real_b')
        score = evaluation.score_run(first, feature_root, feature_root / 'real_a', tmp_path / 'scores.json',
                                     reference_repeat=[feature_root / 'real_b'], seed=5)
        assert score['paired_controls']['context_gain']['mean'] > 0
        assert score['paired_controls']['within_category_specificity']['mean'] > 0
        assert score['paired_controls']['context_gain']['independent_overlap_components'] >= 2
        assert score['paired_controls']['context_gain']['conditional_target_loss']['metric'] == 'conditional_target_loss'
        assert len(score['feature_scores']) == 6
        bad_path = first / 'no_context' / 'metadata.json'
        bad = json.loads(bad_path.read_text())
        bad['records'][0]['query_seed'] = [1, 2]
        bad_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='identical intended targets'):
            evaluation.score_run(first, feature_root, feature_root / 'real_a', tmp_path / 'invalid.json')


def _feature_artifact(source, output, *, shift=0.):
    _, metadata = load_trajectories(source)
    values = np.zeros((len(metadata['records']), 512), np.float32)
    values[:, 0] = np.asarray([int(row['intended_neighborhood_id'].split('/')[1]) / 30.
                                for row in metadata['records']]) + shift
    values[:, 1] = np.arange(len(values)) * .0001
    output.mkdir(parents=True)
    np.savez(output / 'features.npz', features=values)
    metadata.update(feature_version=FEATURE_VERSION, feature_normalization='none',
                    feature_dimension=512, actual_count=len(values),
                    extractor_sha256='synthetic-fixture', source_hashes={'fixture': 'synthetic'},
                    extractor_training_provenance={'synthetic_fixture': True},
                    renderer_config=metadata['renderer'],
                    artifact_sha256=file_sha256(source / 'trajectories.npz'),
                    artifact_metadata_sha256=file_sha256(source / 'metadata.json'))
    (output / 'metadata.json').write_text(json.dumps(metadata))


def test_vectorized_copy_descriptor_and_nearest_selection_use_raw_geometry(tmp_path, monkeypatch):
    dataset = _dataset(monkeypatch, tmp_path)
    rows = dataset.rows('train')[:8]
    descriptors = evaluation._geometry_descriptors(dataset, rows, chunk_size=3)
    expected = np.stack([geometry_descriptor(dataset.tokens[row, :dataset.lengths[row], :3]) for row in rows])
    np.testing.assert_allclose(descriptors, expected, atol=1e-7)
    selected, distances = evaluation._nearest_descriptors(descriptors[[5, 2]], descriptors, chunk_size=3)
    np.testing.assert_array_equal(selected, [5, 2])
    np.testing.assert_allclose(distances, 0., atol=1e-7)
