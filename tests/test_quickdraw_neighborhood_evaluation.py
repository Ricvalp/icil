"""Synthetic correctness checks; these do not validate classifier task semantics."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import metrics
from icil_jax_rlbench.quickdraw.aggregate import paired_neighborhood_summary
from icil_jax_rlbench.quickdraw.analysis import neighborhood_response_diversity
from icil_jax_rlbench.quickdraw.data import create_fixture_cache, save_manifest
from icil_jax_rlbench.quickdraw.evaluate import (
    export_references, generate_run, load_evaluation, prepare_evaluation, select_wrong_neighborhood,
)
from icil_jax_rlbench.quickdraw.neighborhoods import build_neighborhood_manifest
from icil_jax_rlbench.quickdraw.populations import export_copy_populations
from icil_jax_rlbench.quickdraw.pilot import _category_ablation_manifest
from icil_jax_rlbench.quickdraw.train import train


def _features(path, values, prefix, *, task_ids, supports=()):
    path.mkdir()
    values = np.pad(values, ((0, 0), (0, 512 - values.shape[1])))
    np.savez(path / 'features.npz', features=values)
    metadata = {
        'feature_version': metrics.FEATURE_VERSION, 'feature_normalization': 'none',
        'extractor_sha256': 'synthetic_fixture', 'feature_dimension': 512,
        'source_hashes': {'renderer': 'fixture'}, 'renderer_config': metrics.RENDERER_CONFIG,
        'manifest_id': 'fixture-manifest', 'actual_count': len(values),
        'extractor_training_provenance': {'synthetic_fixture': True},
        'records': [{'drawing_id': f'{prefix}{i}', 'intended_category': 'cat',
                     'intended_neighborhood_id': task, 'a_protocol': 'a_local',
                     'condition': 'independent_real_reference',
                     'support_ids': list(supports), 'synthetic_fixture': True}
                    for i, task in enumerate(task_ids)],
    }
    (path / 'metadata.json').write_text(json.dumps(metadata))
    return path


def test_same_category_conditional_swap_detected_while_pooled_fd_unchanged(tmp_path):
    # Distinct synthetic feature populations in one category, with independently
    # named real reference IDs. A real-data gate must be run separately.
    left = np.asarray([[-9., 0], [-8.9, .1], [-9.1, -.1]])
    right = -left
    x = np.concatenate([left, right])
    labels = ['cat/left'] * 3 + ['cat/right'] * 3
    a = _features(tmp_path / 'a', x, 'a', task_ids=labels)
    b = _features(tmp_path / 'b', x + .01, 'b', task_ids=labels)
    result = metrics.same_category_swap_check(a, b, left_task='cat/left', right_task='cat/right')
    assert result['pooled_fd_correct'] == pytest.approx(result['pooled_fd_swapped'], abs=1e-8)
    assert result['pooled_fd_invariant']
    assert result['conditional_mmd_increase'] > 1
    assert result['detects_swap']
    assert result['validation_scope'] == 'synthetic_or_smoke_software_check'
    score = metrics.score_feature_artifacts(a, b)
    assert score['conditional']['per_category'].keys() == {'cat'}
    assert score['neighborhood_conditional']['task_weighting'] == 'uniform_macro'
    assert len(score['neighborhood_conditional']['per_neighborhood']) == 2
    original = metrics.load_features(a)[1]
    altered = deepcopy(original)
    for row in altered['records']:
        row['intended_neighborhood_id'] = 'replacement'
    (a / 'metadata.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='neighborhood populations'):
        metrics.score_feature_artifacts(a, b)


def test_neighborhood_macro_weighting_and_overlap_block_uncertainty():
    records = [{'intended_category': 'cat', 'intended_neighborhood_id': task,
                'eligible_member_ids': members}
               for task, members in [('x', ['shared']), ('y', ['shared']), ('z', ['z'])]]
    left = {task: {'sketch_mmd': value} for task, value in zip(('x', 'y', 'z'), (1., 2., 6.))}
    right = {task: {'sketch_mmd': 0.} for task in left}
    result = paired_neighborhood_summary(left, right, records, seed=4)
    assert result['mean'] == 3.
    assert result['neighborhood_count'] == 3
    assert result['independent_overlap_components'] == 2
    assert result['ci95'] is not None
    repeated = records * 50
    assert result == paired_neighborhood_summary(left, right, repeated, seed=4)
    responses = np.asarray([[0., 0], [1., 0], [0., 1]])
    diversity = neighborhood_response_diversity(responses, records)
    assert diversity['category_count'] == 1 and diversity['neighborhood_count'] == 3
    assert diversity['across_category_means']['effective_dimension'] == 0
    assert diversity['within_category_neighborhoods']['cat']['effective_dimension'] > 1


def test_wrong_neighborhood_requires_separation_and_matches_complexity():
    def task(center, members, length=10, strokes=2, category='cat'):
        return dict(category=category, centroid=center, member_ids=members,
                    mean_length=length, mean_strokes=strokes)
    tasks = {'intended': task([1., 0.], ['a', 'b']),
             'overlap': task([-1., 0.], ['a', 'b']),
             'too_close': task([1., 0.], ['c']),
             'wrong_category': task([-1., 0.], ['d'], category='dog'),
             'length_mismatch': task([-1., 0.], ['e'], length=50),
             'matched': task([0., 1.], ['f'])}
    selected, metadata = select_wrong_neighborhood(tasks, 'intended')
    assert selected == 'matched'
    assert metadata['centroid_cosine_distance'] == 1
    assert metadata['relative_length_stroke_mismatch'] == 0
    with pytest.raises(ValueError, match='No separated'):
        select_wrong_neighborhood({key: tasks[key] for key in ('intended', 'overlap', 'too_close')}, 'intended')


def _assets(root, protocol, *, matched_category=False):
    cache = root / 'cache'
    store = create_fixture_cache(cache, categories=8, drawings_per_category=256, seed=3)
    rows = [{'base_id': row.base_id, 'category': row.category,
             'duplicate_cluster_id': row.duplicate_cluster_id} for row in store.records]
    # Three well-separated artificial embedding directions; no ResNet semantics.
    angles = np.asarray([int(row['base_id'].rsplit('/', 1)[1]) % 3 for row in rows]) * (2 * np.pi / 3)
    cosine = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    embedding = SimpleNamespace(cache_id=store.identifier, records=rows, cosine=cosine,
                                identifier='synthetic-fixture', manifest={'synthetic_fixture': True})
    manifest = build_neighborhood_manifest(store, embedding, a_protocol=protocol,
        split_regime='familiar_drawings', family_count=2, neighborhood_size=3, top_m=3,
        reference_per_neighborhood=4, reference_per_category=4, max_points=15, seed=13)
    if matched_category:
        manifest = _category_ablation_manifest(store, manifest, embedding)
    manifest_path = root / 'manifest.json'
    save_manifest(manifest_path, manifest)
    return cache, store, manifest_path, manifest


@pytest.mark.parametrize('protocol', ['a_nn', 'a_local'])
def test_frozen_local_controls_references_and_actual_copy_populations(tmp_path, protocol):
    cache, store, manifest_path, manifest = _assets(tmp_path, protocol)
    episodes = prepare_evaluation(cache, manifest_path, tmp_path / 'episodes', tasks=4,
                                  support_count=1, query_count=1, max_steps=16, seed=7)
    sections, metadata = load_evaluation(episodes)
    for intended, negative, pairing in zip(metadata['records'], metadata['same_category_wrong_records'],
                                           metadata['same_category_wrong_pairings']):
        assert intended['intended_category'] == negative['intended_category']
        assert intended['intended_neighborhood_id'] != negative['intended_neighborhood_id']
        assert pairing['centroid_cosine_distance'] >= .05
        assert pairing['member_jaccard'] <= .1
    refs = []
    for half in ('real_a', 'real_b'):
        path = export_references(cache, manifest_path, tmp_path / half, episodes=episodes,
                                 half=half, max_steps=16)
        _, references = metrics.load_trajectories(path)
        refs.append(metrics._reference_ids(references))
        assert {row['intended_neighborhood_id'] for row in references['records']} == {
            row['intended_neighborhood_id'] for row in metadata['records']}
        for role in ('records', 'wrong_records', 'same_category_wrong_records'):
            assert refs[-1].isdisjoint(item for row in metadata[role] for item in row['support_ids'])
    assert refs[0].isdisjoint(refs[1])
    populations = export_copy_populations(cache, manifest_path, episodes, tmp_path / 'copy', training_count=4)
    _, support_metadata = metrics.load_trajectories(populations / 'supports')
    exported = {row['drawing_id'] for row in support_metadata['records']}
    assert exported == {item for role in ('records', 'wrong_records', 'same_category_wrong_records')
                        for row in metadata[role] for item in row['support_ids']}


def test_coarse_ablation_uses_same_split_safe_reference_api(tmp_path):
    cache, _, manifest_path, _ = _assets(tmp_path, 'a_category')
    episodes = prepare_evaluation(cache, manifest_path, tmp_path / 'episodes', tasks=2,
                                  support_count=1, max_steps=16)
    _, evaluation = load_evaluation(episodes)
    assert not evaluation['same_category_wrong_records']
    path = export_references(cache, manifest_path, tmp_path / 'real', episodes=episodes, max_steps=16)
    _, references = metrics.load_trajectories(path)
    assert len({row['intended_category'] for row in references['records']}) == 2


@pytest.mark.parametrize('matched_category', [False, True])
def test_category_sampling_full_feature_score_boundary(tmp_path, matched_category):
    # Emulate the external feature worker's metadata forwarding with explicitly
    # synthetic vectors; exercise generated/reference/calibration scoring end to end.
    protocol = 'a_nn' if matched_category else 'a_category'
    cache, _, manifest_path, manifest = _assets(tmp_path, protocol, matched_category=matched_category)
    episodes = prepare_evaluation(cache, manifest_path, tmp_path / 'episodes', tasks=4,
                                  support_count=1, max_steps=16)
    _, evaluation = load_evaluation(episodes)
    assert all(bool(row['intended_neighborhood_id']) == matched_category for row in evaluation['records'])
    generated_records = [{**record, 'drawing_id': f'generated-{i}-{sample}'}
                         for i, record in enumerate(evaluation['records']) for sample in range(2)]
    metadata_sets = [generated_records]
    for half in ('real_a', 'real_b'):
        path = export_references(cache, manifest_path, tmp_path / half, episodes=episodes,
                                 half=half, max_steps=16)
        metadata_sets.append(metrics.load_trajectories(path)[1]['records'])
    feature_paths = []
    rng = np.random.default_rng(34)
    for i, records in enumerate(metadata_sets):
        path = _features(tmp_path / f'features-{i}', rng.normal(size=(len(records), 2)), f'population{i}',
                         task_ids=['placeholder'] * len(records))
        metadata = json.loads((path / 'metadata.json').read_text())
        metadata['records'] = records
        metadata['manifest_id'] = manifest['identifier']
        (path / 'metadata.json').write_text(json.dumps(metadata))
        feature_paths.append(path)
    score = metrics.score_feature_artifacts(feature_paths[0], feature_paths[1], reference_repeats=feature_paths[2:])
    assert (score['neighborhood_conditional'] is not None) == matched_category
    assert (score['real_real'][0].get('neighborhood_conditional') is not None) == matched_category
    if matched_category:
        assert 'target-centered local-distribution proxy' in score['local_reference_interpretation']
        assert set(score['neighborhood_conditional']['per_neighborhood']) == {
            record['intended_neighborhood_id'] for record in evaluation['records']}
    else:
        assert score['local_reference_interpretation'] is None


def test_new_protocol_generation_metadata_boundary_and_probabilistic_controls(tmp_path):
    cache, store, manifest_path, manifest = _assets(tmp_path, 'a_nn')
    episodes = prepare_evaluation(cache, manifest_path, tmp_path / 'episodes', tasks=2,
                                  support_count=1, max_steps=16, seed=8)
    cfg = {'cache_root': str(cache), 'manifest_path': str(manifest_path),
           'experiment': 'a', 'a_protocol': 'a_nn', 'model_type': 'ttt_kvb_full', 'seed': 1,
           'batch_size': 1, 'support_count': 1, 'query_count': 1,
           'num_steps': 1, 'checkpoint_every': 1, 'log_every': 1, 'output_dir': str(tmp_path / 'run'),
           'model': {'hidden_dim': 4, 'fast_dim': 2, 'fast_hidden_dim': 3, 'mixture_components': 2,
                     'segment_size': 16, 'max_steps': 16}}
    run = train(cfg)
    original = generate_run(run / 'last.pkl', episodes, tmp_path / 'original',
        conditions=('no_update', 'correct_support', 'same_category_wrong_neighborhood', 'support_copy',
                    'nearest_training_example'), samples_per_task=2, seed=19)
    metadata_path = episodes / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    for record in metadata['records']:
        record['anchor_id'] = 'changed-private-anchor'
        record['intended_neighborhood_id'] = 'changed-private-neighborhood'
        record['task_id'] = 'changed-private-task'
        record['query_ids'] = ['changed-private-query']
    metadata.pop('identifier')
    metadata['identifier'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    changed = generate_run(run / 'last.pkl', episodes, tmp_path / 'changed',
                           conditions=('correct_support',), samples_per_task=2, seed=19)
    a, ma = metrics.load_trajectories(original / 'correct_support')
    b, mb = metrics.load_trajectories(changed / 'correct_support')
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert ma['records'][0]['intended_neighborhood_id'] != mb['records'][0]['intended_neighborhood_id']
    _, wrong = metrics.load_trajectories(original / 'same_category_wrong_neighborhood')
    for correct, negative in zip(ma['records'], wrong['records']):
        assert correct['query_ids'] == negative['query_ids']
        assert correct['query_seed'] == negative['query_seed']
        assert correct['intended_neighborhood_id'] == negative['intended_neighborhood_id']
        assert correct['source_neighborhood_id'] != negative['source_neighborhood_id']
        assert 'coordinate_loss' in negative['conditional_target_loss']
    _, copied = metrics.load_trajectories(original / 'nearest_training_example')
    allowed = {item for ids in manifest['a_ids']['train'].values() for item in ids}
    assert all(row['copied_training_id'] in allowed for row in copied['records'])
    # Frozen copying geometry uses actual trajectories and exact permitted IDs,
    # independently of the classifier used for task construction.
    populations = export_copy_populations(cache, manifest_path, episodes, tmp_path / 'geometry_populations')
    geometry = metrics.score_geometry_copying(original / 'support_copy', populations / 'supports',
                                               populations / 'training_reservoir', manifest_path=manifest_path)
    assert geometry['invalid_or_empty_count'] == 0
    assert all(row['nearest_support_geometry_distance'] == 0 for row in geometry['per_generated'])
    nearest_geometry = metrics.score_geometry_copying(original / 'nearest_training_example', populations / 'supports',
                                                       populations / 'training_reservoir', manifest_path=manifest_path)
    assert all(row['nearest_training_geometry_distance'] == 0 for row in nearest_geometry['per_generated'])
    jax.clear_caches()
