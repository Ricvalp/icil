from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from icil_jax_rlbench.quickdraw.data import (
    SketchRecord,
    SketchSampler,
    SketchStore,
    absolute_to_deltas,
    b1_baselines,
    build_manifest,
    category_prototype,
    create_fixture_cache,
    deltas_to_absolute,
    import_ndjson,
    load_manifest,
    prepare_sequence,
    preprocess_drawing,
    save_manifest,
    trim_generated,
    validate_manifest,
)
from icil_jax_rlbench.quickdraw.pen import (
    PenConfig,
    PenEnvironment,
    apply_frame,
    invert_frame,
    replay_rollout,
    timed_reference,
    transformed_replay,
)


@pytest.fixture
def fixture_store(tmp_path):
    return create_fixture_cache(tmp_path / 'fixture', categories=8, drawings_per_category=24, seed=13)


def _assert_batches_equal(left, right):
    assert left['meta'] == right['meta']
    for section in ('support', 'query'):
        for key in left[section]:
            np.testing.assert_array_equal(left[section][key], right[section][key])


def test_preprocessing_preserves_lifted_travel_singletons_and_delta_origin():
    record = preprocess_drawing(
        [[[0, 10], [0, 0]], [[10, 10], [10, 20]], [[5], [5]]],
        base_id='cat/1', category='cat',
    )
    np.testing.assert_array_equal(record.incoming_pen, [0, 1, 0, 1, 0])
    np.testing.assert_allclose(record.absolute, [[-.5, -1], [.5, -1], [.5, 0], [.5, 1], [0, -.5]])
    np.testing.assert_allclose(record.deltas[0], record.absolute[0])
    np.testing.assert_allclose(record.deltas[2], [0, 1])
    np.testing.assert_allclose(deltas_to_absolute(absolute_to_deltas(record.absolute)), record.absolute)
    assert not record.absolute.flags.writeable
    with pytest.raises(ValueError, match='pen up'):
        SketchRecord('bad', 'cat', np.zeros((1, 2)), np.ones(1))


def test_sequence_has_one_stop_separate_masks_and_no_fabricated_boundary():
    sequence = prepare_sequence(np.asarray([[.4, .2], [.6, .1]]), np.asarray([0, 1]), 6)
    np.testing.assert_array_equal(sequence['event_mask'], [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(sequence['point_mask'], [1, 1, 0, 0, 0, 0])
    assert sequence['tokens'][:, 3].sum() == 1
    assert not sequence['state'].any()
    result = trim_generated(sequence['tokens'])
    assert result['has_stop'] and result['length'] == 2
    np.testing.assert_array_equal(result['incoming_pen'], [0, 1])
    assert not trim_generated(np.zeros((4, 4), np.float32))['has_stop']
    blank = prepare_sequence(np.zeros((0, 2)), np.zeros(0), 3)
    assert trim_generated(blank['tokens'])['length'] == 0
    with pytest.raises(ValueError, match='budget'):
        prepare_sequence(np.zeros((4, 2)), np.zeros(4), 4)


def test_cache_and_manifest_hashes_reject_tampering_and_overwriting(fixture_store, tmp_path):
    reopened = SketchStore.open(tmp_path / 'fixture')
    assert reopened.identifier == fixture_store.identifier
    np.testing.assert_array_equal(reopened.records[0].absolute, fixture_store.records[0].absolute)
    with pytest.raises(FileExistsError):
        SketchStore.write(tmp_path / 'fixture', fixture_store.records)
    manifest = build_manifest(fixture_store, max_points=31)
    path = tmp_path / 'manifest.json'
    save_manifest(path, manifest)
    assert load_manifest(path, fixture_store) == manifest
    changed = deepcopy(manifest)
    changed['budget']['family_count'] += 1
    with pytest.raises(ValueError, match='hash'):
        validate_manifest(fixture_store, changed)
    with pytest.raises(FileExistsError):
        save_manifest(path, changed)
    index_path = tmp_path / 'fixture' / 'index.json'
    index = json.loads(index_path.read_text())
    index['provenance']['seed'] = -1
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='hash'):
        SketchStore.open(tmp_path / 'fixture')


def test_native_ndjson_import_records_retention_and_does_not_invent_assets(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'cat.ndjson').write_text('\n'.join([
        json.dumps({'key_id': '1', 'drawing': [[[0, 2], [0, 1]]]}),
        json.dumps({'key_id': '2', 'drawing': [[[0, 1, 2, 3], [0, 1, 2, 3]]]}),
        json.dumps({'key_id': '3', 'drawing': []}),
    ]))
    store = import_ndjson(raw, tmp_path / 'cache', max_points=3)
    assert len(store) == 1 and store.records[0].base_id == 'cat/1'
    counts = store.provenance['retention']['cat']
    assert counts['retained'] == counts['outside_length_bounds'] == counts['invalid'] == 1
    with pytest.raises(FileNotFoundError, match='No local'):
        import_ndjson(tmp_path / 'absent', tmp_path / 'absent-cache')


def test_manifests_have_nested_category_and_program_reservoirs(fixture_store):
    small = build_manifest(fixture_store, family_count=1, unique_budget=12, max_points=31)
    larger = build_manifest(fixture_store, family_count=3, unique_budget=12, max_points=31)
    assert set(small['a_ids']['train']) < set(larger['a_ids']['train'])
    assert small['categories'] == larger['categories']
    assert small['a_ids']['test'] == larger['a_ids']['test']
    assert small['reference_ids'] == larger['reference_ids']
    assert sum(map(len, small['a_ids']['train'].values())) == 12
    assert sum(map(len, larger['a_ids']['train'].values())) == 12
    n = len(small['categories']['train'])
    programs_a = build_manifest(fixture_store, program_count=n, max_points=31)
    programs_b = build_manifest(fixture_store, program_count=n * 2, max_points=31)
    assert set(programs_a['b_ids']['train']) == set(programs_b['b_ids']['train'])
    for category, ids in programs_a['b_ids']['train'].items():
        assert set(ids) < set(programs_b['b_ids']['train'][category])
    with pytest.raises(ValueError, match='cover'):
        build_manifest(fixture_store, program_count=n - 1)


def test_duplicate_clusters_and_reference_ids_cannot_cross_splits(fixture_store):
    original = fixture_store.records[0]
    duplicate = SketchRecord('same_category_duplicate', original.category,
                             original.absolute + .00001, original.incoming_pen)
    cross = SketchRecord('cross_category_duplicate', 'new_category',
                         original.absolute, original.incoming_pen)
    # The exact duplicate crosses categories and is conservatively excluded.
    store = SketchStore([*fixture_store.records, duplicate, cross])
    manifest = build_manifest(store)
    assert original.duplicate_cluster_id in manifest['eligibility']['excluded_cross_category_clusters']
    for kind in ('a_ids', 'b_ids'):
        split_sets = [set(item for ids in manifest[kind][split].values() for item in ids)
                      for split in ('train', 'development', 'test')]
        assert not split_sets[0] & split_sets[1]
        assert not split_sets[0] & split_sets[2]
        assert not split_sets[1] & split_sets[2]
    references = {item for groups in manifest['reference_ids'].values() for ids in groups.values() for item in ids}
    assert not references & set.union(*split_sets)


@pytest.mark.parametrize('experiment', ['a', 'b1', 'b2'])
def test_sampler_axes_resume_and_task_boundaries(fixture_store, experiment):
    manifest = build_manifest(fixture_store)
    sampler = SketchSampler(fixture_store, manifest, experiment=experiment,
                            support_count=2, query_count=2, max_steps=96, seed=71)
    before = sampler.state_dict()
    batch = sampler.build_batch(3)
    restored = SketchSampler(fixture_store, manifest, experiment=experiment,
                             support_count=2, query_count=2, max_steps=96, seed=99)
    restored.load_state_dict(before)
    _assert_batches_equal(batch, restored.build_batch(3))
    _assert_batches_equal(sampler.build_batch(2), restored.build_batch(2))
    for section in ('support', 'query'):
        assert batch[section]['tokens'].shape == (3, 2, 96, 4)
        assert batch[section]['frame'].shape == (3, 2, 4)
        assert batch[section]['state'].shape == (3, 2, 96, 3)
        np.testing.assert_array_equal(batch[section]['tokens'][..., 3].sum(axis=-1), 1)
    for task in batch['meta']['tasks']:
        if experiment == 'a':
            assert not set(task['support_ids']) & set(task['query_ids'])
            assert len(set(task['support_ids'] + task['query_ids'])) == 4
        else:
            assert len(set(task['support_ids'] + task['query_ids'])) == 1
    if experiment != 'a':
        assert not np.array_equal(batch['support']['frame'][:, 0], batch['query']['frame'][:, 0])


def test_transform_and_start_rng_do_not_change_later_task_sampling(fixture_store):
    manifest = build_manifest(fixture_store)
    left = SketchSampler(fixture_store, manifest, experiment='b1', seed=12, support_count=1)
    right = SketchSampler(fixture_store, manifest, experiment='b2', seed=12, support_count=3)
    one, two = left.build_batch(5), right.build_batch(5)
    assert [task['task_id'] for task in one['meta']['tasks']] == [task['task_id'] for task in two['meta']['tasks']]
    assert set(left.state_dict()['rngs']) == {'category', 'drawing', 'frame', 'start'}


def test_sampler_can_replay_declared_task_and_rejects_missing_splits(fixture_store):
    manifest = build_manifest(fixture_store)
    sampler = SketchSampler(fixture_store, manifest, split='test', support_count=2)
    category = sampler.categories[0]
    assert sampler.build_batch(1, task_ids=[category])['meta']['tasks'][0]['category'] == category
    with pytest.raises(ValueError, match='outside'):
        sampler.build_batch(1, task_ids=[manifest['categories']['train'][0]])
    familiar = SketchSampler(fixture_store, manifest, experiment='b1', split='test')
    heldout = SketchSampler(fixture_store, manifest, experiment='b1', split='test', b_partition='heldout_category')
    assert set(familiar.categories).isdisjoint(heldout.categories)
    with pytest.raises(ValueError, match='only'):
        SketchSampler(fixture_store, manifest, support_count=100)


def test_b1_frame_inversion_and_wrong_instance_baselines(fixture_store):
    original = fixture_store.records[0]
    other = fixture_store.records[1]
    support_frame = np.asarray([.15, -.08, .4, .7])
    query_frame = np.asarray([-.12, .21, -.3, .8])
    support = apply_frame(original.absolute, support_frame)
    np.testing.assert_allclose(invert_frame(support, support_frame), original.absolute, atol=1e-7)
    target = apply_frame(original.absolute, query_frame)
    np.testing.assert_allclose(transformed_replay(support, support_frame, query_frame), target, atol=1e-7)
    baselines = b1_baselines(original, support, support_frame, query_frame)
    assert 'category_prototype' not in baselines
    assert np.mean((baselines['untransformed_replay'] - target) ** 2) > .01
    wrong = transformed_replay(apply_frame(other.absolute, support_frame), support_frame, query_frame)
    assert np.linalg.norm(wrong[0] - target[0]) > .01
    proto = category_prototype(fixture_store, [original.base_id, other.base_id])
    assert proto.base_id in {original.base_id, other.base_id}


def test_b2_teacher_states_are_actual_executions_and_initial_travel_is_pen_up(fixture_store):
    sampler = SketchSampler(fixture_store, build_manifest(fixture_store), experiment='b2',
                            support_count=1, max_steps=96)
    batch = sampler.build_batch(1)
    section = batch['query']
    mask = section['point_mask'][0, 0]
    commands = section['tokens'][0, 0, mask]
    states = section['state'][0, 0, :len(commands) + 1]
    np.testing.assert_allclose(states[1:, :2], states[:-1, :2] + commands[:, :2], atol=1e-7)
    assert np.all(np.linalg.norm(commands[:, :2], axis=-1) <= .1 + 1e-7)
    assert commands[0, 2] == 0
    target = batch['meta']['tasks'][0]['query'][0]['reference_absolute']
    np.testing.assert_allclose(states[1:, :2], target, atol=1e-7)


def test_b2_oracle_does_not_teleport_and_feedback_recovers_perturbation():
    absolute = np.asarray([[.0, .0], [.6, .0], [.6, .6]], np.float32)
    start = np.asarray([-.5, -.5], np.float32)
    reference, pen = timed_reference(absolute, np.asarray([0, 1, 1]), start)
    assert np.linalg.norm(reference[0] - start) <= .1
    assert pen[0] == 0 and len(reference) > len(absolute)
    oracle = replay_rollout(reference, pen, start_xy=start)
    np.testing.assert_allclose(oracle['absolute'], reference, atol=1e-7)
    assert oracle['terminated'] and oracle['pen_up_travel'] > .7
    perturbation = {8: np.asarray([0, .18], np.float32)}
    open_loop = replay_rollout(reference, pen, start_xy=start, feedback=False, perturbations=perturbation)
    feedback = replay_rollout(reference, pen, start_xy=start, feedback=True, perturbations=perturbation)
    open_error = np.mean((open_loop['absolute'] - reference) ** 2)
    feedback_error = np.mean((feedback['absolute'] - reference) ** 2)
    assert feedback_error < open_error
    assert feedback['invalid_actions'] == 0


def test_public_support_knots_recover_collinear_vertices_and_singletons():
    absolute = np.asarray([[0, 0], [.2, 0], [.4, 0], [.4, 0], [.6, .2]], np.float32)
    incoming_pen = np.asarray([0, 1, 1, 0, 1], np.float32)
    reference, pen, knots = timed_reference(absolute, incoming_pen, [-.5, -.5], return_knots=True)
    np.testing.assert_allclose(reference[knots], absolute, atol=1e-7)
    np.testing.assert_array_equal(pen[knots], incoming_pen)
    assert len(reference) > len(absolute)
    sequence = prepare_sequence(reference, pen, len(reference) + 2, knot_mask=knots)
    assert sequence['knot_mask'].sum() == len(absolute)
    assert not sequence['knot_mask'][len(reference):].any()


def test_pen_noise_delay_bounds_and_stop_are_explicit():
    env = PenEnvironment(PenConfig(max_motion=.1, delay_steps=1), seed=2)
    start = env.reset(np.asarray([.9, 0]))
    first, _ = env.step(np.asarray([1, 0, 1, 0]))
    np.testing.assert_array_equal(first, start)
    second, info = env.step(np.asarray([1, 0, 1, 0]))
    assert info['clipped_action']
    np.testing.assert_allclose(second[:2], [1, 0])
    _, info = env.step(np.asarray([np.nan, 0, 0, 0]))
    assert info['invalid_action']
    stopped, info = env.step(np.asarray([1, 0, 1, 1]))
    assert info['terminated']
    np.testing.assert_array_equal(stopped, env.state)
    with pytest.raises(RuntimeError, match='Reset'):
        env.step(np.zeros(4))
    noisy = PenConfig(noise_std=.01)
    a, b = PenEnvironment(noisy, seed=3), PenEnvironment(noisy, seed=3)
    np.testing.assert_array_equal(a.step(np.zeros(4))[0], b.step(np.zeros(4))[0])
