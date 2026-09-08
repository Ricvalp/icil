from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import jax
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw.data import (
    SketchSampler, build_manifest, create_fixture_cache, save_manifest,
)
from icil_jax_rlbench.quickdraw.evaluate import (
    _b1_ordered_tracking, export_references, generate_run, load_evaluation, prepare_evaluation,
)
from icil_jax_rlbench.quickdraw.metrics import _reference_ids, file_sha256, load_trajectories
from icil_jax_rlbench.quickdraw.pen import PenConfig
from icil_jax_rlbench.quickdraw.train import train
from icil_jax_rlbench.train.checkpoints import load_checkpoint


def _tree_equal(left, right):
    lhs, lhs_tree = jax.tree_util.tree_flatten(left)
    rhs, rhs_tree = jax.tree_util.tree_flatten(right)
    assert lhs_tree == rhs_tree
    for x, y in zip(lhs, rhs):
        np.testing.assert_array_equal(x, y)


@pytest.fixture(scope='module')
def runner_assets(tmp_path_factory):
    root = tmp_path_factory.mktemp('quickdraw_runner')
    cache = root / 'fixture_cache'
    store = create_fixture_cache(cache, categories=16, drawings_per_category=24, seed=31)
    manifest = build_manifest(store, max_points=15, seed=9)
    manifest_path = root / 'manifest.json'
    save_manifest(manifest_path, manifest)
    yield root, cache, manifest_path, store, manifest
    # Several independent runner compilations should not retain device memory
    # for unrelated scientific tests in the same pytest process.
    jax.clear_caches()


def _config(assets, name, *, experiment='a', model_type='ttt_kvb_full', steps=1):
    root, cache, manifest, _, _ = assets
    return {
        'cache_root': str(cache), 'manifest_path': str(manifest),
        'experiment': experiment, 'model_type': model_type, 'seed': 11,
        'batch_size': 1, 'support_count': 1, 'query_count': 1,
        'num_steps': steps, 'checkpoint_every': 1, 'log_every': 1,
        'output_dir': str(root / name),
        'model': {'hidden_dim': 4, 'fast_dim': 2, 'fast_hidden_dim': 3,
                  'mixture_components': 2, 'segment_size': 16,
                  'max_steps': 96 if experiment == 'b2' else 16},
    }


@pytest.fixture(scope='module')
def a_resume_runs(runner_assets):
    uninterrupted = train(_config(runner_assets, 'uninterrupted', steps=3))
    partial_config = _config(runner_assets, 'partial', steps=1)
    partial = train(partial_config)
    resumed_config = _config(runner_assets, 'resumed', steps=3)
    resumed_config['resume_path'] = str(partial / 'last.pkl')
    resumed = train(resumed_config)
    return uninterrupted, partial, resumed


def test_actual_runner_exact_resume_restores_optimizer_and_all_streams(runner_assets, a_resume_runs):
    uninterrupted, partial, resumed = a_resume_runs
    expected = load_checkpoint(uninterrupted / 'last.pkl')
    actual = load_checkpoint(resumed / 'last.pkl')
    assert load_checkpoint(partial / 'last.pkl')['step'] == 1
    assert expected['step'] == actual['step'] == 3
    for field in ('params', 'opt_state', 'rng'):
        _tree_equal(expected[field], actual[field])
    assert expected['extra']['sampler_state'] == actual['extra']['sampler_state']
    assert expected['extra']['exposure'] == actual['extra']['exposure']
    assert set(actual['extra']['sampler_state']['rngs']) == {'category', 'drawing', 'frame', 'start'}
    assert not actual['extra']['transient_fast_state_saved']
    assert 'fast_init' in actual['params'] and 'fast_state' not in actual['params']
    _, _, _, store, manifest = runner_assets
    samplers = [SketchSampler(store, manifest, support_count=1, query_count=1, max_steps=16,
                              seed=seed) for seed in (1, 999)]
    for sampler, payload in zip(samplers, (expected, actual)):
        sampler.load_state_dict(payload['extra']['sampler_state'])
    left, right = (sampler.build_batch(2) for sampler in samplers)
    assert left['meta'] == right['meta']
    _tree_equal(left['support'], right['support'])
    _tree_equal(left['query'], right['query'])


def test_evaluation_episodes_are_reproducible_immutable_and_control_targets_fixed(runner_assets):
    root, cache, manifest_path, store, _ = runner_assets
    paths = [prepare_evaluation(cache, manifest_path, root / f'episodes_{i}',
                                tasks=4, support_count=1, max_steps=16, seed=25)
             for i in range(2)]
    left, left_meta = load_evaluation(paths[0])
    right, right_meta = load_evaluation(paths[1])
    _tree_equal(left, right)
    assert left_meta['records'] == right_meta['records']
    assert left_meta['wrong_records'] == right_meta['wrong_records']
    for target, wrong in zip(left_meta['records'], left_meta['wrong_records']):
        assert target['intended_category'] != wrong['intended_category']
        assert set(target['support_ids']).isdisjoint(target['query_ids'])
        assert store.get(target['query_ids'][0]).category == target['intended_category']
    with pytest.raises(FileExistsError):
        prepare_evaluation(cache, manifest_path, paths[0], tasks=4, support_count=1, max_steps=16)
    metadata_path = paths[1] / 'metadata.json'
    tampered = json.loads(metadata_path.read_text())
    tampered['seed'] += 1
    metadata_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='changed'):
        load_evaluation(paths[1])
    with pytest.raises(ValueError, match='Untouched'):
        prepare_evaluation(cache, manifest_path, root / 'test_unrequested', split='test')


def test_reference_exporter_supplies_frozen_ids_expected_by_metric_boundary(runner_assets):
    root, cache, manifest_path, _, manifest = runner_assets
    groups = []
    for half in ('real_a', 'real_b'):
        output = export_references(cache, manifest_path, root / ('reference_' + half),
                                   half=half, max_steps=16)
        arrays, metadata = load_trajectories(output)
        ids = _reference_ids(metadata)
        expected = {item for category in manifest['categories']['development']
                    for item in manifest['reference_ids'][category][half]}
        assert ids == expected and len(ids) == len(arrays['lengths'])
        groups.append(ids)
    assert groups[0].isdisjoint(groups[1])


def _alternate_private_labels(source: Path, destination: Path) -> Path:
    """Build a distinct fixture artifact with same public inputs, different labels."""
    shutil.copytree(source, destination)
    with np.load(destination / 'episodes.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays['query.tokens'][..., :2] = 20.0 - arrays['query.tokens'][..., :2]
    arrays['query.tokens'][..., 2] = 1.0 - arrays['query.tokens'][..., 2]
    arrays['query.point_mask'][:] = False
    arrays['query.event_mask'][:] = False
    if 'query.knot_mask' in arrays:
        arrays['query.knot_mask'][:] = ~arrays['query.knot_mask']
    np.savez_compressed(destination / 'episodes.npz', **arrays)
    metadata = json.loads((destination / 'metadata.json').read_text())
    del metadata['identifier']
    metadata['data_sha256'] = file_sha256(destination / 'episodes.npz')
    metadata['identifier'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    (destination / 'metadata.json').write_text(json.dumps(metadata))
    return destination


def test_generation_ignores_query_labels_and_condition_order_with_paired_keys(runner_assets, a_resume_runs):
    root, cache, manifest_path, _, _ = runner_assets
    episodes = prepare_evaluation(cache, manifest_path, root / 'label_boundary_episodes',
                                  tasks=2, support_count=1, max_steps=16, seed=67)
    alternate = _alternate_private_labels(episodes, root / 'alternate_private_labels')
    checkpoint = a_resume_runs[0] / 'last.pkl'
    original = generate_run(checkpoint, episodes, root / 'generated_original',
                            conditions=('no_update', 'correct_support'), samples_per_task=3, seed=73)
    changed = generate_run(checkpoint, alternate, root / 'generated_alternate',
                           conditions=('correct_support', 'no_update'), samples_per_task=3, seed=73)
    for condition in ('no_update', 'correct_support'):
        left, left_meta = load_trajectories(original / condition)
        right, right_meta = load_trajectories(changed / condition)
        _tree_equal(left, right)
        assert len(left['lengths']) == 6
        assert left_meta['expected_count'] == right_meta['expected_count'] == 6
        assert [r['query_seed'] for r in left_meta['records']] == [r['query_seed'] for r in right_meta['records']]
        assert left_meta['evaluation_id'] != right_meta['evaluation_id']


@pytest.fixture(scope='module')
def b_runs(runner_assets):
    paths = {}
    for experiment in ('b1', 'b2'):
        config = _config(runner_assets, f'{experiment}_run', experiment=experiment, model_type='no_support')
        paths[experiment] = train(config)
    return paths


def test_b1_pipeline_oracle_and_replay_use_requested_frame_and_original_target(runner_assets, b_runs):
    root, cache, manifest_path, _, _ = runner_assets
    episodes = prepare_evaluation(cache, manifest_path, root / 'b1_episodes', experiment='b1',
                                  tasks=2, support_count=1, query_count=2, max_steps=16, seed=21)
    output = generate_run(b_runs['b1'] / 'last.pkl', episodes, root / 'b1_generated',
                          conditions=('oracle', 'transformed_replay', 'untransformed_replay'),
                          samples_per_task=2, seed=24)
    oracle, oracle_meta = load_trajectories(output / 'oracle')
    replay, replay_meta = load_trajectories(output / 'transformed_replay')
    np.testing.assert_array_equal(oracle['lengths'], replay['lengths'])
    np.testing.assert_allclose(oracle['tokens'], replay['tokens'], atol=2e-7)
    assert oracle['stopped'].all() and replay['stopped'].all()
    for record in oracle_meta['records'] + replay_meta['records']:
        assert record['geometry']['frame_alignment'] == 'none'
        assert record['geometry']['resampled_ordered_point_rmse'] < 2e-7
        assert record['geometry']['endpoint_error'] < 2e-7
        assert record['geometry']['pen_mismatch_fraction'] == 0
        assert record['ordered_tracking']['raw_program_index_xy_rmse'] < 2e-7
        assert record['ordered_tracking']['invalid_pair_count'] == 0
        assert record['stopped']
    _, wrong_frame = load_trajectories(output / 'untransformed_replay')
    assert np.mean([r['geometry']['resampled_ordered_point_rmse'] for r in wrong_frame['records']]) > .01
    sections, evaluation = load_evaluation(episodes)
    np.testing.assert_array_equal(sections['support']['frame'], sections['wrong_support']['frame'])
    for correct, wrong in zip(evaluation['records'], evaluation['wrong_records']):
        assert correct['intended_category'] == wrong['intended_category']
        assert correct['intended_base_id'] != wrong['intended_base_id']


def test_b1_raw_order_metric_rejects_wrong_path_with_same_endpoint_and_stop_failures():
    target = np.asarray([[0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32)
    wrong_order = target[[1, 0, 2, 3]]
    wrong = _b1_ordered_tracking(wrong_order, target, stopped=True)
    assert wrong['endpoint_error'] == 0
    assert wrong['raw_program_index_xy_rmse'] > .5
    assert wrong['time_alignment'] == 'same_program_point_index_no_resampling'
    missing_stop = _b1_ordered_tracking(target, target, stopped=False)
    assert missing_stop['raw_program_index_xy_rmse'] == 0
    assert missing_stop['no_stop_failure'] and missing_stop['invalid_pair_count'] == 1
    truncated = _b1_ordered_tracking(target[:2], target, stopped=True)
    assert truncated['raw_program_index_xy_rmse'] is None and truncated['invalid_pair_count'] == 1


def _executed_tracking_error(arrays, metadata):
    errors, final_errors = [], []
    for index, record in enumerate(metadata['records']):
        mask = arrays['point_mask'][index]
        executed = arrays['tokens'][index, mask, :2]
        target = np.asarray(record['query'][record['query_index']]['reference_absolute'])
        assert len(executed) == len(target), 'Timed-reference replay must preserve target duration.'
        errors.append(np.mean(np.sum((executed - target) ** 2, axis=-1)))
        final_errors.append(np.linalg.norm(executed[-1] - target[-1]))
    return float(np.mean(errors)), float(np.mean(final_errors))


def test_b2_pipeline_replay_is_executed_bounded_and_recovers_perturbations(runner_assets, b_runs):
    root, cache, manifest_path, _, _ = runner_assets
    episodes = prepare_evaluation(cache, manifest_path, root / 'b2_episodes', experiment='b2',
                                  tasks=2, support_count=1, query_count=2, max_steps=96, seed=87)
    sections, evaluation = load_evaluation(episodes)
    np.testing.assert_array_equal(sections['support']['frame'], sections['wrong_support']['frame'])
    np.testing.assert_array_equal(sections['support']['state'][..., 0, :],
                                   sections['wrong_support']['state'][..., 0, :])
    for correct, wrong in zip(evaluation['records'], evaluation['wrong_records']):
        assert correct['intended_category'] == wrong['intended_category']
        assert correct['intended_base_id'] != wrong['intended_base_id']
    with pytest.raises(ValueError, match='re-execution'):
        generate_run(b_runs['b2'] / 'last.pkl', episodes, root / 'invalid_b2_reversal',
                     conditions=('stroke_reversal',))
    assert not (root / 'invalid_b2_reversal').exists()
    conditions = ('oracle', 'open_loop_replay', 'feedback_replay')
    output = generate_run(b_runs['b2'] / 'last.pkl', episodes, root / 'b2_generated',
                          conditions=('no_update', *conditions), samples_per_task=2, seed=91)
    oracle, _ = load_trajectories(output / 'oracle')
    for condition in conditions:
        arrays, metadata = load_trajectories(output / condition)
        np.testing.assert_array_equal(arrays['lengths'], oracle['lengths'])
        np.testing.assert_allclose(arrays['tokens'], oracle['tokens'], atol=3e-7)
        assert arrays['stopped'].all()
        tracking_error, final_error = _executed_tracking_error(arrays, metadata)
        assert tracking_error < 1e-12 and final_error < 3e-7
        for index, record in enumerate(metadata['records']):
            actions = np.asarray(record['issued_actions'])
            states = np.asarray(record['executed_states_before_action'])
            length = arrays['lengths'][index]
            assert len(actions) == len(states) == length + 1
            assert np.linalg.norm(actions[:-1, :2], axis=-1).max() <= .1 + 1e-7
            np.testing.assert_allclose(arrays['tokens'][index, :length, :2],
                                       states[:length, :2] + actions[:length, :2], atol=1e-7)
            assert actions[0, 2] == 0 and actions[-1, 3] == 1
            assert record['execution']['pen_up_travel'] > 0
            assert record['execution']['invalid_actions'] == 0
    network, network_metadata = load_trajectories(output / 'no_update')
    for index, record in enumerate(network_metadata['records']):
        length = network['lengths'][index]
        actions = np.asarray(record['issued_actions'])
        states = np.asarray(record['executed_states_before_action'])
        assert len(actions) == len(states) == length + int(network['stopped'][index])
        np.testing.assert_allclose(network['tokens'][index, :length, :2],
                                   states[:length, :2] + actions[:length, :2], atol=1e-7)
        assert record['execution']['invalid_actions'] == 0
    disturbed = generate_run(b_runs['b2'] / 'last.pkl', episodes, root / 'b2_disturbed',
                             conditions=('open_loop_replay', 'feedback_replay'),
                             samples_per_task=2, seed=91,
                             perturbations={2: np.asarray([.08, -.14], np.float32)})
    disturbed_metadata = load_trajectories(disturbed / 'feedback_replay')[1]
    np.testing.assert_allclose(disturbed_metadata['perturbations']['2'], [.08, -.14])
    open_error = _executed_tracking_error(*load_trajectories(disturbed / 'open_loop_replay'))
    feedback_error = _executed_tracking_error(*load_trajectories(disturbed / 'feedback_replay'))
    assert feedback_error[0] < open_error[0]
    assert feedback_error[1] < open_error[1]


def test_nondefault_motion_bound_controls_training_targets_and_eval_schedule(runner_assets):
    root, cache, manifest_path, store, manifest = runner_assets
    config = _config(runner_assets, 'b2_larger_motion', experiment='b2', model_type='no_support')
    config['model']['motion_bound'] = .2
    run = train(config)
    payload = load_checkpoint(run / 'last.pkl')
    expected_sampler = SketchSampler(
        store, manifest, experiment='b2', split='train', support_count=1, query_count=1,
        max_steps=96, seed=1012, pen_config=PenConfig(max_motion=.2),
    )
    expected_sampler.build_batch(1)
    assert payload['extra']['sampler_state'] == expected_sampler.state_dict()
    kwargs = dict(experiment='b2', tasks=2, support_count=1, max_steps=96, seed=42)
    episodes = prepare_evaluation(cache, manifest_path, root / 'b2_larger_episodes', motion_bound=.2, **kwargs)
    batch, metadata = load_evaluation(episodes)
    assert metadata['motion_bound'] == .2
    commands = batch['query']['tokens'][..., :2][batch['query']['point_mask']]
    magnitudes = np.linalg.norm(commands, axis=-1)
    assert magnitudes.max() <= .2 + 1e-7 and magnitudes.max() > .1
    output = generate_run(run / 'last.pkl', episodes, root / 'b2_larger_oracle',
                          conditions=('oracle',), samples_per_task=1)
    error, final_error = _executed_tracking_error(*load_trajectories(output / 'oracle'))
    assert error < 1e-12 and final_error < 3e-7
    default_episodes = prepare_evaluation(cache, manifest_path, root / 'b2_wrong_bound', **kwargs)
    with pytest.raises(ValueError, match='motion bound'):
        generate_run(run / 'last.pkl', default_episodes, root / 'must_reject_wrong_bound',
                     conditions=('oracle',), samples_per_task=1)
    neural = generate_run(run/'last.pkl',episodes,root/'default_no_support_controls',samples_per_task=1)
    summary = json.loads((neural/'summary.json').read_text())
    assert set(summary['conditions']) == {'correct_support'}
    invalid_output = root/'invalid_no_support_controls'
    with pytest.raises(ValueError,match='TTT checkpoint'):
        generate_run(run/'last.pkl',episodes,invalid_output,
                     conditions=('correct_support','random_update_matched_norm'))
    assert not invalid_output.exists()
