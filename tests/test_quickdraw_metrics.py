from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import metrics


def _artifact():
    # Empty STOP, isolated point, lifted travel, and full-budget missing STOP.
    tokens = np.zeros((4, 5, 4), dtype=np.float32)
    lengths = np.asarray([0, 1, 3, 5])
    stopped = np.asarray([True, True, True, False])
    tokens[0, 0, 3] = 1
    tokens[1, 0, :3] = [0.0, 0.0, 0.0]
    tokens[1, 1, 3] = 1
    tokens[2, :3, :3] = [[-0.5, 0.0, 0], [0.0, 0.5, 1], [0.5, 0.5, 0]]
    tokens[2, 3, 3] = 1
    tokens[3, :, :3] = [[0.0, 0.0, 0], [0.3, 0.2, 1], [1.5, 0.0, 1], [0.2, 0.0, 0], [0.1, 0.0, 1]]
    # Raw garbage in padding must not be exposed to the renderer.
    tokens[0, 2, :] = [9, -9, 0, 0]
    time = np.arange(5)[None, :]
    arrays = {'tokens': tokens, 'lengths': lengths, 'stopped': stopped,
              'point_mask': time < lengths[:, None], 'event_mask': time < (lengths + stopped)[:, None]}
    metadata = {'schema_version': 1, 'coordinate_mode': 'absolute', 'pen_semantics': 'incoming', 'expected_count': 4,
                'records': [{'intended_category': 'fixture', 'sample_id': str(i), 'condition': 'correct', 'support_ids': [], 'seed': i} for i in range(4)]}
    return arrays, metadata


def test_trajectory_artifact_roundtrip_masks_stop_and_counts(tmp_path):
    arrays, metadata = _artifact()
    metrics.save_trajectories(tmp_path, arrays, metadata)
    actual, loaded_metadata = metrics.load_trajectories(tmp_path)
    assert metadata == loaded_metadata
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, actual[key])
    statistics = metrics.trajectory_statistics(actual)
    assert statistics['actual_count'] == 4
    assert statistics['empty_count'] == 1
    assert statistics['no_stop_count'] == 1
    assert statistics['out_of_bounds_sample_count'] == 1
    assert statistics['out_of_bounds_point_count'] == 1
    bad = dict(arrays, event_mask=np.ones_like(arrays['event_mask']))
    with pytest.raises(ValueError, match='event_mask'):
        metrics.validate_trajectories(bad, metadata)
    with pytest.raises(ValueError, match='count'):
        metrics.validate_trajectories(arrays, dict(metadata, expected_count=5))
    arrays['tokens'][2, 1, 3] = 1
    with pytest.raises(ValueError, match='STOP'):
        metrics.validate_trajectories(arrays, metadata)


def test_invalid_outputs_counted_before_clipping():
    arrays, _ = _artifact()
    arrays['tokens'][1, 0, 0] = np.nan
    result = metrics.trajectory_statistics(arrays, np.zeros((4, 64, 64)))
    assert result['invalid_count'] == 1
    assert result['blank_count'] == 4
    assert result['out_of_bounds_point_count'] == 1


def test_fd_float64_symmetry_identical_and_donor_formula_parity():
    linalg = pytest.importorskip('scipy.linalg')
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(40, 6)), rng.normal(size=(33, 6)) + 0.3
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    # The donor compute_fid formula, on a benign full-rank real-valued fixture.
    root = linalg.sqrtm(cx @ cy)
    assert np.max(np.abs(np.imag(root))) < 1e-12
    legacy = np.sum((x.mean(0) - y.mean(0)) ** 2) + np.trace(cx + cy - 2 * root.real)
    assert metrics.sketch_fd(x, y) == pytest.approx(float(legacy), rel=1e-10, abs=1e-10)
    assert metrics.sketch_fd(x, y) == pytest.approx(metrics.sketch_fd(y, x), rel=1e-10, abs=1e-10)
    assert metrics.sketch_fd(x, x) == pytest.approx(0.0, abs=1e-10)
    assert metrics.sketch_fd(np.ones((3, 1)), np.ones((4, 1))) == pytest.approx(0.0)
    with pytest.raises(ValueError, match='n >= 2'):
        metrics.sketch_fd(x[:1], y)
    with pytest.raises(ValueError, match='nonfinite'):
        metrics.sketch_fd(x * np.nan, y)


def test_pooled_permutation_invariance_and_condition_sensitive_score():
    # Each population contains exactly the same two separated feature clusters.
    x = np.asarray([[-2.1, 0], [-2.0, 0.1], [-1.9, -0.1], [1.9, -0.1], [2.0, 0.1], [2.1, 0]])
    labels = ['cat'] * 3 + ['dog'] * 3
    swapped = np.concatenate([x[3:], x[:3]], axis=0)
    assert metrics.sketch_fd(x, x) == pytest.approx(metrics.sketch_fd(swapped, x), abs=1e-8)
    correct = metrics.category_metrics(x, labels, x, labels)
    wrong = metrics.category_metrics(swapped, labels, x, labels)
    assert wrong['macro_sketch_mmd'] > correct['macro_sketch_mmd'] + 10
    # A negative unbiased estimate is valid and remains visible.
    assert correct['macro_sketch_mmd'] < 0
    assert correct['per_category']['cat']['generated_count'] == 3
    with pytest.raises(ValueError, match='populations differ'):
        metrics.category_metrics(x, labels, x, ['cat'] * 6)


def test_fd_rank_deficient_512d_does_not_amplify_roundoff():
    x = np.random.default_rng(0).normal(size=(4, 512))
    y = np.random.default_rng(1).normal(size=(6, 512))
    assert metrics.sketch_fd(x, x) == pytest.approx(0.0, abs=1e-10)
    assert metrics.sketch_fd(x[::-1], x) == pytest.approx(0.0, abs=1e-10)
    assert metrics.sketch_fd(x, y) == pytest.approx(metrics.sketch_fd(y, x), rel=1e-12, abs=1e-10)


def test_geometry_rejects_wrong_frame_and_order_even_for_same_points():
    target = np.asarray([[-0.5, 0, 0], [0, 0.5, 1], [0.5, 0, 1]], dtype=float)
    same = metrics.paired_geometry(target, target)
    moved = target.copy()
    moved[:, :2] += [0.3, 0.1]
    translated = metrics.paired_geometry(moved, target)
    reversed_program = metrics.paired_geometry(target[::-1], target)
    assert same['resampled_ordered_point_rmse'] == 0
    assert translated['resampled_ordered_point_rmse'] > 0.3
    assert reversed_program['resampled_ordered_point_rmse'] > 0.5
    assert reversed_program['symmetric_chamfer_squared'] == 0
    assert reversed_program['pen_mismatch_fraction'] > 0
    assert not metrics.paired_geometry(np.zeros((0, 3)), target)['valid_pair']
    assert same['time_alignment'] == 'secondary_normalized_point_index_interpolation'


def test_raw_timestep_tracking_does_not_hide_time_warp_or_missing_output():
    target = np.asarray([[0, 0, 0], [1, 0, 1], [2, 0, 1]], dtype=float)
    slow = np.asarray([[0, 0, 0], [0.5, 0, 1], [1, 0, 1], [1.5, 0, 1], [2, 0, 1]])
    secondary = metrics.paired_geometry(slow, target)
    assert secondary['resampled_ordered_point_rmse'] == 0
    raw = metrics.timed_tracking_metrics(slow, target, stopped=True)
    assert raw['raw_timestep_xy_rmse'] == pytest.approx(np.sqrt(1.25 / 3))
    assert raw['endpoint_error'] == 0
    assert raw['requested_final_timestep_error'] == 1
    assert raw['extra_step_count'] == 2
    assert raw['absolute_length_error'] == 2
    assert not raw['valid_pair']
    partial = metrics.timed_tracking_metrics(target[:1], target, perturbation_steps=[0, 1], recovery_window=2, stopped=True)
    assert partial['observed_overlap_xy_rmse'] == 0
    assert partial['raw_timestep_xy_rmse'] is None
    assert partial['missing_step_fraction'] == pytest.approx(2 / 3)
    assert partial['completion_fraction'] == pytest.approx(1 / 3)
    assert partial['invalid_pair_count'] == 1
    assert partial['requested_final_timestep_error'] is None
    assert partial['recovery_windows'][0]['missing_step_count'] == 1
    assert partial['recovery_windows'][1]['raw_timestep_xy_rmse'] is None
    assert not partial['recovery_windows'][1]['valid_window']
    empty = metrics.timed_tracking_metrics(np.zeros((0, 3)), target, stopped=True)
    assert empty['invalid_pair_count'] == 1
    assert empty['missing_step_fraction'] == 1
    assert empty['observed_overlap_xy_rmse'] is None


def test_tracking_recovery_pen_runs_invalid_states_and_missing_stop():
    target = np.column_stack([np.arange(8) * 0.1, np.zeros(8), [0, 0, 0, 1, 1, 0, 0, 1]])
    generated = target.copy()
    generated[2:5, 1] = [0.4, 0.2, 0.0]
    raw = metrics.timed_tracking_metrics(generated, target, perturbation_steps=[2, 7, 12], recovery_window=3, stopped=True)
    assert raw['valid_pair']
    assert raw['generated_pen_down_runs'] == 2  # Five pen-up points are not five strokes.
    assert raw['pen_down_run_count_error'] == 0
    recovery = raw['recovery_windows'][0]
    assert recovery['valid_window']
    assert recovery['raw_timestep_xy_rmse'] == pytest.approx(np.sqrt(0.2 / 3))
    assert recovery['initial_error'] == pytest.approx(0.4)
    assert recovery['final_error'] == 0
    assert raw['recovery_windows'][1]['target_step_count'] == 1
    assert raw['recovery_windows'][2]['outside_target_schedule']
    assert not raw['recovery_windows'][2]['valid_window']
    missing_stop = metrics.timed_tracking_metrics(target, target, stopped=np.bool_(False))
    assert missing_stop['raw_timestep_xy_rmse'] == 0
    assert missing_stop['no_stop_failure']
    assert missing_stop['invalid_pair_count'] == 1
    generated[3, 0] = np.nan
    invalid = metrics.timed_tracking_metrics(generated, target, perturbation_steps=[2], stopped=True)
    assert invalid['raw_timestep_xy_rmse'] is None
    assert invalid['invalid_step_count'] == 1
    assert invalid['invalid_generated_step_count'] == 1
    assert not invalid['recovery_windows'][0]['valid_window']
    assert invalid['recovery_windows'][0]['invalid_step_count'] == 1


def _save_features(path, x, prefix, *, category='cat', support_ids=()):
    path.mkdir()
    x = np.pad(x, [(0, 0), (0, 512 - x.shape[1])])
    np.savez(path / 'features.npz', features=x)
    metadata = {
        'feature_version': metrics.FEATURE_VERSION, 'extractor_sha256': 'test-fixture-original-hash',
        'feature_dimension': 512, 'feature_normalization': 'none',
        'renderer_config': metrics.RENDERER_CONFIG, 'source_hashes': {'resnet18': 'a', 'rasterizer': 'b'},
        'actual_count': len(x),
        'records': [{'drawing_id': f'{prefix}-{i}', 'intended_category': category, 'support_ids': list(support_ids)} for i in range(len(x))],
    }
    (path / 'metadata.json').write_text(json.dumps(metadata))


def test_score_reference_governance_and_repeated_calibration(tmp_path):
    rng = np.random.default_rng(3)
    paths = [tmp_path / name for name in ('generated', 'reference', 'repeat1', 'repeat2')]
    for i, path in enumerate(paths):
        _save_features(path, rng.normal(size=(8, 2)), f'population{i}')
    result = metrics.score_feature_artifacts(paths[0], paths[1], reference_repeats=paths[2:])
    assert result['generated_count'] == 8
    assert result['reference_count'] == 8
    assert len(result['real_real']) == 2
    assert result['real_real_fd_std'] is not None
    with pytest.raises(ValueError, match='disjoint'):
        metrics.score_feature_artifacts(paths[0], paths[1], reference_repeats=[paths[1]])
    with pytest.raises(ValueError, match='identical reference'):
        metrics.score_feature_artifacts(paths[0], paths[1], reference_repeats=[paths[2], paths[2]])
    metadata = json.loads((paths[0] / 'metadata.json').read_text())
    metadata['records'][0]['support_ids'] = ['population1-0']
    (paths[0] / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='independent'):
        metrics.score_feature_artifacts(paths[0], paths[1])
    metadata['feature_normalization'] = 'l2'
    (paths[0] / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='unnormalized'):
        metrics.load_features(paths[0])


def test_near_copy_distances_are_not_normalized():
    x = np.asarray([[1, 0], [0, 2]], dtype=float)
    y = np.asarray([[1, 0], [0, 1]], dtype=float)
    np.testing.assert_allclose(metrics.nearest_feature_distance(x, y), [0, 1])


def test_copying_uses_actual_prompt_ids_permitted_training_and_separate_sequence_hashes(tmp_path):
    # Synthetic vectors test numerical/accounting helpers only, not real scores.
    generated, supports, training = [tmp_path / name for name in ('generated', 'supports', 'training')]
    _save_features(generated, np.asarray([[0, 0], [3, 0], [0, 0], [3, 0]], float), 'g')
    _save_features(supports, np.asarray([[0, 0], [3, 0]], float), 's')
    _save_features(training, np.asarray([[0, 0], [9, 0]], float), 't')
    manifest = {'a_ids': {'train': {'cat': ['t-0', 't-1']}}}
    manifest['identifier'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    for path in (generated, supports, training):
        metadata = json.loads((path / 'metadata.json').read_text())
        metadata['manifest_id'] = manifest['identifier']
        for index, record in enumerate(metadata['records']):
            record['trajectory_sha256'] = ['shape-a', 'shape-b'][index % 2]
            if path == generated:
                record.update(support_ids=['s-1'], condition='wrong_support', task_index=0,
                              query_index=0, sample_index=index)
        (path / 'metadata.json').write_text(json.dumps(metadata))
    result = metrics.score_copying(generated, supports, training, manifest_path=manifest_path, max_pairs_per_prompt=2, seed=4)
    assert result['per_generated'][0]['nearest_support_id'] == 's-1'
    assert result['per_generated'][0]['nearest_support_feature_distance'] == 3
    assert result['zero_support_feature_distance_count'] == 2
    assert result['exact_support_point_sequence_count'] == 2
    assert result['zero_training_feature_distance_count'] == 2
    assert result['exact_training_point_sequence_count'] == 4  # Hashes and feature coincidence are distinct diagnostics.
    assert result['within_prompt_diversity'][0]['available_pair_count'] == 6
    assert result['within_prompt_diversity'][0]['evaluated_pair_count'] == 2
    assert result == metrics.score_copying(generated, supports, training, manifest_path=manifest_path, max_pairs_per_prompt=2, seed=4)
    changed = json.loads((generated / 'metadata.json').read_text())
    changed['records'][0]['support_ids'] = ['not-exported']
    (generated / 'metadata.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='Shown support feature IDs are missing'):
        metrics.score_copying(generated, supports, training, manifest_path=manifest_path)
    changed['records'][0]['support_ids'] = ['s-1']
    changed['records'][0]['condition'] = 'action_corruption'
    (generated / 'metadata.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='actual supplied trajectories'):
        metrics.score_copying(generated, supports, training, manifest_path=manifest_path)
    changed['records'][0]['condition'] = 'wrong_support'
    (generated / 'metadata.json').write_text(json.dumps(changed))
    forbidden = json.loads((training / 'metadata.json').read_text())
    forbidden['records'][0]['drawing_id'] = 'heldout-program'
    (training / 'metadata.json').write_text(json.dumps(forbidden))
    with pytest.raises(ValueError, match='outside permitted'):
        metrics.score_copying(generated, supports, training, manifest_path=manifest_path)


def test_worker_missing_checkpoint_fails_before_torch_import(tmp_path):
    worker = Path(metrics.__file__).with_name('metrics_worker.py')
    result = subprocess.run([sys.executable, str(worker), '--artifact', str(tmp_path), '--output', str(tmp_path / 'out'), '--donor-root', str(tmp_path), '--extractor-checkpoint', str(tmp_path / 'missing.pt')], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'QuickDraw evaluator checkpoint is missing' in result.stderr
    assert 'trained grayscale 345-class ResNet18 checkpoint' in result.stderr


def _load_renderer():
    pytest.importorskip('PIL')
    donor = Path(os.environ.get('QUICKDRAW_DONOR_ROOT', Path(__file__).resolve().parents[2] / 'quick-robot-draw'))
    path = donor / 'dataset/rasterize.py'
    if not path.exists():
        pytest.skip('Donor rasterizer unavailable; set QUICKDRAW_DONOR_ROOT for parity')
    spec = importlib.util.spec_from_file_location('_test_quickdraw_donor_rasterizer', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_donor_raster_roundtrip_exact_pen_stop_and_padding(tmp_path):
    renderer = _load_renderer()
    arrays, metadata = _artifact()
    config = renderer.RasterizerConfig(**metrics.RENDERER_CONFIG)
    direct = [renderer.rasterize_absolute_points(tokens[mask, :3], config=config) for tokens, mask in zip(arrays['tokens'], arrays['point_mask'])]
    metrics.save_trajectories(tmp_path, arrays, metadata)
    restored, _ = metrics.load_trajectories(tmp_path)
    images = [renderer.rasterize_absolute_points(tokens[mask, :3], config=config) for tokens, mask in zip(restored['tokens'], restored['point_mask'])]
    np.testing.assert_array_equal(direct, images)
    assert not np.any(images[0])
    assert np.any(images[1])  # A legitimate isolated point is not blank.
    assert config.line_width == 2.0
    lifted = np.asarray([[-0.8, 0, 0], [0.8, 0, 0]], dtype=np.float32)
    joined = lifted.copy()
    joined[1, 2] = 1
    left = renderer.rasterize_absolute_points(lifted, config=config)
    right = renderer.rasterize_absolute_points(joined, config=config)
    assert left[32, 32] == 0
    assert right[32, 32] > 0


def test_original_checkpoint_feature_parity_requires_real_asset(tmp_path):
    checkpoint = os.environ.get('QUICKDRAW_EVALUATOR_CHECKPOINT')
    python = os.environ.get('QUICKDRAW_METRIC_PYTHON')
    if not checkpoint or not python:
        pytest.skip('Original evaluator parity needs QUICKDRAW_EVALUATOR_CHECKPOINT and QUICKDRAW_METRIC_PYTHON; no random substitute')
    donor = Path(os.environ.get('QUICKDRAW_DONOR_ROOT', Path(__file__).resolve().parents[2] / 'quick-robot-draw'))
    arrays, metadata = _artifact()
    artifact, output = tmp_path / 'trajectories', tmp_path / 'features'
    metrics.save_trajectories(artifact, arrays, metadata)
    metrics.extract_features(artifact, output, python=python, donor_root=donor, extractor_checkpoint=checkpoint, batch_size=3)
    with np.load(output / 'features.npz') as data:
        features = data['features']
    # Direct donor constructor parity uses a CPU copy of the same state dict;
    # this changes serialization placement, never tensor values or architecture.
    script = '''
import importlib.util, pathlib, sys, tempfile
import numpy as np
import torch
path, checkpoint, raster_file, output = sys.argv[1:]
spec = importlib.util.spec_from_file_location('donor_resnet_parity', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as temporary:
    cpu_checkpoint = pathlib.Path(temporary) / 'cpu.pt'
    torch.save(torch.load(checkpoint, map_location='cpu', weights_only=True), cpu_checkpoint)
    model = module.ResNet18FeatureExtractor(cpu_checkpoint).eval()
    with np.load(raster_file) as archive:
        images = torch.from_numpy(archive['rasters'][:, None])
    with torch.inference_mode():
        features = model(images).numpy()
    np.save(output, features)
'''
    direct = tmp_path / 'direct.npy'
    subprocess.run([python, '-c', script, str(donor / 'metrics/resnet18.py'), str(checkpoint), str(output / 'rasters.npz'), str(direct)], check=True)
    np.testing.assert_allclose(features, np.load(direct), atol=2e-5, rtol=2e-5)
