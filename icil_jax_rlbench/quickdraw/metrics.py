"""Frozen sketch metric artifacts and NumPy analysis; never imports PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
FEATURE_VERSION = 'quickdraw_resnet18_raw_pool512_v1'
FD_VERSION = 'float64_psd_factor_nuclear_v1'
KERNEL_CONFIG = {'name': 'polynomial', 'degree': 3, 'offset': 1.0, 'scale': '1/d'}
RENDERER_CONFIG = {
    'img_size': 64,
    'antialias': 2,
    'line_width': 2.0,
    'background_value': 0.0,
    'stroke_value': 1.0,
    'normalize_inputs': False,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trajectories(arrays: Mapping, metadata: Mapping) -> None:
    """Validate compact point prefixes, separate STOP events, and padded storage.

    Nonfinite coordinates remain legal raw outputs so failures can be counted.
    They are rendered as explicit blank invalid samples by the metric worker.
    """
    if metadata.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported QuickDraw trajectory schema_version')
    if metadata.get('coordinate_mode') != 'absolute':
        raise ValueError('Metric trajectories must use absolute coordinates')
    if metadata.get('pen_semantics') != 'incoming':
        raise ValueError('Metric trajectories must use incoming pen semantics')
    tokens = np.asarray(arrays['tokens'])
    if tokens.ndim != 3 or tokens.shape[-1] != 4:
        raise ValueError('tokens must have shape [samples, time, 4]')
    n, t, _ = tokens.shape
    lengths = np.asarray(arrays['lengths'])
    points = np.asarray(arrays['point_mask'])
    events = np.asarray(arrays['event_mask'])
    stopped = np.asarray(arrays['stopped'])
    if lengths.shape != (n,) or stopped.shape != (n,):
        raise ValueError('lengths and stopped must have shape [samples]')
    if points.shape != (n, t) or events.shape != (n, t):
        raise ValueError('point_mask and event_mask must have shape [samples, time]')
    if lengths.dtype.kind not in 'iu' or np.any(lengths < 0) or np.any(lengths > t):
        raise ValueError('lengths must contain bounded integer point counts')
    for name, mask in [('point_mask', points), ('event_mask', events), ('stopped', stopped)]:
        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError(f'{name} must be binary')
    points, events, stopped = points.astype(bool), events.astype(bool), stopped.astype(bool)
    time = np.arange(t)[None, :]
    if not np.array_equal(points, time < lengths[:, None]):
        raise ValueError('point_mask must select exactly the first lengths points')
    if np.any(stopped & (lengths >= t)):
        raise ValueError('STOP needs a separate event slot after the last point')
    expected_events = time < (lengths + stopped)[:, None]
    if not np.array_equal(events, expected_events):
        raise ValueError('event_mask must select points and the optional first STOP')
    selected_flags = tokens[..., 2:4][events]
    if not np.all(np.isin(selected_flags, [0.0, 1.0])):
        raise ValueError('Selected incoming pen and STOP flags must be binary')
    if np.any(tokens[..., 3][points] != 0):
        raise ValueError('Point events cannot also be STOP events')
    if np.any(tokens[..., 3][events & ~points] != 1):
        raise ValueError('Non-point events must be STOP events')
    if len(metadata.get('records', [])) != n:
        raise ValueError('metadata must contain exactly one record per sample')
    if 'expected_count' in metadata and metadata['expected_count'] != n:
        raise ValueError('Actual sample count differs from requested expected_count')
    for record in metadata['records']:
        if 'intended_category' not in record:
            raise ValueError('Every sample needs intended_category metadata')


def save_trajectories(path: str | Path, arrays: Mapping, metadata: Mapping) -> Path:
    validate_trajectories(arrays, metadata)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / 'trajectories.npz', **arrays)
    (path / 'metadata.json').write_text(json.dumps(dict(metadata), indent=2) + '\n')
    return path


def load_trajectories(path: str | Path) -> tuple[dict, dict]:
    path = Path(path)
    metadata = json.loads((path / 'metadata.json').read_text())
    with np.load(path / 'trajectories.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_trajectories(arrays, metadata)
    return arrays, metadata


def trajectory_statistics(arrays: Mapping, rasters: np.ndarray | None = None) -> dict:
    tokens = np.asarray(arrays['tokens'])
    mask = np.asarray(arrays['point_mask'], dtype=bool)
    xy = tokens[..., :2]
    finite = np.all(np.isfinite(xy), axis=-1)
    invalid = np.any(mask & ~finite, axis=1)
    outside = mask & finite & np.any(np.abs(xy) > 1.0, axis=-1)
    n = len(tokens)
    count = int(mask.sum())
    result = {
        'actual_count': n,
        'point_count': count,
        'invalid_count': int(invalid.sum()),
        'invalid_fraction': float(invalid.mean()) if n else 0.0,
        'empty_count': int(np.sum(np.asarray(arrays['lengths']) == 0)),
        'no_stop_count': int(np.sum(~np.asarray(arrays['stopped'], dtype=bool))),
        'no_stop_fraction': float(np.mean(~np.asarray(arrays['stopped'], dtype=bool))) if n else 0.0,
        'out_of_bounds_sample_count': int(np.any(outside, axis=1).sum()),
        'out_of_bounds_point_count': int(outside.sum()),
        'out_of_bounds_point_fraction': float(outside.sum() / max(1, count)),
    }
    if rasters is not None:
        if len(rasters) != n:
            raise ValueError('Raster count differs from trajectory count')
        blank = ~np.any(np.asarray(rasters) > 0, axis=(1, 2))
        result.update(blank_count=int(blank.sum()), blank_fraction=float(blank.mean()) if n else 0.0)
    return result


def _features(value: np.ndarray, *, minimum: int = 2) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < minimum or value.shape[1] == 0:
        raise ValueError(f'Features must have shape [n >= {minimum}, dimension > 0]')
    if not np.all(np.isfinite(value)):
        raise ValueError('Features contain nonfinite values')
    return value


def _psd_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues.min() < -tolerance:
        raise ValueError('Covariance has a materially negative eigenvalue')
    return np.maximum(eigenvalues, 0.0), eigenvectors


def sketch_fd(generated: np.ndarray, reference: np.ndarray) -> float:
    """Unnormalized pooled-feature FD; float64 covariance-factor nuclear norm.

    Tiny negative numerical results are retained. Material negative results fail.
    This is a versioned numerical implementation of the donor's Gaussian FD.
    """
    x, y = _features(generated), _features(reference)
    if x.shape[1] != y.shape[1]:
        raise ValueError('Feature dimensions differ')
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    centered_x = (x - mu_x) / np.sqrt(len(x) - 1)
    centered_y = (y - mu_y) / np.sqrt(len(y) - 1)
    if len(x) * len(y) <= x.shape[1] ** 2:
        # Nonzero singular values equal those of covariance-root products.
        # Raw factors avoid amplifying roundoff in rank-deficient covariances.
        product = centered_x @ centered_y.T
    else:
        cx, cy = centered_x.T @ centered_x, centered_y.T @ centered_y
        eig_x, vec_x = _psd_eigh(cx)
        eig_y, vec_y = _psd_eigh(cy)
        root_x = (vec_x * np.sqrt(eig_x)) @ vec_x.T
        root_y = (vec_y * np.sqrt(eig_y)) @ vec_y.T
        product = root_x @ root_y
    fidelity = float(np.linalg.svd(product, compute_uv=False).sum())
    mean_distance = float(np.sum((mu_x - mu_y) ** 2))
    covariance_trace = float(np.sum(centered_x ** 2) + np.sum(centered_y ** 2))
    score = mean_distance + covariance_trace - 2.0 * fidelity
    tolerance = 1e-7 * max(1.0, covariance_trace, mean_distance)
    if not np.isfinite(score) or score < -tolerance:
        raise ValueError(f'Unstable Sketch-FD numerical result: {score}')
    return float(score)


def sketch_mmd(generated: np.ndarray, reference: np.ndarray) -> float:
    """Unbiased degree-three polynomial MMD; negative estimates are valid."""
    x, y = _features(generated), _features(reference)
    if x.shape[1] != y.shape[1]:
        raise ValueError('Feature dimensions differ')
    dimension = x.shape[1]

    def kernel_sum(left, right):
        total = 0.0
        for i in range(0, len(left), 512):
            for j in range(0, len(right), 512):
                total += float(np.sum((1.0 + left[i:i + 512] @ right[j:j + 512].T / dimension) ** 3))
        return total

    diagonal_x = np.sum((1.0 + np.sum(x * x, axis=1) / dimension) ** 3)
    diagonal_y = np.sum((1.0 + np.sum(y * y, axis=1) / dimension) ** 3)
    n, m = len(x), len(y)
    score = (kernel_sum(x, x) - diagonal_x) / (n * (n - 1)) + (kernel_sum(y, y) - diagonal_y) / (m * (m - 1)) - 2.0 * kernel_sum(x, y) / (n * m)
    if not np.isfinite(score):
        raise ValueError('Nonfinite Sketch-MMD; check feature scale')
    return float(score)


def category_metrics(generated: np.ndarray, generated_categories: Sequence[str], reference: np.ndarray, reference_categories: Sequence[str]) -> dict:
    x, y = _features(generated), _features(reference)
    labels_x, labels_y = np.asarray(generated_categories, dtype=str), np.asarray(reference_categories, dtype=str)
    if labels_x.shape != (len(x),) or labels_y.shape != (len(y),):
        raise ValueError('Category label and feature counts differ')
    categories = sorted(set(labels_x))
    if set(categories) != set(labels_y):
        raise ValueError('Generated and reference category populations differ')
    details = {}
    for category in categories:
        xs, ys = x[labels_x == category], y[labels_y == category]
        if min(len(xs), len(ys)) < 2:
            raise ValueError(f'Category {category!r} needs at least two generated and two reference features; got {len(xs)} and {len(ys)}')
        details[category] = {
            'generated_count': len(xs), 'reference_count': len(ys),
            'sketch_mmd': sketch_mmd(xs, ys),
            'sketch_fd': sketch_fd(xs, ys),
            'covariance_rank_limited': min(len(xs), len(ys)) <= x.shape[1],
        }
    return {
        'macro_sketch_mmd': float(np.mean([item['sketch_mmd'] for item in details.values()])),
        'category_weighting': 'uniform_macro',
        'kernel': KERNEL_CONFIG,
        'per_category': details,
    }


def neighborhood_metrics(generated, generated_records, reference, reference_records) -> dict:
    """Unnormalized-feature polynomial MMD, with equal weight per local task.

    A task may contain several support sets and generated repetitions. Those
    repetitions provide distribution samples, not independent training runs.
    Tiny local reservoirs never receive a covariance-based FD estimate here.
    """
    x, y = _features(generated), _features(reference)
    if len(generated_records) != len(x) or len(reference_records) != len(y):
        raise ValueError('Neighborhood record and feature counts differ.')
    def labels(records):
        if any(not row.get('intended_neighborhood_id') for row in records):
            raise ValueError('Every local sample requires its intended_neighborhood_id.')
        return np.asarray([str(row['intended_neighborhood_id']) for row in records])
    lx, ly = labels(generated_records), labels(reference_records)
    if set(lx) != set(ly):
        raise ValueError('Generated and reference neighborhood populations differ.')
    details = {}
    for task in sorted(set(lx)):
        xi, yi = np.flatnonzero(lx == task), np.flatnonzero(ly == task)
        categories = {str(generated_records[i]['intended_category']) for i in xi}
        categories.update(str(reference_records[i]['intended_category']) for i in yi)
        if len(categories) != 1:
            raise ValueError('A neighborhood must retain the same intended category in all controls.')
        if min(len(xi), len(yi)) < 2:
            raise ValueError(f'Neighborhood {task!r} needs at least two generated and reference samples.')
        details[task] = {
            'intended_category': categories.pop(), 'generated_count': len(xi),
            'reference_count': len(yi), 'sketch_mmd': sketch_mmd(x[xi], y[yi]),
            'support_set_count': len({tuple(generated_records[i].get('support_ids', ())) for i in xi}),
        }
    return {
        'macro_sketch_mmd': float(np.mean([row['sketch_mmd'] for row in details.values()])),
        'task_weighting': 'uniform_macro', 'kernel': KERNEL_CONFIG,
        'per_neighborhood': details, 'neighborhood_count': len(details),
        'uncertainty_unit': 'neighborhood; overlapping neighborhoods are not independent tasks',
        'feature_normalization': 'none',
    }


def same_category_swap_check(real_features, reference_features, *, left_task, right_task) -> dict:
    """Validate sensitivity on two fixed, disjoint real neighborhood reservoirs.

    This operates on already frozen offline features. It never chooses regions
    by generated output or changes the selector to improve a model's score.
    """
    x, mx = load_features(real_features)
    y, my = load_features(reference_features)
    _compatible(mx, my)
    if mx.get('manifest_id') != my.get('manifest_id'):
        raise ValueError('Swap validation requires one frozen manifest.')
    if _reference_ids(mx) & _reference_ids(my):
        raise ValueError('Swap validation requires disjoint real reference drawings.')
    tasks = {str(left_task), str(right_task)}
    if len(tasks) != 2:
        raise ValueError('Swap validation requires two distinct neighborhoods.')
    ix = [i for i, row in enumerate(mx['records']) if row.get('intended_neighborhood_id') in tasks]
    iy = [i for i, row in enumerate(my['records']) if row.get('intended_neighborhood_id') in tasks]
    rx, ry = [mx['records'][i] for i in ix], [my['records'][i] for i in iy]
    if any(row.get('condition') != 'independent_real_reference' for row in rx + ry):
        raise ValueError('Swap validation requires reserved real-drawing artifacts, never model generations.')
    if len({row['intended_category'] for row in rx + ry}) != 1:
        raise ValueError('Swap validation requires neighborhoods of the same category.')
    x, y = x[ix], y[iy]
    correct = neighborhood_metrics(x, rx, y, ry)
    left = [i for i, row in enumerate(rx) if row['intended_neighborhood_id'] == str(left_task)]
    right = [i for i, row in enumerate(rx) if row['intended_neighborhood_id'] == str(right_task)]
    if len(left) != len(right):
        raise ValueError('A paired real-set swap requires equal sample counts in both neighborhoods.')
    permutation = np.arange(len(x))
    permutation[left], permutation[right] = right, left
    swapped = neighborhood_metrics(x[permutation], rx, y, ry)
    pooled = sketch_fd(x, y)
    swapped_pooled = sketch_fd(x[permutation], y)
    synthetic = any(row.get('synthetic_fixture', False) for row in rx + ry)
    provenance = mx.get('extractor_training_provenance', {})
    provenance = provenance if isinstance(provenance, dict) else {}
    fixture = (synthetic or bool(provenance.get('synthetic_fixture')) or bool(provenance.get('smoke_only'))
               or bool(provenance.get('cache_partition_provenance', {}).get('allow_subset_fixture')))
    gap = swapped['macro_sketch_mmd'] - correct['macro_sketch_mmd']
    return {
        'left_task': str(left_task), 'right_task': str(right_task),
        'pooled_fd_correct': pooled, 'pooled_fd_swapped': swapped_pooled,
        'pooled_fd_invariant': bool(np.isclose(pooled, swapped_pooled, rtol=1e-10, atol=1e-8)),
        'conditional_correct': correct, 'conditional_swapped': swapped,
        'conditional_mmd_increase': float(gap), 'detects_swap': bool(gap > 0),
        'validation_scope': 'synthetic_or_smoke_software_check' if fixture else 'fixed_real_reference_features',
        'classifier_training_provenance': provenance or 'unknown',
        'extractor_sha256': mx['extractor_sha256'], 'manifest_id': mx.get('manifest_id'),
        'reference_ids': sorted(_reference_ids(my)), 'real_ids': sorted(_reference_ids(mx)),
        'selection': 'caller-declared same-category task pair; no model outputs used',
    }


def geometry_descriptor(points, *, samples=16):
    """Fixed raw-coordinate/stroke descriptor, without a learned representation."""
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        return None
    index = np.linspace(0, len(points) - 1, samples)
    xy = np.stack([np.interp(index, np.arange(len(points)), points[:, i]) for i in range(2)], axis=1)
    return np.r_[xy.reshape(-1), len(points) / 128.0, np.sum(points[:, 2] < .5) / 16.0]


def score_geometry_copying(generated, supports, training, *, manifest_path):
    """Raw XY/stroke proximity to actual supports and permitted training data.

    This complements frozen-classifier distances. Descriptor distances are
    explicit diagnostics, not a calibrated near-duplicate decision rule.
    """
    gx, gm = load_trajectories(generated)
    sx, sm = load_trajectories(supports)
    tx, tm = load_trajectories(training)
    manifest = _load_metric_manifest(manifest_path)
    if any(meta.get('manifest_id') != manifest['identifier'] for meta in (gm, sm, tm)):
        raise ValueError('Geometry-copy artifacts require the supplied frozen manifest.')
    if gm.get('experiment', 'a') != 'a':
        raise ValueError('Canonical geometry-copy populations are defined for A only.')
    permitted = {str(item) for ids in manifest['a_ids']['train'].values() for item in ids}
    train_ids = [str(row['drawing_id']) for row in tm['records']]
    support_ids = [str(row['drawing_id']) for row in sm['records']]
    if (not set(train_ids) <= permitted or len(set(train_ids)) != len(train_ids)
            or len(set(support_ids)) != len(support_ids)):
        raise ValueError('Geometry-copy references require unique support IDs and permitted training IDs.')
    def descriptors(arrays):
        return [geometry_descriptor(tokens[mask, :3]) for tokens, mask
                in zip(arrays['tokens'], arrays['point_mask'])]
    tv, sv, gv = descriptors(tx), descriptors(sx), descriptors(gx)
    if any(row is None for row in tv + sv):
        raise ValueError('Real geometry-copy reservoirs must contain finite, nonempty drawings.')
    train_values, support_values = np.stack(tv), np.stack(sv)
    support_index = {item: i for i, item in enumerate(support_ids)}
    rows = []
    for i, (value, record) in enumerate(zip(gv, gm['records'])):
        if record.get('condition') in ('action_corruption', 'stroke_reversal'):
            raise ValueError('Transformed support geometry requires exported actual shown executions.')
        shown = list(map(str, record.get('support_ids', [])))
        if not set(shown) <= set(support_index):
            raise ValueError('Actual shown support geometry IDs are missing from the supplied population.')
        result = {'sample_index': i, 'intended_category': record['intended_category'],
                  'intended_neighborhood_id': record.get('intended_neighborhood_id'),
                  'valid_nonempty_output': value is not None, 'shown_support_ids': shown}
        if value is not None:
            best = (float('inf'), -1)
            for start in range(0, len(train_values), 4096):
                distance = np.linalg.norm(train_values[start:start + 4096] - value, axis=1)
                index = int(np.argmin(distance))
                best = min(best, (float(distance[index]), start + index))
            result.update(nearest_training_geometry_distance=best[0], nearest_training_id=train_ids[best[1]])
            if shown:
                distance = np.linalg.norm(support_values[[support_index[item] for item in shown]] - value, axis=1)
                index = int(np.argmin(distance))
                result.update(nearest_support_geometry_distance=float(distance[index]), nearest_support_id=shown[index])
        rows.append(result)
    return {
        'metric': 'raw_trajectory_geometry_copying_v1', 'manifest_id': manifest['identifier'],
        'descriptor': '16 point-index interpolated raw XY + point_count/128 + stroke_start_count/16; Euclidean norm',
        'normalization': 'no feature classifier, frame fit, or geometric alignment',
        'generated_count': len(rows), 'invalid_or_empty_count': sum(not row['valid_nonempty_output'] for row in rows),
        'training_reservoir_count': len(train_ids), 'shown_support_reservoir_count': len(support_ids),
        'per_generated': rows, 'near_duplicate_threshold': None,
        'interpretation': 'raw coordinate/order/stroke diagnostic; feature-distance circularity check',
    }


def paired_geometry(generated: np.ndarray, target: np.ndarray) -> dict:
    """Secondary shape/order errors with explicit normalized-index resampling.

    Stroke restarts use the original drawing's incoming-pen convention, including
    singleton strokes. Use timed_tracking_metrics for B2 execution and pen runs.
    """
    x, y = np.asarray(generated, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] < 3 or y.shape[1] < 3:
        raise ValueError('Geometry requires [points, x/y/incoming_pen] arrays')
    if len(x) == 0 or len(y) == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {'valid_pair': False, 'generated_count': len(x), 'target_count': len(y)}
    count = max(len(x), len(y))
    grid = np.linspace(0.0, 1.0, count)
    xi = np.stack([np.interp(grid, np.linspace(0, 1, len(x)), x[:, axis]) for axis in range(2)], axis=1)
    yi = np.stack([np.interp(grid, np.linspace(0, 1, len(y)), y[:, axis]) for axis in range(2)], axis=1)
    distances = np.sum((x[:, None, :2] - y[None, :, :2]) ** 2, axis=-1)
    pen_x = x[np.rint(grid * (len(x) - 1)).astype(int), 2] >= 0.5
    pen_y = y[np.rint(grid * (len(y) - 1)).astype(int), 2] >= 0.5
    return {
        'valid_pair': True,
        'generated_count': len(x), 'target_count': len(y),
        'resampled_ordered_point_rmse': float(np.sqrt(np.mean(np.sum((xi - yi) ** 2, axis=-1)))),
        'symmetric_chamfer_squared': float(0.5 * (distances.min(axis=0).mean() + distances.min(axis=1).mean())),
        'endpoint_error': float(np.linalg.norm(x[-1, :2] - y[-1, :2])),
        'pen_mismatch_fraction': float(np.mean(pen_x != pen_y)),
        'stroke_count_error': int(abs(np.sum(x[:, 2] < 0.5) - np.sum(y[:, 2] < 0.5))),
        'frame_alignment': 'none',
        'time_alignment': 'secondary_normalized_point_index_interpolation',
        'stroke_count_convention': 'drawing_restart_points_including_singletons',
    }


def timed_tracking_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    perturbation_steps: Sequence[int] = (),
    recovery_window: int = 5,
    *,
    stopped: bool | None = None,
) -> dict:
    """B2 tracking at the same raw timestep, with no geometric/time alignment.

    Primary RMSE requires every target timestep to have a finite output. The
    separately named overlap RMSE is diagnostic and cannot establish completion.
    Pass stopped when available: positions alone cannot identify a missing STOP.
    Recovery windows are [perturbation step, step + recovery_window), clipped to
    the fixed target schedule, and explicitly invalid when outputs are missing.
    """
    x, y = np.asarray(generated, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != 3 or y.shape[1] != 3:
        raise ValueError('Tracking requires [steps, x/y/incoming_pen] arrays')
    if not isinstance(recovery_window, (int, np.integer)) or recovery_window < 1:
        raise ValueError('recovery_window must be a positive integer')
    steps = list(perturbation_steps)
    if any(not isinstance(step, (int, np.integer)) or step < 0 for step in steps):
        raise ValueError('Perturbation steps must be nonnegative integers')
    if len(set(steps)) != len(steps):
        raise ValueError('Perturbation steps must be distinct')
    n, m = len(x), len(y)
    overlap = min(n, m)
    present = np.arange(m) < n
    valid = np.zeros(m, dtype=bool)
    valid[:overlap] = np.all(np.isfinite(x[:overlap]), axis=1) & np.all(np.isfinite(y[:overlap]), axis=1)
    squared_error = np.full(m, np.nan)
    squared_error[:overlap] = np.sum((x[:overlap, :2] - y[:overlap, :2]) ** 2, axis=1)
    pen_mismatch = np.zeros(m, dtype=bool)
    pen_mismatch[:overlap] = (x[:overlap, 2] >= 0.5) != (y[:overlap, 2] >= 0.5)
    complete = bool(m > 0 and np.all(present) and np.all(valid))
    stop_observed = None if stopped is None else bool(stopped)
    valid_pair = bool(complete and n == m and stop_observed is not False)

    def window_result(start, end):
        count = max(0, end - start)
        available = present[start:end]
        finite = valid[start:end]
        window_complete = bool(count > 0 and np.all(available) and np.all(finite))
        observed = squared_error[start:end][finite]
        return {
            'start_step': int(start), 'end_step_exclusive': int(end),
            'target_step_count': count,
            'valid_window': window_complete,
            'missing_step_count': int(np.sum(~available)),
            'missing_step_fraction': float(np.sum(~available) / count) if count else None,
            'invalid_step_count': int(np.sum(available & ~finite)),
            'raw_timestep_xy_rmse': float(np.sqrt(np.mean(squared_error[start:end]))) if window_complete else None,
            'observed_overlap_xy_rmse': float(np.sqrt(np.mean(observed))) if len(observed) else None,
            'initial_error': float(np.sqrt(squared_error[start])) if count and valid[start] else None,
            'final_error': float(np.sqrt(squared_error[end - 1])) if count and valid[end - 1] else None,
            'pen_mismatch_fraction': float(np.mean(pen_mismatch[start:end])) if window_complete else None,
        }

    def pen_runs(points):
        if not np.all(np.isfinite(points[:, 2])):
            return None
        down = points[:, 2] >= 0.5
        previous = np.concatenate([np.zeros(1, dtype=bool), down[:-1]]) if len(down) else down
        return int(np.sum(down & ~previous))

    generated_runs, target_runs = pen_runs(x), pen_runs(y)
    endpoint_valid = bool(n > 0 and m > 0 and np.all(np.isfinite(x[-1, :2])) and np.all(np.isfinite(y[-1, :2])))
    windows = []
    for step in steps:
        start, end = min(int(step), m), min(int(step) + recovery_window, m)
        window = window_result(start, end)
        window['perturbation_step'] = int(step)
        window['outside_target_schedule'] = bool(step >= m)
        windows.append(window)
    whole = window_result(0, m)
    return {
        'valid_pair': valid_pair,
        'invalid_pair_count': int(not valid_pair),
        'complete_output': complete,
        'generated_count': n, 'target_count': m,
        'length_error': n - m, 'absolute_length_error': abs(n - m),
        'completion_fraction': float(overlap / m) if m else 0.0,
        'valid_step_fraction': float(np.sum(valid) / m) if m else 0.0,
        'missing_step_count': max(m - n, 0),
        'missing_step_fraction': float(max(m - n, 0) / m) if m else None,
        'extra_step_count': max(n - m, 0),
        'invalid_step_count': int(np.sum(present & ~valid)),
        'invalid_generated_step_count': int(np.sum(~np.all(np.isfinite(x), axis=1))),
        'invalid_target_step_count': int(np.sum(~np.all(np.isfinite(y), axis=1))),
        'stop_observed': stop_observed,
        'no_stop_failure': None if stop_observed is None else not stop_observed,
        'raw_timestep_xy_rmse': whole['raw_timestep_xy_rmse'],
        'observed_overlap_xy_rmse': whole['observed_overlap_xy_rmse'],
        'endpoint_error': float(np.linalg.norm(x[-1, :2] - y[-1, :2])) if endpoint_valid else None,
        'requested_final_timestep_error': whole['final_error'],
        'pen_mismatch_fraction': whole['pen_mismatch_fraction'],
        'observed_overlap_pen_mismatch_fraction': float(np.mean(pen_mismatch[valid])) if np.any(valid) else None,
        'generated_pen_down_runs': generated_runs,
        'target_pen_down_runs': target_runs,
        'pen_down_run_count_error': abs(generated_runs - target_runs) if generated_runs is not None and target_runs is not None else None,
        'stroke_count_convention': 'contiguous_pen_down_runs_not_pen_up_travel_points',
        'frame_alignment': 'none', 'time_alignment': 'same_raw_timestep_no_resampling',
        'recovery_window': int(recovery_window), 'recovery_windows': windows,
    }


def nearest_feature_distance(generated: np.ndarray, reservoir: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
    """Evaluation-only near-copy diagnostic in frozen, unnormalized feature space."""
    x, y = _features(generated, minimum=1), _features(reservoir, minimum=1)
    if x.shape[1] != y.shape[1] or batch_size < 1:
        raise ValueError('Invalid feature dimensions or batch size')
    result = []
    for start in range(0, len(x), batch_size):
        block = x[start:start + batch_size]
        minimum = np.full(len(block), np.inf)
        for offset in range(0, len(y), batch_size):
            candidates = y[offset:offset + batch_size]
            distances = np.maximum(np.sum(block**2, axis=1)[:, None] + np.sum(candidates**2, axis=1)[None, :] - 2 * block @ candidates.T, 0.0)
            minimum = np.minimum(minimum, distances.min(axis=1))
        result.extend(np.sqrt(minimum).tolist())
    return np.asarray(result)


def extract_features(artifact: str | Path, output: str | Path, *, python: str, donor_root: str | Path, extractor_checkpoint: str | Path, batch_size: int = 64, label_map: str | Path | None = None, checkpoint_provenance: str | Path | None = None) -> subprocess.CompletedProcess:
    """Run metrics with the selected interpreter and no Torch import in this process."""
    worker = Path(__file__).with_name('metrics_worker.py')
    command = [str(python), str(worker), '--artifact', str(Path(artifact).resolve()), '--output', str(Path(output).resolve()), '--donor-root', str(Path(donor_root).resolve()), '--extractor-checkpoint', str(Path(extractor_checkpoint).resolve()), '--batch-size', str(batch_size)]
    if label_map is not None:
        command += ['--label-map', str(Path(label_map).resolve())]
    if checkpoint_provenance is not None:
        command += ['--checkpoint-provenance', str(Path(checkpoint_provenance).resolve())]
    return subprocess.run(command, check=True, text=True)


def load_features(path: str | Path, *, minimum: int = 2) -> tuple[np.ndarray, dict]:
    path = Path(path)
    metadata = json.loads((path / 'metadata.json').read_text())
    with np.load(path / 'features.npz', allow_pickle=False) as archive:
        features = _features(archive['features'], minimum=minimum)
    if metadata.get('actual_count') != len(features) or len(metadata.get('records', [])) != len(features):
        raise ValueError('Feature artifact metadata count mismatch')
    if metadata.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unknown feature convention')
    if metadata.get('feature_dimension') != 512 or features.shape[1] != 512:
        raise ValueError('Frozen Sketch-FD artifacts require 512 pooled feature channels')
    if metadata.get('feature_normalization') != 'none':
        raise ValueError('Frozen Sketch-FD artifacts require unnormalized features')
    return features, metadata


def _compatible(left: Mapping, right: Mapping) -> None:
    for key in ('feature_version', 'extractor_sha256', 'renderer_config', 'source_hashes'):
        if key not in left or left.get(key) != right.get(key):
            raise ValueError(f'Feature metric provenance differs: {key}')


def _reference_ids(metadata: Mapping) -> set[str]:
    ids = [record.get('drawing_id', record.get('query_id', record.get('sample_id'))) for record in metadata['records']]
    if any(value is None for value in ids):
        raise ValueError('Real references require immutable drawing_id/query_id/sample_id')
    # Overlapping local neighborhoods may share reserved references. Each
    # drawing is unique within a task, and task weights remain explicit.
    keys = [(row.get('intended_neighborhood_id'), str(item))
            for row, item in zip(metadata['records'], ids)]
    if len(set(keys)) != len(keys):
        raise ValueError('Real reference IDs must be unique within each neighborhood')
    return set(map(str, ids))


def _load_metric_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    payload = {key: value for key, value in manifest.items() if key != 'identifier'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if manifest.get('identifier') != digest:
        raise ValueError('QuickDraw metric manifest hash mismatch')
    return manifest


def score_copying(
    generated_features: str | Path,
    support_features: str | Path,
    training_reservoir_features: str | Path,
    *,
    manifest_path: str | Path,
    experiment: str = 'a',
    max_pairs_per_prompt: int = 10000,
    seed: int = 0,
) -> dict:
    """Offline proximity/diversity on fixed features and permitted training IDs.

    Features alone cannot establish an exact trajectory copy. When extraction
    supplied effective-point hashes, exact sequence matches are reported as a
    separate diagnostic. No learned retrieval or feature targets enter policies.
    """
    if experiment not in ('a', 'b1', 'b2') or max_pairs_per_prompt < 1:
        raise ValueError('Invalid copying experiment or pair budget')
    x, mx = load_features(generated_features, minimum=1)
    supports, ms = load_features(support_features, minimum=1)
    reservoir, mt = load_features(training_reservoir_features, minimum=1)
    _compatible(mx, ms)
    _compatible(mx, mt)
    manifest = _load_metric_manifest(manifest_path)
    for metadata in (mx, ms, mt):
        if metadata.get('manifest_id') != manifest['identifier']:
            raise ValueError('Copying artifacts must identify the supplied frozen manifest')
    permitted = manifest['a_ids' if experiment == 'a' else 'b_ids']['train']
    permitted_ids = {str(item): str(category) for category, ids in permitted.items() for item in ids}

    def feature_ids(metadata):
        ids = [record.get('feature_record_id', record.get('drawing_id', record.get('query_id', record.get('sample_id')))) for record in metadata['records']]
        if any(item is None for item in ids) or len(set(map(str, ids))) != len(ids):
            raise ValueError('Support/training feature records need unique immutable IDs')
        return [str(item) for item in ids]

    support_ids, train_ids = feature_ids(ms), feature_ids(mt)
    for item, record in zip(train_ids, mt['records']):
        if item not in permitted_ids or str(record['intended_category']) != permitted_ids[item]:
            raise ValueError('Training feature reservoir contains an ID/category outside permitted training data')
    support_index = {item: index for index, item in enumerate(support_ids)}

    def point_hash(record):
        return record.get('trajectory_sha256') if record.get('trajectory_valid', True) else None

    train_hashes = {point_hash(record) for record in mt['records']} - {None}
    training_hashes_complete = all(point_hash(record) is not None for record in mt['records'])
    per_generated, groups = [], {}
    for index, (feature, record) in enumerate(zip(x, mx['records'])):
        if record.get('condition') in ('action_corruption', 'stroke_reversal') and 'shown_support_feature_ids' not in record:
            raise ValueError('Transformed support copying requires shown_support_feature_ids for actual supplied trajectories')
        shown_ids = list(map(str, record.get('shown_support_feature_ids', record.get('support_ids', []))))
        missing = set(shown_ids) - set(support_index)
        if missing:
            raise ValueError(f'Shown support feature IDs are missing: {sorted(missing)}')
        shown_indices = [support_index[item] for item in shown_ids]
        if shown_indices:
            distances = np.linalg.norm(supports[shown_indices] - feature, axis=1)
            closest = int(np.argmin(distances))
            nearest_support_id, nearest_support_distance = shown_ids[closest], float(distances[closest])
        else:
            nearest_support_id, nearest_support_distance = None, None
        best_distance, best_index = float('inf'), -1
        for start in range(0, len(reservoir), 512):
            distances = np.linalg.norm(reservoir[start:start + 512] - feature, axis=1)
            closest = int(np.argmin(distances))
            if float(distances[closest]) < best_distance:
                best_distance, best_index = float(distances[closest]), start + closest
        sequence_hash = point_hash(record)
        shown_hashes = {point_hash(ms['records'][item]) for item in shown_indices} - {None}
        per_generated.append({
            'sample_index': index, 'intended_category': record['intended_category'],
            'condition': record.get('condition'), 'shown_support_ids': shown_ids,
            'shown_support_count': len(shown_indices),
            'nearest_support_id': nearest_support_id,
            'nearest_support_feature_distance': nearest_support_distance,
            'nearest_training_id': train_ids[best_index],
            'nearest_training_feature_distance': best_distance,
            'zero_support_feature_distance': nearest_support_distance == 0.0 if shown_indices else None,
            'zero_training_feature_distance': best_distance == 0.0,
            'exact_support_point_sequence': sequence_hash in shown_hashes if sequence_hash is not None and shown_indices and all(point_hash(ms['records'][item]) is not None for item in shown_indices) else None,
            'exact_training_point_sequence': sequence_hash in train_hashes if sequence_hash is not None and training_hashes_complete else None,
        })
        group_identity = {
            'task_index': record.get('task_index', record.get('task_id')),
            'query_index': record.get('query_index'), 'intended_category': record['intended_category'],
            'shown_support_ids': shown_ids, 'condition': record.get('condition'), 'frame': record.get('frame'),
        }
        group_key = json.dumps(group_identity, sort_keys=True, separators=(',', ':'))
        groups.setdefault(group_key, []).append(index)
    diversity = []
    rng = np.random.default_rng(seed)
    for group_key, indices in sorted(groups.items()):
        total = len(indices) * (len(indices) - 1) // 2
        budget = min(total, max_pairs_per_prompt)
        if total <= max_pairs_per_prompt:
            pairs = [(indices[left], indices[right]) for left in range(len(indices)) for right in range(left + 1, len(indices))]
        else:
            pairs = set()
            while len(pairs) < budget:
                left, right = sorted(rng.choice(indices, 2, replace=False).tolist())
                pairs.add((left, right))
            pairs = sorted(pairs)
        distances = [float(np.linalg.norm(x[left] - x[right])) for left, right in pairs]
        known_hash_pairs = [(point_hash(mx['records'][left]), point_hash(mx['records'][right])) for left, right in pairs if point_hash(mx['records'][left]) is not None and point_hash(mx['records'][right]) is not None]
        diversity.append({
            'prompt': json.loads(group_key), 'sample_indices': indices,
            'available_pair_count': total, 'evaluated_pair_count': len(pairs),
            'pair_selection': 'all_pairs' if total <= max_pairs_per_prompt else 'seeded_uniform_without_replacement',
            'mean_pair_feature_distance': float(np.mean(distances)) if distances else None,
            'zero_feature_distance_pair_count': sum(value == 0.0 for value in distances),
            'point_hash_evaluable_pair_count': len(known_hash_pairs),
            'exact_point_sequence_pair_count': sum(left == right for left, right in known_hash_pairs),
        })
    support_distances = [row['nearest_support_feature_distance'] for row in per_generated if row['nearest_support_feature_distance'] is not None]
    return {
        'schema_version': SCHEMA_VERSION, 'metric': 'frozen_sketch_copying_and_diversity_v1',
        'manifest_id': manifest['identifier'], 'experiment': experiment,
        'feature_version': FEATURE_VERSION, 'feature_normalization': 'none',
        'extractor_sha256': mx['extractor_sha256'], 'source_hashes': mx['source_hashes'],
        'renderer_config': mx['renderer_config'], 'generated_count': len(x),
        'available_support_feature_count': len(supports), 'training_reservoir_feature_count': len(reservoir),
        'permitted_training_reservoir_count': len(permitted_ids), 'training_reservoir_ids': train_ids,
        'mean_nearest_support_feature_distance': float(np.mean(support_distances)) if support_distances else None,
        'support_distance_evaluable_count': len(support_distances),
        'mean_nearest_training_feature_distance': float(np.mean([row['nearest_training_feature_distance'] for row in per_generated])),
        'zero_support_feature_distance_count': sum(row['zero_support_feature_distance'] is True for row in per_generated),
        'zero_training_feature_distance_count': sum(row['zero_training_feature_distance'] for row in per_generated),
        'exact_support_point_sequence_count': sum(row['exact_support_point_sequence'] is True for row in per_generated),
        'exact_support_point_sequence_evaluable_count': sum(row['exact_support_point_sequence'] is not None for row in per_generated),
        'exact_training_point_sequence_count': sum(row['exact_training_point_sequence'] is True for row in per_generated),
        'exact_training_point_sequence_evaluable_count': sum(row['exact_training_point_sequence'] is not None for row in per_generated),
        'feature_coincidence_interpretation': 'zero feature distance is not proof of an exact trajectory copy',
        'near_copy_threshold': None, 'near_copy_threshold_status': 'uncalibrated; distances only',
        'per_generated': per_generated, 'within_prompt_diversity': diversity,
        'pair_selection_seed': int(seed), 'max_pairs_per_prompt': max_pairs_per_prompt,
        'statistics': mx.get('statistics', {}),
    }


def score_feature_artifacts(generated: str | Path, reference: str | Path, *, reference_repeats: Sequence[str | Path] = ()) -> dict:
    x, mx = load_features(generated)
    y, my = load_features(reference)
    _compatible(mx, my)
    if mx.get('manifest_id') != my.get('manifest_id'):
        raise ValueError('Generated and real feature artifacts refer to different manifests')
    reference_ids = _reference_ids(my)
    support_ids = {str(value) for record in mx['records'] for value in record.get('support_ids', [])}
    if reference_ids & support_ids:
        raise ValueError('Primary real references must be independent of support IDs')
    labels_x = [str(record['intended_category']) for record in mx['records']]
    labels_y = [str(record['intended_category']) for record in my['records']]
    conditional = category_metrics(x, labels_x, y, labels_y)
    local = any(row.get('intended_neighborhood_id') for row in mx['records'])
    neighborhood = neighborhood_metrics(x, mx['records'], y, my['records']) if local else None
    weights_x = {label: labels_x.count(label) / len(labels_x) for label in set(labels_x)}
    weights_y = {label: labels_y.count(label) / len(labels_y) for label in set(labels_y)}
    if weights_x != weights_y:
        raise ValueError('Pooled metric requires matched intended-category weights')
    repeats = []
    seen_reference_sets = {frozenset(reference_ids)}
    for repeat_path in reference_repeats:
        z, mz = load_features(repeat_path)
        _compatible(my, mz)
        if mz.get('manifest_id') != my.get('manifest_id'):
            raise ValueError('Real-real feature artifacts refer to different manifests')
        ids = _reference_ids(mz)
        if ids & reference_ids or ids & support_ids:
            raise ValueError('Real-real references must use disjoint drawing IDs and exclude supports')
        if frozenset(ids) in seen_reference_sets:
            raise ValueError('Repeated calibration cannot reuse the identical reference ID set')
        seen_reference_sets.add(frozenset(ids))
        labels_z = [str(record['intended_category']) for record in mz['records']]
        weights_z = {label: labels_z.count(label) / len(labels_z) for label in set(labels_z)}
        if weights_z != weights_y or len(z) != len(y):
            raise ValueError('Real-real sample counts and category weights must match')
        calibration = {'reference_ids': sorted(ids), 'actual_count': len(z),
                       'sketch_fd': sketch_fd(z, y), 'conditional': category_metrics(z, labels_z, y, labels_y)}
        if local:
            calibration['neighborhood_conditional'] = neighborhood_metrics(z, mz['records'], y, my['records'])
            for task, row in calibration['neighborhood_conditional']['per_neighborhood'].items():
                if row['generated_count'] != row['reference_count']:
                    raise ValueError('Real-real local references need matched per-neighborhood sample counts.')
        repeats.append(calibration)
    scores = np.asarray([value['sketch_fd'] for value in repeats])
    return {
        'schema_version': SCHEMA_VERSION, 'metric': 'Sketch-FD', 'fd_version': FD_VERSION,
        'protocol': ('reserved_neighborhood_references' if local else 'independent_balanced_references'),
        'sketch_fd': sketch_fd(x, y), 'conditional': conditional,
        'neighborhood_conditional': neighborhood,
        'local_reference_interpretation': (
            'A-NN target-centered local-distribution proxy; top-K supports are not independent posterior samples'
            if any(row.get('a_protocol') == 'a_nn' for row in mx['records']) or
               any(row.get('local_reference_role') == 'offline_target_local_distribution_proxy' for row in my['records'])
            else None),
        'generated_count': len(x), 'reference_count': len(y),
        'reference_ids': sorted(reference_ids), 'category_weights': weights_y,
        'manifest_id': mx.get('manifest_id'),
        'extractor_sha256': mx['extractor_sha256'], 'feature_version': FEATURE_VERSION,
        'renderer_config': mx['renderer_config'], 'source_hashes': mx['source_hashes'],
        'extractor_training_provenance': mx.get('extractor_training_provenance', 'unknown'),
        'statistics': mx.get('statistics', {}),
        'classification': mx.get('classification'),
        'real_real': repeats,
        'real_real_fd_mean': float(scores.mean()) if len(scores) else None,
        'real_real_fd_std': float(scores.std(ddof=1)) if len(scores) > 1 else None,
        'uncertainty_method': 'sample SD across fixed independent reference sets; no bias correction',
        'covariance_rank_limited': min(len(x), len(y)) <= x.shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    extract = commands.add_parser('extract')
    extract.add_argument('artifact')
    extract.add_argument('--output', required=True)
    extract.add_argument('--python', required=True)
    extract.add_argument('--donor-root', required=True)
    extract.add_argument('--extractor-checkpoint', required=True)
    extract.add_argument('--batch-size', type=int, default=64)
    extract.add_argument('--label-map')
    extract.add_argument('--checkpoint-provenance')
    score = commands.add_parser('score')
    score.add_argument('generated')
    score.add_argument('reference')
    score.add_argument('--reference-repeat', action='append', default=[])
    score.add_argument('--output', required=True)
    copying = commands.add_parser('copying')
    copying.add_argument('generated_features')
    copying.add_argument('support_features')
    copying.add_argument('training_reservoir_features')
    copying.add_argument('--manifest-path', required=True)
    copying.add_argument('--experiment', choices=('a', 'b1', 'b2'), default='a')
    copying.add_argument('--max-pairs-per-prompt', type=int, default=10000)
    copying.add_argument('--seed', type=int, default=0)
    copying.add_argument('--output', required=True)
    swap = commands.add_parser('swap-check')
    swap.add_argument('real_features')
    swap.add_argument('reference_features')
    swap.add_argument('--left-task', required=True)
    swap.add_argument('--right-task', required=True)
    swap.add_argument('--output', required=True)
    geometry = commands.add_parser('geometry-copying')
    geometry.add_argument('generated')
    geometry.add_argument('supports')
    geometry.add_argument('training')
    geometry.add_argument('--manifest-path', required=True)
    geometry.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'extract':
        extract_features(args.artifact, args.output, python=args.python, donor_root=args.donor_root, extractor_checkpoint=args.extractor_checkpoint, batch_size=args.batch_size, label_map=args.label_map, checkpoint_provenance=args.checkpoint_provenance)
    elif args.command == 'score':
        result = score_feature_artifacts(args.generated, args.reference, reference_repeats=args.reference_repeat)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    elif args.command == 'swap-check':
        result = same_category_swap_check(args.real_features, args.reference_features,
                                         left_task=args.left_task, right_task=args.right_task)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    elif args.command == 'geometry-copying':
        result = score_geometry_copying(args.generated, args.supports, args.training, manifest_path=args.manifest_path)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    else:
        result = score_copying(args.generated_features, args.support_features, args.training_reservoir_features, manifest_path=args.manifest_path, experiment=args.experiment, max_pairs_per_prompt=args.max_pairs_per_prompt, seed=args.seed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
