from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from ml_collections import ConfigDict
from phi_mujoco.integrations import family_composition, family_contract

from icil_jax_rlbench.data.metaworld_hidden_goal import (
    MetaWorldTaskDataset,
    MetaWorldTaskSampler,
    benchmark_from_config,
)
from icil_jax_rlbench.eval.support_controls import condition_support
from icil_jax_rlbench.models.fast_weight_ttt import (
    adapt_fast_state,
    fast_state_effective_rank,
    initial_fast_state,
    predict_action,
    query_imitation_loss,
    segment_support,
    write_loss,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.metaworld_query_runner import metaworld_model_config_from
from icil_jax_rlbench.train.metaworld_ttt_runner import validate_metaworld_ttt_config
from icil_jax_rlbench.train.provenance import config_to_dict
from icil_jax_rlbench.train.ttt_config import adaptation_config_from

_LOGGER = logging.getLogger(__name__)
_SUPPORT_FIELDS = ('observation', 'action', 'next_observation', 'write_mask')
_QUERY_FIELDS = ('observation', 'action', 'outer_loss_mask')
_INFORMATION_REPRESENTATIONS = (
    'raw_support_statistics',
    'first_write_gradient',
    'final_fast_delta',
    'read_action_delta',
)
_SUPPORTED_CONDITIONS = (
    'correct_support',
    'same_family_wrong_instance',
    'different_family_support',
    'shuffled_actions',
    'shuffled_time',
    'observations_only',
    'actions_only',
)


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _remove_task_axis(
    section: Mapping[str, np.ndarray], fields: Sequence[str]
) -> dict[str, np.ndarray]:
    return {name: np.asarray(section[name][0]) for name in fields}


def raw_support_statistics(
    support: Mapping[str, np.ndarray], *, phase_bins: int = 4
) -> np.ndarray:
    """Summarize raw support without using privileged task metadata.

    The fixed-size result contains global mean/std statistics, phase-binned
    means, the valid fraction, and log transition count. Phase is measured
    independently inside each demonstration, preserving the demo/time axes.
    """

    observation = np.asarray(support['observation'], dtype=np.float64)
    action = np.asarray(support['action'], dtype=np.float64)
    next_observation = np.asarray(support['next_observation'], dtype=np.float64)
    valid = np.asarray(support['write_mask'], dtype=np.bool_)
    if observation.ndim != 3 or action.ndim != 3 or valid.shape != observation.shape[:2]:
        raise ValueError('Support must have explicit [demo, time, feature] axes.')
    if next_observation.shape != observation.shape:
        raise ValueError('Support observations and next observations must align.')
    if not np.any(valid):
        raise ValueError('Support must contain at least one valid transition.')
    transition = next_observation - observation
    evidence = np.concatenate((observation, action, transition), axis=-1)
    valid_evidence = evidence[valid]
    values = [np.mean(valid_evidence, axis=0), np.std(valid_evidence, axis=0)]

    phase_values: list[list[np.ndarray]] = [[] for _ in range(int(phase_bins))]
    for demo_index in range(valid.shape[0]):
        demo_evidence = evidence[demo_index, valid[demo_index]]
        count = demo_evidence.shape[0]
        if not count:
            continue
        phase = np.minimum(
            (np.arange(count) * int(phase_bins)) // count,
            int(phase_bins) - 1,
        )
        for phase_index in range(int(phase_bins)):
            selected = demo_evidence[phase == phase_index]
            if selected.size:
                phase_values[phase_index].append(selected)
    width = evidence.shape[-1]
    for selected in phase_values:
        values.append(
            np.mean(np.concatenate(selected, axis=0), axis=0)
            if selected
            else np.zeros((width,), dtype=np.float64)
        )
    values.append(
        np.asarray(
            [np.mean(valid), np.log1p(np.count_nonzero(valid))], dtype=np.float64
        )
    )
    result = np.concatenate(values).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError('Raw support statistics contain NaN or infinity.')
    return result


def _standardize_features(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, dtype=np.float64)
    scale = np.std(train, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def ridge_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """Fit a frozen linear ridge probe, using the smaller primal/dual system."""

    train = np.asarray(train_features, dtype=np.float64)
    test = np.asarray(test_features, dtype=np.float64)
    target = np.asarray(train_targets, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError('Probe features must be aligned rank-two arrays.')
    if target.shape[0] != train.shape[0] or train.shape[0] < 2:
        raise ValueError('Probe targets must align with at least two training rows.')
    if not np.isfinite(ridge) or float(ridge) <= 0.0:
        raise ValueError('ridge must be finite and positive.')
    train, test = _standardize_features(train, test)
    target_mean = np.mean(target, axis=0, keepdims=True)
    centered_target = target - target_mean
    if train.shape[1] > train.shape[0]:
        gram = train @ train.T
        dual = np.linalg.solve(
            gram + float(ridge) * np.eye(train.shape[0]), centered_target
        )
        prediction = test @ train.T @ dual
    else:
        covariance = train.T @ train
        weights = np.linalg.solve(
            covariance + float(ridge) * np.eye(train.shape[1]),
            train.T @ centered_target,
        )
        prediction = test @ weights
    return prediction + target_mean


def _classification_probe(
    train_features: np.ndarray,
    train_labels: Sequence[str],
    test_features: np.ndarray,
    test_labels: Sequence[str],
    *,
    ridge: float,
) -> dict[str, Any]:
    classes = tuple(sorted(set(train_labels)))
    if not classes or set(test_labels) - set(classes):
        raise ValueError('Classification test labels must occur in probe training.')
    class_index = {name: index for index, name in enumerate(classes)}
    targets = np.zeros((len(train_labels), len(classes)), dtype=np.float64)
    targets[np.arange(len(train_labels)), [class_index[name] for name in train_labels]] = 1.0
    scores = ridge_predict(train_features, targets, test_features, ridge=ridge)
    predictions = np.argmax(scores, axis=1)
    expected = np.asarray([class_index[name] for name in test_labels])
    per_class = {
        name: float(np.mean(predictions[expected == index] == index))
        for name, index in class_index.items()
        if np.any(expected == index)
    }
    return {
        'accuracy': float(np.mean(predictions == expected)),
        'balanced_accuracy': float(np.mean(tuple(per_class.values()))),
        'chance_accuracy': 1.0 / len(classes),
        'classes': list(classes),
        'per_class_accuracy': per_class,
        'train_rows': len(train_labels),
        'test_rows': len(test_labels),
    }


def _regression_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    *,
    ridge: float,
) -> dict[str, float | int]:
    prediction = ridge_predict(
        train_features, train_targets, test_features, ridge=ridge
    )
    target = np.asarray(test_targets, dtype=np.float64)
    error = prediction - target
    rmse = float(np.sqrt(np.mean(np.square(error))))
    baseline = np.mean(np.asarray(train_targets, dtype=np.float64), axis=0)
    baseline_error = target - baseline
    baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_error))))
    # Out-of-sample skill relative to the train-mean predictor. Unlike the
    # conventional test-mean R2, this remains defined for one test instance.
    denominator = float(np.sum(np.square(baseline_error)))
    r2 = 1.0 - float(np.sum(np.square(error))) / max(denominator, 1e-12)
    return {
        'rmse': rmse,
        'mean_predictor_rmse': baseline_rmse,
        'normalized_rmse': rmse / max(baseline_rmse, 1e-12),
        'r2_vs_train_mean': r2,
        'train_rows': int(train_features.shape[0]),
        'test_rows': int(test_features.shape[0]),
        'target_dimension': int(target.shape[1]),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def _geometry_summary(
    features: np.ndarray, task_ids: Sequence[str], families: Sequence[str]
) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    similarities = normalized @ normalized.T
    task_ids = np.asarray(task_ids)
    families = np.asarray(families)
    upper = np.triu(np.ones(similarities.shape, dtype=np.bool_), k=1)
    categories = {
        'same_task': upper & (task_ids[:, None] == task_ids[None, :]),
        'same_family_different_task': (
            upper
            & (task_ids[:, None] != task_ids[None, :])
            & (families[:, None] == families[None, :])
        ),
        'different_family': upper & (families[:, None] != families[None, :]),
    }
    result: dict[str, Any] = {}
    for name, mask in categories.items():
        selected = similarities[mask]
        result[name] = {
            'pairs': int(selected.size),
            'mean_cosine': float(np.mean(selected)) if selected.size else None,
            'median_cosine': float(np.median(selected)) if selected.size else None,
            'std_cosine': float(np.std(selected)) if selected.size else None,
        }
    return result


def _task_latent(dataset: MetaWorldTaskDataset, task_id: str) -> tuple[np.ndarray, np.ndarray]:
    descriptor = dataset.task_descriptor(task_id)
    native = np.asarray(descriptor.get('native_rand_vec', ()), dtype=np.float32)
    contract = family_contract(dataset.task_family(task_id))
    selected = []
    for field in contract.reset_fields:
        if field.role == 'task_latent':
            selected.extend(range(field.start, field.stop))
    latent = native[selected] if selected else np.empty((0,), dtype=np.float32)
    padded = np.zeros((6,), dtype=np.float32)
    mask = np.zeros((6,), dtype=np.bool_)
    padded[: latent.size] = latent
    mask[: latent.size] = True
    return padded, mask


def _fixed_probe_observations(
    dataset: MetaWorldTaskDataset, count: int
) -> np.ndarray:
    if int(count) < 1:
        raise ValueError('probe_observations must be positive.')
    task_ids = dataset.task_ids('train')
    collected = []
    task_position = 0
    while len(collected) < int(count):
        task_id = task_ids[task_position % len(task_ids)]
        episodes = dataset.task_index.episode_indices(task_id)
        episode_index = episodes[(task_position // len(task_ids)) % len(episodes)]
        section = dataset.section((episode_index,), support=False)
        valid = np.asarray(section['outer_loss_mask'][0], dtype=np.bool_)
        observations = np.asarray(section['observation'][0][valid])
        source_index = (task_position // max(1, len(task_ids))) % len(observations)
        collected.append(observations[source_index])
        task_position += 1
    return np.asarray(collected, dtype=np.float32)


def _make_extractor(params, model_cfg, adapt_cfg):
    def extract(support, query, probe_observations):
        initial = initial_fast_state(params)
        segments = segment_support(support, int(adapt_cfg.write_segment_size))
        first_segment = jax.tree_util.tree_map(lambda value: value[0], segments)
        first_gradient = jax.grad(write_loss, argnums=1)(
            params,
            initial,
            first_segment,
            model_cfg,
            adapt_cfg,
            initial,
        )

        def query_for_fast(fast_state):
            value, _ = query_imitation_loss(
                params, fast_state, query, model_cfg, adapt_cfg
            )
            return value

        oracle_gradient = jax.grad(query_for_fast)(initial)
        adapted, trace = adapt_fast_state(
            params, support, model_cfg, adapt_cfg
        )
        before, _ = query_imitation_loss(
            params, initial, query, model_cfg, adapt_cfg
        )
        after, _ = query_imitation_loss(
            params, adapted, query, model_cfg, adapt_cfg
        )
        first_vector, _ = ravel_pytree(first_gradient)
        oracle_vector, _ = ravel_pytree(oracle_gradient)
        initial_vector, _ = ravel_pytree(initial)
        adapted_vector, _ = ravel_pytree(adapted)
        final_delta = adapted_vector - initial_vector
        before_action = predict_action(
            params,
            initial,
            probe_observations,
            model_cfg,
            read_enabled=bool(adapt_cfg.read_enabled),
            read_mode=str(adapt_cfg.read_mode),
            read_scale=float(adapt_cfg.read_scale),
        )
        after_action = predict_action(
            params,
            adapted,
            probe_observations,
            model_cfg,
            read_enabled=bool(adapt_cfg.read_enabled),
            read_mode=str(adapt_cfg.read_mode),
            read_scale=float(adapt_cfg.read_scale),
        )
        gradient_cosine = jnp.vdot(first_vector, oracle_vector) / (
            jnp.linalg.norm(first_vector) * jnp.linalg.norm(oracle_vector) + 1e-8
        )
        return {
            'first_write_gradient': first_vector,
            'oracle_query_gradient': oracle_vector,
            'final_fast_delta': final_delta,
            'read_action_delta': (after_action - before_action).reshape(-1),
            'query_loss_before': before,
            'query_loss_after': after,
            'query_gain': before - after,
            'write_query_gradient_cosine': gradient_cosine,
            'update_query_gradient_cosine': -gradient_cosine,
            'first_write_loss': trace['write_loss'][0],
            'mean_write_loss': jnp.mean(trace['write_loss']),
            'first_write_gradient_norm': jnp.linalg.norm(first_vector),
            'oracle_query_gradient_norm': jnp.linalg.norm(oracle_vector),
            'final_fast_delta_norm': jnp.linalg.norm(final_delta),
            'adapted_fast_effective_rank': fast_state_effective_rank(adapted),
            'delta_fast_effective_rank': fast_state_effective_rank(
                jax.tree_util.tree_map(lambda end, start: end - start, adapted, initial)
            ),
        }

    return jax.jit(extract)


def _host_result(value: Mapping[str, Any]) -> dict[str, np.ndarray | float]:
    result: dict[str, np.ndarray | float] = {}
    for name, item in jax.device_get(value).items():
        array = np.asarray(item)
        result[name] = float(array) if array.shape == () else array.astype(np.float32)
    return result


def _selected_tasks(
    dataset: MetaWorldTaskDataset, split: str, limit: int
) -> tuple[str, ...]:
    return dataset.balanced_task_ids(split, int(limit))


def _probe_summary(
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, np.ndarray],
    *,
    support_counts: Sequence[int],
    conditions: Sequence[str],
    ridge: float,
) -> dict[str, Any]:
    row_split = np.asarray([row['target_split'] for row in rows])
    row_condition = np.asarray([row['condition'] for row in rows])
    row_count = np.asarray([row['support_count'] for row in rows], dtype=np.int32)
    row_family = np.asarray([row['support_family'] for row in rows])
    row_task = np.asarray([row['support_task_id'] for row in rows])
    latent = np.asarray([row['support_task_latent'] for row in rows], dtype=np.float32)
    latent_mask = np.asarray(
        [row['support_task_latent_mask'] for row in rows], dtype=np.bool_
    )
    summary: dict[str, Any] = {}
    for support_count in support_counts:
        count_result: dict[str, Any] = {}
        for condition in conditions:
            selected = (row_count == int(support_count)) & (row_condition == condition)
            train = selected & (row_split == 'train')
            test = selected & (row_split == 'latent_validation')
            condition_result: dict[str, Any] = {}
            if np.count_nonzero(train) < 2 or np.count_nonzero(test) < 1:
                continue
            for representation in _INFORMATION_REPRESENTATIONS:
                representation_result: dict[str, Any] = {}
                representation_result['family_classification'] = _classification_probe(
                    features[representation][train],
                    row_family[train].tolist(),
                    features[representation][test],
                    row_family[test].tolist(),
                    ridge=float(ridge),
                )
                family_regression = {}
                for family in sorted(set(row_family[train]) & set(row_family[test])):
                    family_train = train & (row_family == family)
                    family_test = test & (row_family == family)
                    active = np.flatnonzero(np.any(latent_mask[family_train], axis=0))
                    if not active.size or np.count_nonzero(family_train) < 2:
                        continue
                    family_regression[family] = _regression_probe(
                        features[representation][family_train],
                        latent[family_train][:, active],
                        features[representation][family_test],
                        latent[family_test][:, active],
                        ridge=float(ridge),
                    )
                representation_result['latent_regression_by_family'] = family_regression
                if family_regression:
                    representation_result['mean_latent_regression'] = {
                        name: float(
                            np.mean([value[name] for value in family_regression.values()])
                        )
                        for name in ('normalized_rmse', 'r2_vs_train_mean')
                    }
                geometry = selected & np.isin(
                    row_split, ('train', 'latent_validation', 'family_validation')
                )
                representation_result['cosine_geometry'] = _geometry_summary(
                    features[representation][geometry],
                    row_task[geometry].tolist(),
                    row_family[geometry].tolist(),
                )
                condition_result[representation] = representation_result
            count_result[condition] = condition_result
        summary[str(support_count)] = count_result
    return summary


def _condition_summary(
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, np.ndarray],
    *,
    target_split: str | None = None,
) -> dict[str, Any]:
    index = {
        (
            row['target_task_id'],
            row['target_split'],
            row['support_count'],
            row['sample_index'],
            row['condition'],
        ): position
        for position, row in enumerate(rows)
        if target_split is None or row['target_split'] == target_split
    }
    conditions = sorted({str(row['condition']) for row in rows} - {'correct_support'})
    result: dict[str, Any] = {}
    for condition in conditions:
        correct_indices = []
        condition_indices = []
        for key, correct_index in index.items():
            if key[-1] != 'correct_support':
                continue
            other = (*key[:-1], condition)
            if other in index:
                correct_indices.append(correct_index)
                condition_indices.append(index[other])
        if not correct_indices:
            continue
        left = np.asarray(correct_indices)
        right = np.asarray(condition_indices)
        result[condition] = {
            'paired_rows': int(left.size),
            'first_gradient_cosine_to_correct': float(
                np.mean(
                    _cosine(
                        features['first_write_gradient'][left],
                        features['first_write_gradient'][right],
                    )
                )
            ),
            'final_delta_cosine_to_correct': float(
                np.mean(
                    _cosine(
                        features['final_fast_delta'][left],
                        features['final_fast_delta'][right],
                    )
                )
            ),
            'mean_query_gain': float(
                np.mean([float(rows[position]['query_gain']) for position in right])
            ),
            'mean_query_gain_difference_from_correct': float(
                np.mean(
                    [
                        float(rows[other]['query_gain']) - float(rows[correct]['query_gain'])
                        for correct, other in zip(left, right, strict=True)
                    ]
                )
            ),
        }
    return result


def _scalar_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    names = (
        'query_loss_before',
        'query_loss_after',
        'query_gain',
        'write_query_gradient_cosine',
        'update_query_gradient_cosine',
        'first_write_loss',
        'mean_write_loss',
        'first_write_gradient_norm',
        'oracle_query_gradient_norm',
        'final_fast_delta_norm',
        'adapted_fast_effective_rank',
        'delta_fast_effective_rank',
    )
    return {
        name: float(np.mean([float(row[name]) for row in rows])) for name in names
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    benchmark_label = str(summary['benchmark_label'])
    lines = [
        f'# MetaWorld {benchmark_label} Update-Information Diagnostic',
        '',
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        f"- WRITE objective: `{summary['write_objective']}`",
        f"- READ mode: `{summary['read_mode']}`",
        f"- Feature rows: {summary['feature_rows']}",
        '',
        '## Interpretation',
        '',
        (
            'Family accuracy measures whether a frozen linear probe can recover the '
            f'familiar {benchmark_label} family on disjoint latent-validation instances. '
            'Latent regression measures '
            'within-family native task-vector recovery; normalized RMSE below 1 beats a '
            'train-mean predictor. Raw support statistics are a non-privileged information '
            'upper bound. Oracle query gradients are saved for diagnosis only and are never '
            'used by adaptation.'
        ),
        '',
        '## Frozen Probes',
        '',
        '| supports | condition | representation | family acc. | latent nRMSE | latent skill |',
        '|---:|---|---|---:|---:|---:|',
    ]
    for support_count, count_values in summary['probes'].items():
        for condition, condition_values in count_values.items():
            for representation, values in condition_values.items():
                family = values['family_classification']['accuracy']
                latent = values.get('mean_latent_regression', {})
                lines.append(
                    f'| {support_count} | `{condition}` | `{representation}` | '
                    f'{family:.3f} | '
                    f"{latent.get('normalized_rmse', float('nan')):.3f} | "
                    f"{latent.get('r2_vs_train_mean', float('nan')):.3f} |"
                )
    lines.extend(
        [
            '',
            '## Matched Support Controls',
            '',
            '| condition | first-gradient cosine | final-delta cosine | query gain | gain vs correct |',
            '|---|---:|---:|---:|---:|',
        ]
    )
    for condition, values in summary['condition_comparisons'].items():
        lines.append(
            f"| `{condition}` | {values['first_gradient_cosine_to_correct']:.3f} | "
            f"{values['final_delta_cosine_to_correct']:.3f} | "
            f"{values['mean_query_gain']:.5f} | "
            f"{values['mean_query_gain_difference_from_correct']:.5f} |"
        )
    visualization = summary.get('visualization')
    if isinstance(visualization, Mapping):
        lines.extend(
            [
                '',
                '## Update Embeddings',
                '',
                (
                    'Interactive PCA+t-SNE views are in `tsne/`. The family view uses '
                    'marker shape for split; the held-out view colors held-out families, '
                    'shows familiar-family updates as gray context, and connects repeated '
                    'support samples from the same task instance. Treat t-SNE as a local '
                    'visualization, not evidence by itself; original-space nearest-neighbor '
                    'and cosine measurements are in `tsne/manifest.json`.'
                ),
            ]
        )
    lines.extend(
        [
            '',
            (
                'The complete per-family regressions, cosine geometry, scalar metrics, and '
                'provenance are in `summary.json`. Full, unprojected vectors are in '
                '`features.npz`.'
            ),
            '',
        ]
    )
    return '\n'.join(lines)


def _plot_probe_scores(summary: Mapping[str, Any], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _LOGGER.warning('matplotlib is unavailable; skipped probe score plot.')
        return
    support_count = next(iter(summary['probes']))
    correct = summary['probes'][support_count].get('correct_support', {})
    if not correct:
        return
    names = list(correct)
    family = [correct[name]['family_classification']['accuracy'] for name in names]
    latent = [
        correct[name]
        .get('mean_latent_regression', {})
        .get('r2_vs_train_mean', np.nan)
        for name in names
    ]
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(x, family, color='#176B87')
    axes[0].set_title(f'Family probe ({support_count} support)')
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel('accuracy')
    axes[1].bar(x, latent, color='#B24C3D')
    axes[1].axhline(0.0, color='black', linewidth=0.8)
    axes[1].set_title('Mean within-family latent probe')
    axes[1].set_ylabel('skill vs train mean')
    for axis in axes:
        axis.set_xticks(x, names, rotation=25, ha='right')
        axis.grid(axis='y', alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyze_metaworld_update_information(cfg: ConfigDict) -> Path:
    """Extract and probe information represented by family-aware WRITE updates."""

    checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError('checkpoint_path must point at a TTT checkpoint.')
    payload = load_checkpoint(checkpoint_path)
    train_cfg = ConfigDict(payload['config'])
    benchmark = benchmark_from_config(train_cfg)
    requested_integration = str(cfg.get('integration', ''))
    if requested_integration and requested_integration != benchmark.integration_name:
        raise ValueError(
            f'Diagnostic config requests {requested_integration!r}, but checkpoint '
            f'belongs to {benchmark.integration_name!r}.'
        )
    if not benchmark.family_aware:
        raise ValueError(
            'The update-information diagnostic requires a family-aware benchmark.'
        )
    validate_metaworld_ttt_config(train_cfg)
    extra = payload.get('extra', {})
    if extra.get('checkpoint_type') != benchmark.ttt_mode:
        raise ValueError(
            f'checkpoint_path is not a {benchmark.label} TTT checkpoint.'
        )

    conditions = tuple(str(value) for value in cfg.conditions)
    unknown = set(conditions) - set(_SUPPORTED_CONDITIONS)
    if unknown or 'correct_support' not in conditions:
        raise ValueError(
            'conditions must include correct_support and use only '
            f'{_SUPPORTED_CONDITIONS}; unknown={sorted(unknown)}.'
        )
    support_counts = tuple(int(value) for value in cfg.support_counts)
    if not support_counts or any(value < 1 for value in support_counts):
        raise ValueError('support_counts must contain positive integers.')
    if int(cfg.query_episodes) < 1 or int(cfg.samples_per_task) < 1:
        raise ValueError('query_episodes and samples_per_task must be positive.')

    cache_root = str(cfg.cache_root) or str(train_cfg.dataset.cache_root)
    normalization = extra.get('normalization')
    if not isinstance(normalization, Mapping):
        raise TypeError('Checkpoint lacks normalization statistics.')
    dataset = MetaWorldTaskDataset(
        cache_root,
        integration_name=benchmark.integration_name,
        protocol=str(train_cfg.dataset.get('protocol', 'default')),
        horizon_buckets=tuple(train_cfg.dataset.get('horizon_buckets', ())),
        normalization=normalization,
        cache_prepared_episodes=bool(cfg.cache_prepared_episodes),
    )
    if dataset.bundle.data_sha256 != extra.get('cache_data_sha256'):
        raise ValueError('Diagnostic cache differs from the checkpoint cache.')
    if dataset.normalization_id != extra.get('normalizer_id'):
        raise ValueError('Diagnostic normalization differs from the checkpoint.')
    required_episodes = max(support_counts) + int(cfg.query_episodes)
    if required_episodes > dataset.task_index.episodes_per_task:
        raise ValueError(
            f'Diagnostic needs {required_episodes} episodes per task, but cache has '
            f'{dataset.task_index.episodes_per_task}.'
        )

    model_cfg = metaworld_model_config_from(train_cfg, dataset)
    if asdict(model_cfg) != extra.get('model_config'):
        raise ValueError('Checkpoint model metadata differs from its config.')
    adapt_cfg = adaptation_config_from(train_cfg)
    saved_adapt = dict(extra.get('adaptation_config', {}))
    saved_adapt.setdefault('read_mode', 'absolute_gated')
    saved_adapt.setdefault('read_scale', 1.0)
    if asdict(adapt_cfg) != saved_adapt:
        raise ValueError('Checkpoint adaptation metadata differs from its config.')

    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])
    probe_observations = jnp.asarray(
        _fixed_probe_observations(dataset, int(cfg.probe_observations))
    )
    extract = _make_extractor(params, model_cfg, adapt_cfg)
    split_limits = {
        'train': int(cfg.max_train_tasks),
        'latent_validation': int(cfg.max_latent_tasks),
        'family_validation': int(cfg.max_family_tasks),
    }
    selected_by_split = {
        split: _selected_tasks(dataset, split, limit)
        for split, limit in split_limits.items()
    }
    samplers = {
        split: MetaWorldTaskSampler(
            dataset, split=split, seed=int(cfg.seed) + 10_000 * split_index
        )
        for split_index, split in enumerate(split_limits)
    }
    task_units = sum(len(values) for values in selected_by_split.values())
    total_units = task_units * len(support_counts) * int(cfg.samples_per_task)

    rows: list[dict[str, Any]] = []
    feature_lists: dict[str, list[np.ndarray]] = {
        **{name: [] for name in _INFORMATION_REPRESENTATIONS},
        'oracle_query_gradient': [],
    }
    completed = 0
    for support_count in support_counts:
        for split, task_ids in selected_by_split.items():
            sampler = samplers[split]
            for task_id in task_ids:
                target_descriptor = dataset.task_descriptor(task_id)
                target_composition = family_composition(dataset.task_family(task_id))
                same_family_task = dataset.same_family_wrong_task(task_id, split)
                different_family_task = dataset.different_family_task(task_id, split)
                for sample_index in range(int(cfg.samples_per_task)):
                    correct_batch = sampler.build_batch(
                        1,
                        support_episodes=support_count,
                        query_episodes=int(cfg.query_episodes),
                        task_ids=[task_id],
                    )
                    correct_support = _remove_task_axis(
                        correct_batch['support'], _SUPPORT_FIELDS
                    )
                    query = {
                        name: jnp.asarray(value)
                        for name, value in _remove_task_axis(
                            correct_batch['query'], _QUERY_FIELDS
                        ).items()
                    }
                    source_batches = {
                        'same_family_wrong_instance': sampler.build_batch(
                            1,
                            support_episodes=support_count,
                            query_episodes=1,
                            task_ids=[same_family_task],
                        ),
                        'different_family_support': sampler.build_batch(
                            1,
                            support_episodes=support_count,
                            query_episodes=1,
                            task_ids=[different_family_task],
                        ),
                    }
                    source_supports = {
                        name: _remove_task_axis(batch['support'], _SUPPORT_FIELDS)
                        for name, batch in source_batches.items()
                    }
                    for condition_index, condition in enumerate(conditions):
                        source_task_id = {
                            'same_family_wrong_instance': same_family_task,
                            'different_family_support': different_family_task,
                        }.get(condition, task_id)
                        source = source_supports.get(condition, correct_support)
                        rng = np.random.default_rng(
                            int(cfg.seed)
                            + support_count * 10_000_000
                            + completed * 100
                            + condition_index
                        )
                        conditioned = condition_support(
                            condition,
                            correct_support,
                            source,
                            rng,
                        )
                        host = _host_result(
                            extract(
                                jax.tree_util.tree_map(jnp.asarray, conditioned),
                                query,
                                probe_observations,
                            )
                        )
                        target_latent, target_mask = _task_latent(dataset, task_id)
                        source_latent, source_mask = _task_latent(
                            dataset, source_task_id
                        )
                        source_descriptor = dataset.task_descriptor(source_task_id)
                        row = {
                            'target_task_id': task_id,
                            'target_family': dataset.task_family(task_id),
                            'target_split': split,
                            'support_task_id': source_task_id,
                            'support_family': dataset.task_family(source_task_id),
                            'target_instance_index': int(
                                target_descriptor['instance_index']
                            ),
                            'support_instance_index': int(
                                source_descriptor['instance_index']
                            ),
                            'target_motion_phases': list(target_composition.phases),
                            'target_motion_signature': target_composition.signature,
                            'target_terminal_phase': target_composition.phases[-1],
                            'target_gripper_mode': target_composition.gripper,
                            'target_manipulated_entity': (
                                target_composition.manipulated_entity
                            ),
                            'target_relation': target_composition.target_relation,
                            'target_structure': list(target_composition.structure),
                            'target_ml45_development_role': (
                                target_composition.ml45_development_role
                            ),
                            'support_count': support_count,
                            'sample_index': sample_index,
                            'condition': condition,
                            'support_episode_ids': np.asarray(
                                (
                                    correct_batch
                                    if source_task_id == task_id
                                    else source_batches[condition]
                                )['support']['episode_id'][0]
                            )
                            .astype(int)
                            .tolist(),
                            'query_episode_ids': np.asarray(
                                correct_batch['query']['episode_id'][0]
                            )
                            .astype(int)
                            .tolist(),
                            'target_task_latent': target_latent.tolist(),
                            'target_task_latent_mask': target_mask.tolist(),
                            'support_task_latent': source_latent.tolist(),
                            'support_task_latent_mask': source_mask.tolist(),
                            **{
                                name: float(value)
                                for name, value in host.items()
                                if np.asarray(value).shape == ()
                            },
                        }
                        rows.append(row)
                        feature_lists['raw_support_statistics'].append(
                            raw_support_statistics(conditioned)
                        )
                        for name in (
                            'first_write_gradient',
                            'oracle_query_gradient',
                            'final_fast_delta',
                            'read_action_delta',
                        ):
                            feature_lists[name].append(np.asarray(host[name]))
                    completed += 1
                    if completed % max(1, int(cfg.progress_every)) == 0:
                        _LOGGER.info(
                            '%s information extraction %d/%d task-support samples',
                            benchmark.label,
                            completed,
                            total_units,
                        )

    features = {
        name: np.stack(values).astype(np.float32)
        for name, values in feature_lists.items()
    }
    run_dir = (
        Path(str(cfg.output_dir)).expanduser().resolve()
        / f'{checkpoint_path.stem}_{datetime.now(UTC).strftime("%Y%m%d-%H%M%S")}'
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        run_dir / 'features.npz',
        **features,
        target_task_id=np.asarray([row['target_task_id'] for row in rows]),
        support_task_id=np.asarray([row['support_task_id'] for row in rows]),
        target_family=np.asarray([row['target_family'] for row in rows]),
        target_instance_index=np.asarray(
            [row['target_instance_index'] for row in rows], dtype=np.int16
        ),
        target_split=np.asarray([row['target_split'] for row in rows]),
        condition=np.asarray([row['condition'] for row in rows]),
        support_count=np.asarray([row['support_count'] for row in rows], dtype=np.int16),
        sample_index=np.asarray([row['sample_index'] for row in rows], dtype=np.int16),
    )
    _write_json(run_dir / 'records.json', rows)
    probes = _probe_summary(
        rows,
        features,
        support_counts=support_counts,
        conditions=conditions,
        ridge=float(cfg.ridge),
    )
    summary = {
        'integration': benchmark.integration_name,
        'benchmark_label': benchmark.label,
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_step': int(payload['step']),
        'checkpoint_type': extra.get('checkpoint_type'),
        'write_objective': str(adapt_cfg.write_objective),
        'read_mode': str(adapt_cfg.read_mode),
        'first_order': bool(adapt_cfg.first_order),
        'feature_rows': len(rows),
        'feature_dimensions': {
            name: int(value.shape[1]) for name, value in features.items()
        },
        'support_counts': list(support_counts),
        'conditions': list(conditions),
        'samples_per_task': int(cfg.samples_per_task),
        'selected_task_counts': {
            split: len(task_ids) for split, task_ids in selected_by_split.items()
        },
        'dataset': dataset.provenance(),
        'model_config': asdict(model_cfg),
        'adaptation_config': asdict(adapt_cfg),
        'analysis_config': config_to_dict(cfg),
        'scalar_means': _scalar_summary(rows),
        'scalar_means_by_split': {
            split: _scalar_summary(
                [row for row in rows if row['target_split'] == split]
            )
            for split in split_limits
            if any(row['target_split'] == split for row in rows)
        },
        'condition_comparisons': _condition_summary(rows, features),
        'condition_comparisons_by_split': {
            split: _condition_summary(rows, features, target_split=split)
            for split in split_limits
        },
        'probes': probes,
        'oracle_query_gradient_deployment_use': False,
        'random_projection_used': False,
    }
    visualization_cfg = cfg.get('visualization')
    if visualization_cfg is not None and bool(
        visualization_cfg.get('enabled', False)
    ):
        from icil_jax_rlbench.visualization.metaworld_update_information import (
            write_update_tsne_artifacts,
        )

        summary['visualization'] = write_update_tsne_artifacts(
            rows,
            features,
            run_dir / 'tsne',
            benchmark_label=benchmark.label,
            condition=str(visualization_cfg.condition),
            support_count=int(visualization_cfg.support_count),
            representations=tuple(visualization_cfg.representations),
            splits=tuple(visualization_cfg.splits),
            pca_components=int(visualization_cfg.pca_components),
            perplexities=tuple(visualization_cfg.perplexities),
            max_iter=int(visualization_cfg.max_iter),
            seed=int(visualization_cfg.seed),
        )
    _write_json(run_dir / 'summary.json', summary)
    (run_dir / 'report.md').write_text(_render_report(summary), encoding='utf-8')
    _plot_probe_scores(summary, run_dir / 'probe_scores.png')
    _LOGGER.info(
        '%s update-information diagnostic complete: %s', benchmark.label, run_dir
    )
    return run_dir


__all__ = [
    'analyze_metaworld_update_information',
    'raw_support_statistics',
    'ridge_predict',
]
