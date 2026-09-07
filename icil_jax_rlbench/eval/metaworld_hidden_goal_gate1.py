from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from phi_mujoco.evaluation import EvaluationConfig, EvaluationRunner

from icil_jax_rlbench.data.metaworld_hidden_goal import (
    MetaWorldTaskDataset,
    MetaWorldTaskSampler,
    benchmark_from_config,
)
from icil_jax_rlbench.eval.metaworld_policy import MetaWorldJaxPolicy
from icil_jax_rlbench.models.fast_weight_ttt import (
    initial_fast_state,
    predict_action,
    robotics_action_loss,
    tree_difference_norm,
    tree_l2_norm,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.metaworld_query_runner import (
    metaworld_model_config_from,
    validate_metaworld_query_config,
)
from icil_jax_rlbench.train.provenance import config_to_dict

CONDITIONS = (
    'no_update',
    'correct_support',
    'wrong_task_support',
    'shuffled_actions',
    'observations_only',
)
_LOGGER = logging.getLogger(__name__)


def _parameter_mask(params: Mapping[str, Any], subset: str) -> Mapping[str, Any]:
    if subset == 'action_heads':
        included = {'translation_head', 'gripper_head'}
    elif subset == 'query_policy':
        included = {'query_encoder', 'translation_head', 'gripper_head'}
    elif subset == 'all':
        included = set(params)
    else:
        raise ValueError("adapt_subset must be 'action_heads', 'query_policy', or 'all'.")

    def mask_group(value, selected: bool):
        return jax.tree_util.tree_map(lambda _: selected, value)

    return {
        name: mask_group(value, name in included) for name, value in params.items()
    }


def _ordinary_adapt(
    params: Mapping[str, Any],
    support: Mapping[str, jax.Array],
    *,
    model_cfg,
    parameter_mask: Mapping[str, Any],
    steps: int,
    learning_rate: float,
    clip_norm: float,
):
    def support_loss(value):
        prediction = predict_action(
            value,
            initial_fast_state(value),
            support['observation'],
            model_cfg,
            read_enabled=False,
        )
        loss, _ = robotics_action_loss(
            prediction,
            support['action'],
            support['write_mask'],
            model_cfg,
        )
        return loss

    def body(value, _):
        loss, gradient = jax.value_and_grad(support_loss)(value)
        gradient = jax.tree_util.tree_map(
            lambda grad, selected: grad if selected else jnp.zeros_like(grad),
            gradient,
            parameter_mask,
        )
        norm = tree_l2_norm(gradient)
        scale = (
            jnp.minimum(1.0, float(clip_norm) / (norm + 1e-8))
            if float(clip_norm) > 0.0
            else jnp.asarray(1.0, dtype=jnp.float32)
        )
        next_value = jax.tree_util.tree_map(
            lambda current, grad: current
            - float(learning_rate) * scale.astype(grad.dtype) * grad,
            value,
            gradient,
        )
        return next_value, {'support_loss': loss, 'gradient_norm': norm}

    return jax.lax.scan(body, params, xs=None, length=int(steps))


def _query_loss(params, query, model_cfg):
    prediction = predict_action(
        params,
        initial_fast_state(params),
        query['observation'],
        model_cfg,
        read_enabled=False,
    )
    return robotics_action_loss(
        prediction, query['action'], query['outer_loss_mask'], model_cfg
    )


def _remove_task_axis(section: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value[0]) for name, value in section.items()}


def _select_fields(
    section: Mapping[str, np.ndarray], names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    return {name: np.asarray(section[name]) for name in names}


def _condition_support(
    condition: str,
    correct_support: Mapping[str, np.ndarray],
    wrong_support: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    source = wrong_support if condition == 'wrong_task_support' else correct_support
    support = {name: np.array(value, copy=True) for name, value in source.items()}
    if condition in ('correct_support', 'wrong_task_support'):
        return support
    if condition == 'shuffled_actions':
        valid = np.asarray(support['write_mask'], dtype=np.bool_)
        values = np.array(support['action'][valid], copy=True)
        support['action'][valid] = values[rng.permutation(len(values))]
        return support
    if condition == 'observations_only':
        support['action'].fill(0.0)
        return support
    raise ValueError(f'Unknown adapting support condition {condition!r}.')


def _confidence_interval(successes: int, count: int) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    probability = successes / count
    standard_error = np.sqrt(
        max(probability * (1.0 - probability), 1e-12) / count
    )
    return (
        float(max(0.0, probability - 1.96 * standard_error)),
        float(min(1.0, probability + 1.96 * standard_error)),
    )


def _mean_confidence_interval(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, mean
    margin = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(array.size)
    return mean - margin, mean + margin


def _fresh_evaluation_seed(
    requested_seed: int,
    episodes: int,
    dataset: MetaWorldTaskDataset,
) -> int:
    used = {episode.seed for episode in dataset.bundle.episodes}
    candidate = int(requested_seed)
    while any(candidate + offset in used for offset in range(int(episodes))):
        candidate += int(episodes)
    if candidate + int(episodes) - 1 > 2**32 - 1:
        raise ValueError('Could not choose fresh closed-loop evaluation seeds.')
    return candidate


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def evaluate_metaworld_gate1(cfg: ConfigDict) -> Path:
    if not str(cfg.checkpoint_path):
        raise ValueError('checkpoint_path is required.')
    payload = load_checkpoint(str(cfg.checkpoint_path))
    extra = payload.get('extra', {})
    train_cfg = ConfigDict(payload['config'])
    benchmark = benchmark_from_config(train_cfg)
    requested_integration = str(cfg.get('integration', ''))
    if requested_integration and requested_integration != benchmark.integration_name:
        raise ValueError(
            f'Evaluation config requests {requested_integration!r}, but checkpoint '
            f'belongs to {benchmark.integration_name!r}.'
        )
    if extra.get('checkpoint_type') != benchmark.query_mode:
        raise ValueError(
            f'Gate 1 requires a {benchmark.label} query-only checkpoint.'
        )
    validate_metaworld_query_config(train_cfg)
    cache_root = str(cfg.cache_root) or str(train_cfg.dataset.cache_root)
    checkpoint_normalization = extra.get('normalization')
    if checkpoint_normalization is None:
        raise ValueError('Checkpoint does not contain normalization statistics.')
    if not isinstance(checkpoint_normalization, Mapping):
        raise TypeError('Checkpoint normalization statistics must be a mapping.')
    dataset = MetaWorldTaskDataset(
        cache_root,
        integration_name=benchmark.integration_name,
        protocol=str(train_cfg.dataset.get('protocol', 'default')),
        horizon_buckets=tuple(train_cfg.dataset.get('horizon_buckets', ())),
        normalization=checkpoint_normalization,
        cache_prepared_episodes=bool(cfg.cache_prepared_episodes),
    )
    if dataset.bundle.data_sha256 != extra.get('cache_data_sha256'):
        raise ValueError('Evaluation cache differs from the checkpoint training cache.')
    if dataset.normalization_id != extra.get('normalizer_id'):
        raise ValueError('Evaluation normalization differs from the checkpoint.')
    model_cfg = metaworld_model_config_from(train_cfg, dataset)
    if asdict(model_cfg) != extra.get('model_config'):
        raise ValueError('Checkpoint model metadata differs from its resolved config.')
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])

    split = str(cfg.split)
    all_task_ids = dataset.task_ids(split)
    task_ids = dataset.balanced_task_ids(split, int(cfg.max_tasks))
    if len(task_ids) < 1:
        raise ValueError(f'No tasks selected from split {split!r}.')
    conditions = tuple(str(condition) for condition in cfg.conditions)
    unknown_conditions = set(conditions) - set(CONDITIONS)
    if unknown_conditions:
        raise ValueError(f'Unknown Gate 1 conditions: {sorted(unknown_conditions)}.')
    if 'no_update' not in conditions or 'correct_support' not in conditions:
        raise ValueError("Gate 1 requires both 'no_update' and 'correct_support'.")

    sampler = MetaWorldTaskSampler(
        dataset, split=split, seed=int(cfg.seed) + 100
    )
    parameter_mask = _parameter_mask(params, str(cfg.adapt_subset))
    adapt = jax.jit(
        lambda value, support: _ordinary_adapt(
            value,
            support,
            model_cfg=model_cfg,
            parameter_mask=parameter_mask,
            steps=int(cfg.inner_steps),
            learning_rate=float(cfg.inner_lr),
            clip_norm=float(cfg.inner_grad_clip_norm),
        )
    )
    rng = np.random.default_rng(int(cfg.seed) + 200)
    closed_loop_episodes = int(cfg.closed_loop_episodes)
    evaluation_seed = _fresh_evaluation_seed(
        int(cfg.closed_loop_base_seed), closed_loop_episodes, dataset
    )

    output_dir = (
        Path(cfg.output_dir).expanduser().resolve()
        / f'{Path(cfg.checkpoint_path).stem}_{datetime.now(UTC).strftime("%Y%m%d-%H%M%S")}'
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    records: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in conditions
    }
    base_integration = benchmark.create_integration(
        catalog_seed=dataset.task_index.catalog.catalog_seed
    )
    for task_position, task_id in enumerate(task_ids):
        correct = sampler.build_batch(
            1,
            support_episodes=int(cfg.support_episodes),
            query_episodes=int(cfg.offline_query_episodes),
            task_ids=[task_id],
        )
        wrong_task_id = all_task_ids[
            (all_task_ids.index(task_id) + 1) % len(all_task_ids)
        ]
        wrong = sampler.build_batch(
            1,
            support_episodes=int(cfg.support_episodes),
            query_episodes=1,
            task_ids=[wrong_task_id],
        )
        correct_support = _select_fields(
            _remove_task_axis(correct['support']),
            ('observation', 'action', 'write_mask'),
        )
        wrong_support = _select_fields(
            _remove_task_axis(wrong['support']),
            ('observation', 'action', 'write_mask'),
        )
        query = jax.tree_util.tree_map(
            jnp.asarray,
            _select_fields(
                _remove_task_axis(correct['query']),
                ('observation', 'action', 'outer_loss_mask'),
            ),
        )
        for condition in conditions:
            if condition == 'no_update':
                adapted_params = params
                trace = {
                    'support_loss': jnp.zeros((0,), dtype=jnp.float32),
                    'gradient_norm': jnp.zeros((0,), dtype=jnp.float32),
                }
            else:
                support = _condition_support(
                    condition, correct_support, wrong_support, rng
                )
                adapted_params, trace = adapt(
                    params, jax.tree_util.tree_map(jnp.asarray, support)
                )
            offline_loss, offline_metrics = _query_loss(
                adapted_params, query, model_cfg
            )
            integration = base_integration.for_task(task_id)
            policy = MetaWorldJaxPolicy(
                integration=integration,
                params=adapted_params,
                model_cfg=model_cfg,
                normalization=dataset.normalization,
            )
            save_rollouts = bool(cfg.save_rollout_artifacts) or bool(cfg.record_video)
            rollout_directory = (
                output_dir / 'rollouts' / task_id / condition
                if save_rollouts
                else None
            )
            result = EvaluationRunner(
                integration,
                policy,
                EvaluationConfig(
                    episodes=closed_loop_episodes,
                    base_seed=evaluation_seed,
                    max_steps=(
                        None
                        if int(cfg.closed_loop_max_steps) <= 0
                        else int(cfg.closed_loop_max_steps)
                    ),
                    run_directory=rollout_directory,
                    record_video=bool(cfg.record_video),
                    render_width=int(cfg.render_width),
                    render_height=int(cfg.render_height),
                ),
            ).run()
            support_trace = np.asarray(jax.device_get(trace['support_loss']))
            gradient_trace = np.asarray(jax.device_get(trace['gradient_norm']))
            records[condition].append(
                {
                    'task_id': task_id,
                    'wrong_support_task_id': (
                        wrong_task_id if condition == 'wrong_task_support' else None
                    ),
                    'offline_query_loss': float(offline_loss),
                    'translation_loss': float(offline_metrics['translation_loss']),
                    'gripper_loss': float(offline_metrics['gripper_loss']),
                    'parameter_update_norm': float(
                        tree_difference_norm(adapted_params, params)
                    ),
                    'support_loss_first': (
                        None if support_trace.size == 0 else float(support_trace[0])
                    ),
                    'support_loss_last': (
                        None if support_trace.size == 0 else float(support_trace[-1])
                    ),
                    'gradient_norm_first': (
                        None if gradient_trace.size == 0 else float(gradient_trace[0])
                    ),
                    'closed_loop': result.as_dict(),
                }
            )
        _LOGGER.info(
            '%s Gate 1 task %d/%d complete: %s',
            benchmark.label,
            task_position + 1,
            len(task_ids),
            task_id,
        )

    aggregate: dict[str, dict[str, Any]] = {}
    for condition, condition_records in records.items():
        successes = sum(
            int(record['closed_loop']['successful_episodes'])
            for record in condition_records
        )
        attempts = sum(
            int(record['closed_loop']['attempted_episodes'])
            for record in condition_records
        )
        aggregate[condition] = {
            'offline_query_loss': float(
                np.mean([record['offline_query_loss'] for record in condition_records])
            ),
            'parameter_update_norm': float(
                np.mean([record['parameter_update_norm'] for record in condition_records])
            ),
            'success_rate': successes / attempts,
            'success_rate_95pct_ci': list(_confidence_interval(successes, attempts)),
            'successful_episodes': successes,
            'attempted_episodes': attempts,
        }
    no_update_by_task = {
        record['task_id']: record for record in records['no_update']
    }
    for condition, value in aggregate.items():
        if condition == 'no_update':
            continue
        offline_gains = [
            no_update_by_task[record['task_id']]['offline_query_loss']
            - record['offline_query_loss']
            for record in records[condition]
        ]
        success_gains = [
            record['closed_loop']['success_rate']
            - no_update_by_task[record['task_id']]['closed_loop']['success_rate']
            for record in records[condition]
        ]
        value['gain_over_no_update'] = {
            'offline_query_loss': float(np.mean(offline_gains)),
            'offline_query_loss_95pct_ci': list(
                _mean_confidence_interval(offline_gains)
            ),
            'success_rate': float(np.mean(success_gains)),
            'success_rate_95pct_ci': list(
                _mean_confidence_interval(success_gains)
            ),
        }

    summary = {
        'gate': 'ordinary_adaptation_upper_bound',
        'checkpoint': str(Path(cfg.checkpoint_path).expanduser().resolve()),
        'checkpoint_step': int(payload['step']),
        'integration': benchmark.integration_name,
        'dataset': dataset.provenance(),
        'split': split,
        'task_ids': list(task_ids),
        'adapt_subset': str(cfg.adapt_subset),
        'inner_steps': int(cfg.inner_steps),
        'inner_lr': float(cfg.inner_lr),
        'inner_grad_clip_norm': float(cfg.inner_grad_clip_norm),
        'support_episodes': int(cfg.support_episodes),
        'offline_query_episodes': int(cfg.offline_query_episodes),
        'closed_loop_episodes': closed_loop_episodes,
        'closed_loop_base_seed': evaluation_seed,
        'matched_closed_loop_seeds_across_conditions': True,
        'aggregate': aggregate,
        'per_task': records,
    }
    _write_json(output_dir / 'summary.json', summary)
    _write_json(output_dir / 'resolved_eval_config.json', config_to_dict(cfg))
    _LOGGER.info('%s Gate 1 results: %s', benchmark.label, output_dir)
    return output_dir


__all__ = [
    'CONDITIONS',
    '_ordinary_adapt',
    '_parameter_mask',
    'evaluate_metaworld_gate1',
]
