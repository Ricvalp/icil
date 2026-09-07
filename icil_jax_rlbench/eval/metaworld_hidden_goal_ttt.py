from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from phi_mujoco.evaluation import EvaluationConfig, EvaluationRunner
from phi_mujoco.integrations import family_composition

from icil_jax_rlbench.data.metaworld_hidden_goal import (
    MetaWorldTaskDataset,
    MetaWorldTaskSampler,
    benchmark_from_config,
)
from icil_jax_rlbench.eval.metaworld_policy import MetaWorldJaxPolicy
from icil_jax_rlbench.eval.support_controls import (
    SUPPORTED_CONDITIONS,
    bootstrap_mean_confidence_interval,
    condition_support,
    confidence_interval,
    random_fast_state_with_matched_delta,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    adapt_fast_state,
    initial_fast_state,
    named_fast_tensor_metrics,
    query_imitation_loss,
    tree_difference_norm,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.metaworld_query_runner import metaworld_model_config_from
from icil_jax_rlbench.train.metaworld_ttt_runner import validate_metaworld_ttt_config
from icil_jax_rlbench.train.provenance import config_to_dict
from icil_jax_rlbench.train.ttt_config import adaptation_config_from

_LOGGER = logging.getLogger(__name__)
_SUPPORT_FIELDS = ('observation', 'action', 'next_observation', 'write_mask')
_QUERY_FIELDS = ('observation', 'action', 'outer_loss_mask')


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _remove_task_axis(
    section: Mapping[str, np.ndarray],
    fields: Sequence[str],
) -> dict[str, np.ndarray]:
    return {name: np.asarray(section[name][0]) for name in fields}


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


def _trace_summary(trace: Mapping[str, jax.Array]) -> dict[str, dict[str, float]]:
    host = jax.device_get(trace)
    result = {}
    for name, value in host.items():
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size:
            result[name] = {
                'first': float(array[0]),
                'last': float(array[-1]),
                'mean': float(np.mean(array)),
                'maximum': float(np.max(array)),
            }
    return result


def _host_scalars(values: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(np.asarray(value))
        for name, value in jax.device_get(values).items()
    }


def _paired_gain_summary(
    records: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    task_ids = [str(record['task_id']) for record in records]
    baseline_ids = [str(record['task_id']) for record in baseline]
    if task_ids != baseline_ids:
        raise ValueError('Paired MetaWorld records are not aligned by task ID.')
    values = {
        'success_rate': np.asarray(
            [
                float(record['closed_loop']['success_rate'])
                - float(base['closed_loop']['success_rate'])
                for record, base in zip(records, baseline, strict=True)
            ]
        ),
        'offline_loss_reduction': np.asarray(
            [
                float(base['offline_query_loss'])
                - float(record['offline_query_loss'])
                for record, base in zip(records, baseline, strict=True)
            ]
        ),
    }
    summary: dict[str, Any] = {'task_count': len(records)}
    for index, (name, array) in enumerate(values.items()):
        mean, low, high = bootstrap_mean_confidence_interval(
            array, seed=int(seed) + index
        )
        summary[name] = {
            'mean': mean,
            'ci95': [low, high],
            'per_task': array.tolist(),
        }
    return summary


def _aggregate(
    records: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    *,
    seed: int,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for support_index, (count_key, by_condition) in enumerate(records.items()):
        aggregate[count_key] = {}
        for condition, condition_records in by_condition.items():
            successes = sum(
                int(record['closed_loop']['successful_episodes'])
                for record in condition_records
            )
            attempts = sum(
                int(record['closed_loop']['attempted_episodes'])
                for record in condition_records
            )
            low, high = confidence_interval(successes, attempts)
            aggregate[count_key][condition] = {
                'success_rate': successes / max(1, attempts),
                'successful_episodes': successes,
                'attempted_episodes': attempts,
                'success_ci95': [low, high],
                'mean_offline_query_loss': float(
                    np.mean(
                        [record['offline_query_loss'] for record in condition_records]
                    )
                ),
                'mean_fast_delta_norm': float(
                    np.mean(
                        [record['fast_delta_norm'] for record in condition_records]
                    )
                ),
            }
        baseline = by_condition.get('no_update')
        if baseline is not None:
            aggregate[count_key]['paired_gain_over_no_update'] = {
                condition: _paired_gain_summary(
                    condition_records,
                    baseline,
                    seed=int(seed) + support_index * 100 + condition_index,
                )
                for condition_index, (condition, condition_records) in enumerate(
                    by_condition.items()
                )
                if condition != 'no_update'
            }
        correct = aggregate[count_key].get('correct_support')
        no_update = aggregate[count_key].get('no_update')
        if correct is not None and no_update is not None:
            aggregate[count_key]['correct_support_gain'] = {
                'success_rate': correct['success_rate'] - no_update['success_rate'],
                'offline_loss_reduction': (
                    no_update['mean_offline_query_loss']
                    - correct['mean_offline_query_loss']
                ),
            }
        if correct is not None:
            for wrong_name in (
                'wrong_task_support',
                'same_family_wrong_instance',
                'different_family_support',
            ):
                wrong = aggregate[count_key].get(wrong_name)
                if wrong is None:
                    continue
                aggregate[count_key][f'correct_vs_{wrong_name}_gap'] = {
                    'success_rate': correct['success_rate'] - wrong['success_rate'],
                    'offline_loss': (
                        wrong['mean_offline_query_loss']
                        - correct['mean_offline_query_loss']
                    ),
                }
    return aggregate


def _grouped_aggregate(
    records: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    *,
    metadata_key: str,
    seed: int,
) -> dict[str, Any]:
    labels = sorted(
        {
            label
            for by_condition in records.values()
            for condition_records in by_condition.values()
            for record in condition_records
            for label in (
                record[metadata_key]
                if isinstance(record[metadata_key], list)
                else [record[metadata_key]]
            )
        }
    )
    grouped: dict[str, Any] = {}
    for label_index, label in enumerate(labels):
        selected = {
            count: {
                condition: [
                    record
                    for record in condition_records
                    if (
                        label in record[metadata_key]
                        if isinstance(record[metadata_key], list)
                        else record[metadata_key] == label
                    )
                ]
                for condition, condition_records in by_condition.items()
            }
            for count, by_condition in records.items()
        }
        if any(
            not condition_records
            for by_condition in selected.values()
            for condition_records in by_condition.values()
        ):
            continue
        grouped[str(label)] = _aggregate(
            selected, seed=int(seed) + label_index * 1_000
        )
    return grouped


def evaluate_metaworld_ttt(cfg: ConfigDict) -> Path:
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
    if extra.get('checkpoint_type') != benchmark.ttt_mode:
        raise ValueError(
            f'Evaluation requires a {benchmark.label} fast-weight TTT checkpoint.'
        )
    validate_metaworld_ttt_config(train_cfg)

    split = str(cfg.split)
    conditions = tuple(str(condition) for condition in cfg.conditions)
    unknown = sorted(set(conditions) - set(SUPPORTED_CONDITIONS))
    if unknown:
        raise ValueError(f'Unknown support conditions: {unknown}.')
    if 'no_update' not in conditions or 'correct_support' not in conditions:
        raise ValueError("Evaluation requires 'no_update' and 'correct_support'.")
    support_counts = tuple(int(value) for value in cfg.support_counts)
    if not support_counts or any(value < 1 for value in support_counts):
        raise ValueError('support_counts must contain positive values.')
    if int(cfg.offline_query_episodes) < 1:
        raise ValueError('offline_query_episodes must be positive.')
    if int(cfg.closed_loop_episodes) < 1:
        raise ValueError('closed_loop_episodes must be positive.')

    cache_root = str(cfg.cache_root) or str(train_cfg.dataset.cache_root)
    normalization = extra.get('normalization')
    if not isinstance(normalization, Mapping):
        raise ValueError('Checkpoint lacks normalization statistics.')
    dataset = MetaWorldTaskDataset(
        cache_root,
        integration_name=benchmark.integration_name,
        protocol=str(train_cfg.dataset.get('protocol', 'default')),
        horizon_buckets=tuple(train_cfg.dataset.get('horizon_buckets', ())),
        normalization=normalization,
        cache_prepared_episodes=bool(cfg.cache_prepared_episodes),
    )
    if dataset.bundle.data_sha256 != extra.get('cache_data_sha256'):
        raise ValueError('Evaluation cache differs from the checkpoint cache.')
    if dataset.normalization_id != extra.get('normalizer_id'):
        raise ValueError('Evaluation normalization differs from the checkpoint.')
    if max(support_counts) + int(cfg.offline_query_episodes) > int(
        dataset.task_index.episodes_per_task
    ):
        raise ValueError('Requested support/query episodes exceed episodes per task.')

    model_cfg = metaworld_model_config_from(train_cfg, dataset)
    if asdict(model_cfg) != extra.get('model_config'):
        raise ValueError('Checkpoint model metadata differs from its config.')
    checkpoint_adapt_cfg = adaptation_config_from(train_cfg)
    saved_adapt_cfg = dict(extra.get('adaptation_config', {}))
    saved_adapt_cfg.setdefault('read_mode', 'absolute_gated')
    saved_adapt_cfg.setdefault('read_scale', 1.0)
    if asdict(checkpoint_adapt_cfg) != saved_adapt_cfg:
        raise ValueError('Checkpoint adaptation metadata differs from its config.')
    adapt_cfg = checkpoint_adapt_cfg
    if int(cfg.write_steps_per_segment_override) > 0:
        adapt_cfg = replace(
            adapt_cfg,
            write_steps_per_segment=int(cfg.write_steps_per_segment_override),
        )
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])

    all_task_ids = dataset.task_ids(split)
    task_ids = dataset.balanced_task_ids(split, int(cfg.max_tasks))
    if not task_ids:
        raise ValueError(f'No tasks selected from split {split!r}.')
    evaluation_seed = _fresh_evaluation_seed(
        int(cfg.closed_loop_base_seed),
        int(cfg.closed_loop_episodes),
        dataset,
    )
    checkpoint_name = Path(cfg.checkpoint_path).stem
    run_dir = (
        Path(cfg.output_dir).expanduser().resolve()
        / f'{checkpoint_name}_{datetime.now(UTC).strftime("%Y%m%d-%H%M%S")}'
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    sampler = MetaWorldTaskSampler(
        dataset, split=split, seed=int(cfg.seed) + 8001
    )
    adapt = jax.jit(
        lambda value, support: adapt_fast_state(
            value, support, model_cfg, adapt_cfg
        )
    )
    offline_query = jax.jit(
        lambda value, fast_state, query: query_imitation_loss(
            value, fast_state, query, model_cfg, adapt_cfg
        )
    )
    base_integration = benchmark.create_integration(
        catalog_seed=dataset.task_index.catalog.catalog_seed
    )
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(count): {condition: [] for condition in conditions}
        for count in support_counts
    }

    for support_index, support_count in enumerate(support_counts):
        count_key = str(support_count)
        for task_position, task_id in enumerate(task_ids):
            correct_batch = sampler.build_batch(
                1,
                support_episodes=support_count,
                query_episodes=int(cfg.offline_query_episodes),
                task_ids=[task_id],
            )
            wrong_task_id = all_task_ids[
                (all_task_ids.index(task_id) + 1) % len(all_task_ids)
            ]
            control_task_ids = {'wrong_task_support': wrong_task_id}
            if 'same_family_wrong_instance' in conditions:
                control_task_ids['same_family_wrong_instance'] = (
                    dataset.same_family_wrong_task(task_id, split)
                )
            if 'different_family_support' in conditions:
                control_task_ids['different_family_support'] = (
                    dataset.different_family_task(task_id, split)
                )
            control_batches = {
                condition: sampler.build_batch(
                    1,
                    support_episodes=support_count,
                    query_episodes=1,
                    task_ids=[source_task_id],
                )
                for condition, source_task_id in control_task_ids.items()
                if condition in conditions
            }
            correct_support_np = _remove_task_axis(
                correct_batch['support'], _SUPPORT_FIELDS
            )
            control_support = {
                condition: _remove_task_axis(batch['support'], _SUPPORT_FIELDS)
                for condition, batch in control_batches.items()
            }
            query = {
                name: jnp.asarray(value)
                for name, value in _remove_task_axis(
                    correct_batch['query'], _QUERY_FIELDS
                ).items()
            }
            correct_support = jax.tree_util.tree_map(
                jnp.asarray, correct_support_np
            )
            correct_adapted, correct_trace = adapt(params, correct_support)
            task_descriptor = dataset.task_descriptor(task_id)
            task_family = dataset.task_family(task_id)
            composition = (
                family_composition(task_family) if benchmark.family_aware else None
            )

            for condition in conditions:
                condition_index = SUPPORTED_CONDITIONS.index(condition)
                control_rng = np.random.default_rng(
                    int(cfg.seed)
                    + 100_000 * support_index
                    + 1_000 * task_position
                    + condition_index
                )
                source_task_id = task_id
                source_episode_ids = np.asarray(
                    correct_batch['support']['episode_id'][0]
                ).astype(int).tolist()
                if condition == 'no_update':
                    fast_state = initial_fast_state(params)
                    trace = {}
                    source_task_id = ''
                    source_episode_ids = []
                elif condition == 'correct_support':
                    fast_state = correct_adapted
                    trace = correct_trace
                elif condition == 'random_update_matched_norm':
                    fast_state = random_fast_state_with_matched_delta(
                        params, correct_adapted, control_rng
                    )
                    trace = {}
                else:
                    selected_wrong = control_support.get(
                        condition, correct_support_np
                    )
                    conditioned = condition_support(
                        condition,
                        correct_support_np,
                        selected_wrong,
                        control_rng,
                    )
                    fast_state, trace = adapt(
                        params, jax.tree_util.tree_map(jnp.asarray, conditioned)
                    )
                    if condition in control_task_ids:
                        source_task_id = control_task_ids[condition]
                        source_episode_ids = np.asarray(
                            control_batches[condition]['support']['episode_id'][0]
                        ).astype(int).tolist()

                offline_loss, offline_metrics = offline_query(
                    params, fast_state, query
                )
                integration = base_integration.for_task(task_id)
                policy = MetaWorldJaxPolicy(
                    integration=integration,
                    params=params,
                    fast_state=fast_state,
                    read_enabled=bool(adapt_cfg.read_enabled),
                    model_cfg=model_cfg,
                    normalization=dataset.normalization,
                    read_mode=str(adapt_cfg.read_mode),
                    read_scale=float(adapt_cfg.read_scale),
                )
                save_rollouts = bool(cfg.save_rollout_artifacts) or bool(
                    cfg.record_video
                )
                rollout_directory = (
                    run_dir
                    / 'rollouts'
                    / f'support-{support_count}'
                    / task_id
                    / condition
                    if save_rollouts
                    else None
                )
                closed_loop = EvaluationRunner(
                    integration,
                    policy,
                    EvaluationConfig(
                        episodes=int(cfg.closed_loop_episodes),
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
                fast_metrics = named_fast_tensor_metrics(
                    params, fast_state, model_cfg
                )
                records[count_key][condition].append(
                    {
                        'task_id': task_id,
                        'task_family': task_family,
                        'task_motion_phases': (
                            [] if composition is None else list(composition.phases)
                        ),
                        'task_composition': (
                            None
                            if composition is None
                            else {
                                'signature': composition.signature,
                                'gripper': composition.gripper,
                                'manipulated_entity': composition.manipulated_entity,
                                'target_relation': composition.target_relation,
                                'structure': list(composition.structure),
                                'ml45_development_role': (
                                    composition.ml45_development_role
                                ),
                            }
                        ),
                        'privileged_task_descriptor': task_descriptor,
                        'support_source_task_id': source_task_id or None,
                        'support_episode_ids': source_episode_ids,
                        'offline_query_episode_ids': np.asarray(
                            correct_batch['query']['episode_id'][0]
                        ).astype(int).tolist(),
                        'offline_query_loss': float(jax.device_get(offline_loss)),
                        'offline_query_metrics': _host_scalars(offline_metrics),
                        'fast_delta_norm': float(
                            jax.device_get(
                                tree_difference_norm(
                                    fast_state, initial_fast_state(params)
                                )
                            )
                        ),
                        'correct_reference_delta_norm': float(
                            jax.device_get(
                                tree_difference_norm(
                                    correct_adapted, initial_fast_state(params)
                                )
                            )
                        ),
                        'write_trace': _trace_summary(trace),
                        'fast_tensor_metrics': _host_scalars(fast_metrics),
                        'closed_loop': closed_loop.as_dict(),
                    }
                )
            _LOGGER.info(
                '%s TTT support=%d task %d/%d complete: %s',
                benchmark.label,
                support_count,
                task_position + 1,
                len(task_ids),
                task_id,
            )

    aggregate = _aggregate(records, seed=int(cfg.seed) + 50_000)
    aggregate_by_family = _grouped_aggregate(
        records,
        metadata_key='task_family',
        seed=int(cfg.seed) + 60_000,
    )
    aggregate_by_motion_phase = _grouped_aggregate(
        records,
        metadata_key='task_motion_phases',
        seed=int(cfg.seed) + 70_000,
    )
    summary = {
        'gate': 'metaworld_hidden_goal_adaptation_only_generalization',
        'checkpoint_path': str(Path(cfg.checkpoint_path).expanduser().resolve()),
        'checkpoint_step': int(payload['step']),
        'integration': benchmark.integration_name,
        'checkpoint_adaptation_config': asdict(checkpoint_adapt_cfg),
        'evaluation_adaptation_config': asdict(adapt_cfg),
        'dataset': dataset.provenance(),
        'split': split,
        'task_ids': list(task_ids),
        'support_counts': list(support_counts),
        'offline_query_episodes': int(cfg.offline_query_episodes),
        'closed_loop_episodes': int(cfg.closed_loop_episodes),
        'closed_loop_base_seed': evaluation_seed,
        'closed_loop_seeds_fresh_relative_to_cache': True,
        'matched_closed_loop_seeds_across_conditions': True,
        'support_query_episode_overlap': False,
        'goal_provided_to_policy': False,
        'reset_policy': 'reset_to_w0_for_each_task_and_condition',
        'query_fast_state_policy': 'freeze_after_support_and_reuse_across_query_episodes',
        'paired_inference_unit': 'held_out_family_instance',
        'paired_ci_method': (
            '10000-sample percentile bootstrap over held-out family instances'
        ),
        'aggregate': aggregate,
        'aggregate_by_family': aggregate_by_family,
        'aggregate_by_motion_phase': aggregate_by_motion_phase,
        'per_support_count': records,
    }
    _write_json(run_dir / 'resolved_eval_config.json', config_to_dict(cfg))
    _write_json(run_dir / 'summary.json', summary)
    _LOGGER.info(
        '%s fast-weight TTT evaluation complete: %s', benchmark.label, run_dir
    )
    return run_dir


__all__ = ['evaluate_metaworld_ttt']
