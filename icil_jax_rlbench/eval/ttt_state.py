from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from absl import app
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict, config_flags
import numpy as np

from icil_jax_rlbench.data.hidden_goal import (
    HiddenGoalMetaSampler,
    HiddenGoalTaskBank,
    StateNormalizer,
)
from icil_jax_rlbench.eval.ttt_state_common import (
    SUPPORTED_CONDITIONS,
    bootstrap_mean_confidence_interval,
    condition_support,
    confidence_interval,
    random_fast_state_with_matched_delta,
    remove_task_axis,
    rollout,
    to_jax,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    adapt_fast_state,
    initial_fast_state,
    query_imitation_loss,
    tree_difference_norm,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.provenance import config_to_dict
from icil_jax_rlbench.train.ttt_runner import (
    adaptation_config_from,
    fast_weight_config_from,
    hidden_goal_config_from,
    validate_adaptation_only_config,
)


_CONFIG = config_flags.DEFINE_config_file(
    'config', None, 'State TTT closed-loop evaluation config.', lock_config=False
)


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _task_success_rate(record: Dict[str, Any]) -> float:
    rollouts = record['rollouts']
    return float(np.mean([float(rollout['success']) for rollout in rollouts]))


def _task_final_distance(record: Dict[str, Any]) -> float:
    return float(
        np.mean([float(rollout['final_distance']) for rollout in record['rollouts']])
    )


def _paired_gain_summary(
    records: list[Dict[str, Any]],
    baseline_records: list[Dict[str, Any]],
    *,
    seed: int,
) -> Dict[str, Any]:
    if len(records) != len(baseline_records):
        raise ValueError('Paired Gate 3 records must contain the same tasks.')
    task_ids = [int(record['task_id']) for record in records]
    baseline_task_ids = [int(record['task_id']) for record in baseline_records]
    if task_ids != baseline_task_ids:
        raise ValueError('Paired Gate 3 records are not aligned by task ID.')

    values = {
        'success_rate': np.asarray(
            [
                _task_success_rate(record) - _task_success_rate(baseline)
                for record, baseline in zip(records, baseline_records)
            ]
        ),
        'offline_loss_reduction': np.asarray(
            [
                float(baseline['offline_loss']) - float(record['offline_loss'])
                for record, baseline in zip(records, baseline_records)
            ]
        ),
        'final_distance_reduction': np.asarray(
            [
                _task_final_distance(baseline) - _task_final_distance(record)
                for record, baseline in zip(records, baseline_records)
            ]
        ),
    }
    summary: Dict[str, Any] = {'task_count': len(records)}
    for metric_index, (name, metric_values) in enumerate(values.items()):
        mean, low, high = bootstrap_mean_confidence_interval(
            metric_values,
            seed=int(seed) + metric_index,
        )
        summary[name] = {
            'mean': mean,
            'ci95': [low, high],
            'per_task': metric_values.tolist(),
        }
    return summary


def evaluate_ttt_state(cfg: ConfigDict) -> Path:
    if not str(cfg.checkpoint_path):
        raise ValueError('config.checkpoint_path is required.')
    conditions = tuple(str(value) for value in cfg.conditions)
    unknown = sorted(set(conditions) - set(SUPPORTED_CONDITIONS))
    if unknown:
        raise ValueError(f'Unknown evaluation conditions: {unknown}')
    payload = load_checkpoint(str(cfg.checkpoint_path))
    train_cfg = ConfigDict(payload['config'])
    validate_adaptation_only_config(train_cfg)
    benchmark_cfg = hidden_goal_config_from(train_cfg)
    model_cfg = fast_weight_config_from(train_cfg, benchmark_cfg)
    adapt_cfg = adaptation_config_from(train_cfg)
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])
    normalizer_payload = payload.get('extra', {}).get('normalizer')
    if normalizer_payload is None:
        raise ValueError('TTT checkpoint does not contain its training normalizer.')
    normalizer = StateNormalizer.from_dict(normalizer_payload)
    task_bank = HiddenGoalTaskBank(benchmark_cfg)
    split = str(cfg.split)
    num_tasks = task_bank.num_tasks(split)
    if int(cfg.max_tasks) > 0:
        num_tasks = min(num_tasks, int(cfg.max_tasks))
    sampler = HiddenGoalMetaSampler(
        benchmark_cfg,
        task_bank,
        normalizer,
        split=split,
        seed=int(cfg.seed) + 8001,
    )
    rng = np.random.default_rng(int(cfg.seed) + 9001)
    results: Dict[str, Dict[str, list[Dict[str, Any]]]] = {}
    per_task: Dict[str, Any] = {}

    for support_count in tuple(int(value) for value in cfg.support_counts):
        count_key = str(support_count)
        results[count_key] = {condition: [] for condition in conditions}
        per_task[count_key] = {}
        for local_task_id in range(num_tasks):
            query_count = int(cfg.query_episodes_per_task)
            correct_batch = sampler.build_batch(
                1,
                support_episodes=support_count,
                query_episodes=query_count,
                task_ids=[local_task_id],
            )
            wrong_task_id = (local_task_id + 1) % task_bank.num_tasks(split)
            wrong_batch = sampler.build_batch(
                1,
                support_episodes=support_count,
                query_episodes=1,
                task_ids=[wrong_task_id],
            )
            correct_support = remove_task_axis(correct_batch['support'])
            wrong_support = remove_task_axis(wrong_batch['support'])
            query = remove_task_axis(correct_batch['query'])
            goal = task_bank.goal(split, local_task_id)
            correct_adapted, _ = adapt_fast_state(
                params, to_jax(correct_support), model_cfg, adapt_cfg
            )
            task_result = {}
            for condition in conditions:
                conditioned_support = condition_support(
                    condition, correct_support, wrong_support, rng
                )
                if condition == 'no_update':
                    fast_state = initial_fast_state(params)
                elif condition == 'random_update_matched_norm':
                    fast_state = random_fast_state_with_matched_delta(
                        params, correct_adapted, rng
                    )
                else:
                    fast_state, _ = adapt_fast_state(
                        params, to_jax(conditioned_support), model_cfg, adapt_cfg
                    )
                offline_loss, offline_metrics = query_imitation_loss(
                    params, fast_state, to_jax(query), model_cfg, adapt_cfg
                )
                condition_rollouts = [
                    rollout(
                        params,
                        fast_state,
                        goal=goal,
                        episode_id=int(episode_id),
                        benchmark_cfg=benchmark_cfg,
                        normalizer=normalizer,
                        model_cfg=model_cfg,
                        read_enabled=bool(adapt_cfg.read_enabled),
                        read_mode=str(adapt_cfg.read_mode),
                        read_scale=float(adapt_cfg.read_scale),
                    )
                    for episode_id in query['episode_id']
                ]
                record = {
                    'task_id': int(local_task_id),
                    'task_latent': goal.tolist(),
                    'offline_loss': float(jax.device_get(offline_loss)),
                    'translation_loss': float(
                        jax.device_get(offline_metrics['translation_loss'])
                    ),
                    'gripper_loss': float(
                        jax.device_get(offline_metrics['gripper_loss'])
                    ),
                    'fast_delta_norm': float(
                        jax.device_get(
                            tree_difference_norm(
                                fast_state, initial_fast_state(params)
                            )
                        )
                    ),
                    'rollouts': condition_rollouts,
                }
                results[count_key][condition].append(record)
                task_result[condition] = record
            per_task[count_key][str(local_task_id)] = task_result
            logging.info(
                'support=%d task=%d/%d complete',
                support_count,
                local_task_id + 1,
                num_tasks,
            )

    aggregate: Dict[str, Any] = {}
    for support_index, (count_key, condition_records) in enumerate(results.items()):
        aggregate[count_key] = {}
        for condition, records in condition_records.items():
            rollouts = [rollout for record in records for rollout in record['rollouts']]
            successes = sum(int(rollout['success']) for rollout in rollouts)
            low, high = confidence_interval(successes, len(rollouts))
            aggregate[count_key][condition] = {
                'success_rate': successes / max(1, len(rollouts)),
                'success_count': successes,
                'rollout_count': len(rollouts),
                'success_ci95': [low, high],
                'mean_final_distance': float(
                    np.mean([rollout['final_distance'] for rollout in rollouts])
                ),
                'mean_offline_loss': float(
                    np.mean([record['offline_loss'] for record in records])
                ),
                'mean_fast_delta_norm': float(
                    np.mean([record['fast_delta_norm'] for record in records])
                ),
            }
        correct = aggregate[count_key].get('correct_support')
        no_update = aggregate[count_key].get('no_update')
        wrong = aggregate[count_key].get('wrong_task_support')
        if correct is not None and no_update is not None:
            aggregate[count_key]['correct_support_gain'] = {
                'success_rate': correct['success_rate'] - no_update['success_rate'],
                'offline_loss': (
                    no_update['mean_offline_loss'] - correct['mean_offline_loss']
                ),
            }
        if correct is not None and wrong is not None:
            aggregate[count_key]['correct_vs_wrong_gap'] = {
                'success_rate': correct['success_rate'] - wrong['success_rate'],
                'offline_loss': (
                    wrong['mean_offline_loss'] - correct['mean_offline_loss']
                ),
            }
        baseline_records = condition_records.get('no_update')
        if baseline_records is not None:
            aggregate[count_key]['paired_gain_over_no_update'] = {
                condition: _paired_gain_summary(
                    records,
                    baseline_records,
                    seed=int(cfg.seed) + support_index * 100 + condition_index,
                )
                for condition_index, (condition, records) in enumerate(
                    condition_records.items()
                )
                if condition != 'no_update'
            }

    checkpoint_name = Path(cfg.checkpoint_path).stem
    run_dir = (
        Path(cfg.output_dir).expanduser().resolve()
        / f'{checkpoint_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        'checkpoint_path': str(Path(cfg.checkpoint_path).expanduser().resolve()),
        'checkpoint_step': int(payload['step']),
        'split': split,
        'task_count': num_tasks,
        'normalizer_id': normalizer.identifier,
        'reset_policy': 'reset_to_w0_for_each_task_and_condition',
        'query_fast_state_policy': 'frozen_after_support',
        'paired_inference_unit': 'held_out_task_latent',
        'paired_ci_method': '10000-sample percentile bootstrap over task latents',
        'aggregate': aggregate,
        'per_task': per_task,
    }
    _write_json(run_dir / 'resolved_eval_config.json', config_to_dict(cfg))
    _write_json(run_dir / 'summary.json', summary)
    logging.info('State TTT evaluation complete: %s', run_dir)
    return run_dir


def main(argv):
    del argv
    evaluate_ttt_state(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
