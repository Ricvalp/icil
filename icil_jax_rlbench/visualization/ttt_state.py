from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from icil_jax_rlbench.data.hidden_goal import (
    HiddenGoalMetaSampler,
    HiddenGoalTaskBank,
    StateNormalizer,
)
from icil_jax_rlbench.eval.ttt_state_common import (
    SUPPORTED_CONDITIONS,
    condition_support,
    random_fast_state_with_matched_delta,
    remove_task_axis,
    to_jax,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    adapt_fast_state,
    initial_fast_state,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.provenance import config_to_dict
from icil_jax_rlbench.train.ttt_runner import (
    adaptation_config_from,
    fast_weight_config_from,
    hidden_goal_config_from,
    validate_adaptation_only_config,
)
from icil_jax_rlbench.visualization.state_plots import (
    plot_evaluation_summary,
    render_task_artifacts,
)
from icil_jax_rlbench.visualization.state_rollouts import (
    StateTaskVisualization,
    capture_rollout,
    fast_tensor_delta_norms,
    planar_vector_field,
    policy_actions,
    support_position_traces,
)


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _host_trace(trace: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(jax.device_get(value), dtype=np.float32)
        for name, value in trace.items()
    }


def _validate_config(cfg: ConfigDict, task_count: int) -> tuple[str, ...]:
    if not str(cfg.checkpoint_path):
        raise ValueError('config.checkpoint_path is required.')
    conditions = tuple(str(value) for value in cfg.conditions)
    if not conditions:
        raise ValueError('config.conditions must not be empty.')
    unknown = sorted(set(conditions) - set(SUPPORTED_CONDITIONS))
    if unknown:
        raise ValueError(f'Unknown visualization conditions: {unknown}')
    required = {'no_update', 'correct_support'}
    missing = sorted(required - set(conditions))
    if missing:
        raise ValueError(
            'Trajectory visualization requires matched baselines: ' + ', '.join(missing)
        )
    task_ids = tuple(int(value) for value in cfg.task_ids)
    if not task_ids:
        raise ValueError('config.task_ids must contain at least one task.')
    invalid = [task_id for task_id in task_ids if not 0 <= task_id < task_count]
    if invalid:
        raise ValueError(
            f'config.task_ids contains IDs outside [0, {task_count}): {invalid}'
        )
    if int(cfg.support_count) < 1 or int(cfg.query_episodes) < 1:
        raise ValueError('support_count and query_episodes must be positive.')
    if int(cfg.vector_field_grid_size) < 2:
        raise ValueError('vector_field_grid_size must be at least 2.')
    if int(cfg.figure_dpi) < 1 or float(cfg.video_fps) <= 0.0:
        raise ValueError('figure_dpi and video_fps must be positive.')
    return conditions


def _adapt_conditions(
    *,
    conditions: tuple[str, ...],
    params: Mapping[str, Any],
    correct_support: Mapping[str, np.ndarray],
    wrong_support: Mapping[str, np.ndarray],
    model_cfg: Any,
    adapt_cfg: Any,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Mapping[str, np.ndarray]]]:
    initial = initial_fast_state(params)
    correct_adapted, correct_trace = adapt_fast_state(
        params,
        to_jax(correct_support),
        model_cfg,
        adapt_cfg,
    )
    states: dict[str, Any] = {}
    traces: dict[str, Mapping[str, np.ndarray]] = {}
    for condition in conditions:
        if condition == 'no_update':
            states[condition] = initial
            traces[condition] = {}
        elif condition == 'correct_support':
            states[condition] = correct_adapted
            traces[condition] = _host_trace(correct_trace)
        elif condition == 'random_update_matched_norm':
            states[condition] = random_fast_state_with_matched_delta(
                params,
                correct_adapted,
                rng,
            )
            traces[condition] = {}
        else:
            support = condition_support(
                condition,
                correct_support,
                wrong_support,
                rng,
            )
            state, trace = adapt_fast_state(
                params,
                to_jax(support),
                model_cfg,
                adapt_cfg,
            )
            states[condition] = state
            traces[condition] = _host_trace(trace)
    return states, traces


def visualize_ttt_state(cfg: ConfigDict) -> Path:
    checkpoint_path = Path(cfg.checkpoint_path).expanduser().resolve()
    payload = load_checkpoint(checkpoint_path)
    checkpoint_type = payload.get('extra', {}).get('checkpoint_type')
    if checkpoint_type != 'fast_weight_ttt_state':
        raise ValueError(
            'Expected a fast_weight_ttt_state checkpoint, got '
            f'{checkpoint_type!r}.'
        )
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
    task_count = task_bank.num_tasks(split)
    conditions = _validate_config(cfg, task_count)
    task_ids = tuple(int(value) for value in cfg.task_ids)

    sampler = HiddenGoalMetaSampler(
        benchmark_cfg,
        task_bank,
        normalizer,
        split=split,
        seed=int(cfg.seed) + 18_001,
    )
    rng = np.random.default_rng(int(cfg.seed) + 19_001)
    run_name = (
        f'{checkpoint_path.stem}_{split}_'
        f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    run_dir = Path(cfg.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    task_manifests = []
    for task_index, task_id in enumerate(task_ids):
        wrong_task_id = (task_id + 1) % task_count
        correct_batch = sampler.build_batch(
            1,
            support_episodes=int(cfg.support_count),
            query_episodes=int(cfg.query_episodes),
            task_ids=[task_id],
        )
        wrong_batch = sampler.build_batch(
            1,
            support_episodes=int(cfg.support_count),
            query_episodes=1,
            task_ids=[wrong_task_id],
        )
        correct_support = remove_task_axis(correct_batch['support'])
        wrong_support = remove_task_axis(wrong_batch['support'])
        query = remove_task_axis(correct_batch['query'])
        states, write_traces = _adapt_conditions(
            conditions=conditions,
            params=params,
            correct_support=correct_support,
            wrong_support=wrong_support,
            model_cfg=model_cfg,
            adapt_cfg=adapt_cfg,
            rng=rng,
        )

        goal = task_bank.goal(split, task_id)
        wrong_goal = task_bank.goal(split, wrong_task_id)
        rollouts = {
            condition: tuple(
                capture_rollout(
                    params,
                    states[condition],
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
            )
            for condition in conditions
        }
        reference_actions = {
            condition: policy_actions(
                params,
                states[condition],
                query['observation'],
                model_cfg,
                read_enabled=bool(adapt_cfg.read_enabled),
                read_mode=str(adapt_cfg.read_mode),
                read_scale=float(adapt_cfg.read_scale),
            )
            for condition in conditions
        }

        vector_fields = {}
        grid_x = grid_y = None
        for condition in conditions:
            field_grid_x, field_grid_y, field_x, field_y = planar_vector_field(
                params,
                states[condition],
                normalizer=normalizer,
                model_cfg=model_cfg,
                world_limit=float(benchmark_cfg.world_limit),
                grid_size=int(cfg.vector_field_grid_size),
                phase=float(cfg.vector_field_phase),
                read_enabled=bool(adapt_cfg.read_enabled),
                read_mode=str(adapt_cfg.read_mode),
                read_scale=float(adapt_cfg.read_scale),
            )
            if grid_x is None:
                grid_x, grid_y = field_grid_x, field_grid_y
            vector_fields[condition] = (field_x, field_y)
        assert grid_x is not None and grid_y is not None

        initial = initial_fast_state(params)
        data = StateTaskVisualization(
            task_id=task_id,
            wrong_task_id=wrong_task_id,
            goal=goal,
            wrong_goal=wrong_goal,
            conditions=conditions,
            correct_support_positions=support_position_traces(
                correct_support, normalizer
            ),
            wrong_support_positions=support_position_traces(
                wrong_support, normalizer
            ),
            rollouts=rollouts,
            reference_actions=reference_actions,
            write_traces=write_traces,
            fast_tensor_delta_norms={
                condition: fast_tensor_delta_norms(initial, states[condition])
                for condition in conditions
            },
            grid_x=grid_x,
            grid_y=grid_y,
            vector_fields=vector_fields,
            world_limit=float(benchmark_cfg.world_limit),
            success_radius=float(benchmark_cfg.success_radius),
        )
        task_dir = run_dir / f'task-{task_id:03d}'
        task_manifests.append(
            render_task_artifacts(
                data,
                task_dir,
                dpi=int(cfg.figure_dpi),
                write_video=bool(cfg.write_video),
                video_fps=float(cfg.video_fps),
            )
        )
        logging.info(
            'Gate 3 visualization task %d/%d complete: %s',
            task_index + 1,
            len(task_ids),
            task_dir,
        )

    aggregate_artifact = None
    if str(cfg.evaluation_summary_path):
        summary_path = Path(cfg.evaluation_summary_path).expanduser().resolve()
        with summary_path.open('r', encoding='utf-8') as handle:
            evaluation_summary = json.load(handle)
        aggregate_artifact = 'evaluation_summary.png'
        plot_evaluation_summary(
            evaluation_summary,
            run_dir / aggregate_artifact,
            dpi=int(cfg.figure_dpi),
        )

    manifest = {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_step': int(payload['step']),
        'split': split,
        'seed': int(cfg.seed),
        'normalizer_id': normalizer.identifier,
        'support_count': int(cfg.support_count),
        'query_episodes': int(cfg.query_episodes),
        'matched_query_episode_ids_across_conditions': True,
        'reset_policy': 'reset_to_w0_for_each_task_and_condition',
        'privileged_goal_usage': 'visualization_only',
        'aggregate_artifact': aggregate_artifact,
        'tasks': task_manifests,
    }
    _write_json(run_dir / 'resolved_visualization_config.json', config_to_dict(cfg))
    _write_json(run_dir / 'manifest.json', manifest)
    logging.info('Gate 3 visualization complete: %s', run_dir)
    return run_dir


__all__ = ['visualize_ttt_state']
