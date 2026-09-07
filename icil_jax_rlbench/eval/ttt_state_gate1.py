from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

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
    condition_support,
    remove_task_axis,
    rollout,
    to_jax,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    initial_fast_state,
    predict_action,
    robotics_action_loss,
    tree_l2_norm,
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
    'config', None, 'Ordinary adaptation upper-bound diagnostic.', lock_config=False
)


def _parameter_mask(params: Mapping[str, Any], subset: str) -> Mapping[str, Any]:
    if subset == 'action_heads':
        included = {'translation_head', 'gripper_head'}
    elif subset == 'query_policy':
        included = {
            'query_encoder',
            'query_projection',
            'read_projection',
            'read_gate',
            'translation_head',
            'gripper_head',
        }
    elif subset == 'all':
        included = set(params)
    else:
        raise ValueError("adapt_subset must be 'action_heads', 'query_policy', or 'all'.")
    return {
        name: jax.tree_util.tree_map(lambda _: name in included, value)
        for name, value in params.items()
    }


def _ordinary_adapt(
    params: Mapping[str, Any],
    support: Mapping[str, jax.Array],
    *,
    model_cfg,
    read_enabled: bool,
    mask: Mapping[str, Any],
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
            read_enabled=bool(read_enabled),
        )
        loss, _ = robotics_action_loss(
            prediction, support['action'], support['write_mask'], model_cfg
        )
        return loss

    def body(value, _):
        loss, gradient = jax.value_and_grad(support_loss)(value)
        gradient = jax.tree_util.tree_map(
            lambda grad, selected: grad if selected else jnp.zeros_like(grad),
            gradient,
            mask,
        )
        norm = tree_l2_norm(gradient)
        scale = jnp.minimum(1.0, float(clip_norm) / (norm + 1e-8))
        next_value = jax.tree_util.tree_map(
            lambda current, grad: current - float(learning_rate) * scale * grad,
            value,
            gradient,
        )
        return next_value, {'support_loss': loss, 'gradient_norm': norm}

    return jax.lax.scan(body, params, xs=None, length=int(steps))


def _query_loss(params, query, model_cfg, read_enabled: bool):
    prediction = predict_action(
        params,
        initial_fast_state(params),
        query['observation'],
        model_cfg,
        read_enabled=bool(read_enabled),
    )
    loss, metrics = robotics_action_loss(
        prediction, query['action'], query['outer_loss_mask'], model_cfg
    )
    return loss, metrics


def evaluate_gate1(cfg: ConfigDict) -> Path:
    payload = load_checkpoint(str(cfg.checkpoint_path))
    train_cfg = ConfigDict(payload['config'])
    validate_adaptation_only_config(train_cfg)
    benchmark_cfg = hidden_goal_config_from(train_cfg)
    model_cfg = fast_weight_config_from(train_cfg, benchmark_cfg)
    adapt_cfg = adaptation_config_from(train_cfg)
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])
    normalizer = StateNormalizer.from_dict(payload['extra']['normalizer'])
    task_bank = HiddenGoalTaskBank(benchmark_cfg)
    sampler = HiddenGoalMetaSampler(
        benchmark_cfg,
        task_bank,
        normalizer,
        split=str(cfg.split),
        seed=int(cfg.seed) + 100,
    )
    num_tasks = task_bank.num_tasks(str(cfg.split))
    if int(cfg.max_tasks) > 0:
        num_tasks = min(num_tasks, int(cfg.max_tasks))
    parameter_mask = _parameter_mask(params, str(cfg.adapt_subset))
    adapt = jax.jit(
        lambda value, support: _ordinary_adapt(
            value,
            support,
            model_cfg=model_cfg,
            read_enabled=bool(adapt_cfg.read_enabled),
            mask=parameter_mask,
            steps=int(cfg.inner_steps),
            learning_rate=float(cfg.inner_lr),
            clip_norm=float(cfg.inner_grad_clip_norm),
        )
    )
    rng = np.random.default_rng(int(cfg.seed) + 200)
    conditions = tuple(str(value) for value in cfg.conditions)
    records: Dict[str, list[Dict[str, Any]]] = {condition: [] for condition in conditions}
    for task_id in range(num_tasks):
        correct = sampler.build_batch(
            1,
            support_episodes=int(cfg.support_episodes),
            query_episodes=int(cfg.query_episodes_per_task),
            task_ids=[task_id],
        )
        wrong = sampler.build_batch(
            1,
            support_episodes=int(cfg.support_episodes),
            query_episodes=1,
            task_ids=[(task_id + 1) % task_bank.num_tasks(str(cfg.split))],
        )
        correct_support = remove_task_axis(correct['support'])
        wrong_support = remove_task_axis(wrong['support'])
        query = to_jax(remove_task_axis(correct['query']))
        goal = task_bank.goal(str(cfg.split), task_id)
        for condition in conditions:
            if condition == 'no_update':
                adapted_params = params
                trace = {
                    'support_loss': jnp.zeros((0,), dtype=jnp.float32),
                    'gradient_norm': jnp.zeros((0,), dtype=jnp.float32),
                }
            else:
                support = condition_support(
                    condition, correct_support, wrong_support, rng
                )
                adapted_params, trace = adapt(params, to_jax(support))
            loss, metrics = _query_loss(
                adapted_params, query, model_cfg, bool(adapt_cfg.read_enabled)
            )
            rollouts = [
                rollout(
                    adapted_params,
                    initial_fast_state(adapted_params),
                    goal=goal,
                    episode_id=int(episode_id),
                    benchmark_cfg=benchmark_cfg,
                    normalizer=normalizer,
                    model_cfg=model_cfg,
                    read_enabled=bool(adapt_cfg.read_enabled),
                )
                for episode_id in np.asarray(query['episode_id'])
            ]
            records[condition].append(
                {
                    'task_id': task_id,
                    'query_loss': float(loss),
                    'translation_loss': float(metrics['translation_loss']),
                    'gripper_loss': float(metrics['gripper_loss']),
                    'support_loss_first': (
                        None if trace['support_loss'].size == 0 else float(trace['support_loss'][0])
                    ),
                    'support_loss_last': (
                        None if trace['support_loss'].size == 0 else float(trace['support_loss'][-1])
                    ),
                    'rollouts': rollouts,
                }
            )
        logging.info('Gate 1 task %d/%d complete', task_id + 1, num_tasks)

    aggregate = {}
    for condition, condition_records in records.items():
        rollouts = [item for record in condition_records for item in record['rollouts']]
        aggregate[condition] = {
            'query_loss': float(np.mean([record['query_loss'] for record in condition_records])),
            'success_rate': float(np.mean([rollout['success'] for rollout in rollouts])),
            'final_distance': float(
                np.mean([rollout['final_distance'] for rollout in rollouts])
            ),
        }
    if 'correct_support' in aggregate and 'no_update' in aggregate:
        aggregate['correct_support_gain'] = {
            'query_loss': aggregate['no_update']['query_loss']
            - aggregate['correct_support']['query_loss'],
            'success_rate': aggregate['correct_support']['success_rate']
            - aggregate['no_update']['success_rate'],
        }
    output_dir = (
        Path(cfg.output_dir).expanduser().resolve()
        / f'{Path(cfg.checkpoint_path).stem}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        'gate': 'ordinary_adaptation_upper_bound',
        'checkpoint': str(Path(cfg.checkpoint_path).resolve()),
        'adapt_subset': str(cfg.adapt_subset),
        'inner_steps': int(cfg.inner_steps),
        'inner_lr': float(cfg.inner_lr),
        'aggregate': aggregate,
        'per_task': records,
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    with (output_dir / 'resolved_eval_config.json').open('w', encoding='utf-8') as handle:
        json.dump(config_to_dict(cfg), handle, indent=2, sort_keys=True)
        handle.write('\n')
    return output_dir


def main(argv):
    del argv
    evaluate_gate1(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
