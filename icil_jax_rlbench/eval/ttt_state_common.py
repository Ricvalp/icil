from __future__ import annotations

from typing import Any, Dict, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from icil_jax_rlbench.data.hidden_goal import HiddenGoalEnvironment, StateNormalizer
from icil_jax_rlbench.eval.support_controls import (
    SUPPORTED_CONDITIONS,
    bootstrap_mean_confidence_interval,
    condition_support,
    confidence_interval,
    random_fast_state_with_matched_delta,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    executable_action,
    predict_action,
)


def to_jax(value: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, value)


def remove_task_axis(section: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: np.asarray(value[0]) for name, value in section.items()}


def rollout(
    params: Mapping[str, Any],
    fast_state: Any,
    *,
    goal: np.ndarray,
    episode_id: int,
    benchmark_cfg,
    normalizer: StateNormalizer,
    model_cfg,
    read_enabled: bool,
    read_mode: str = 'absolute_gated',
    read_scale: float = 1.0,
) -> Dict[str, Any]:
    env = HiddenGoalEnvironment(benchmark_cfg, goal, int(episode_id))
    action_trace = []
    for _ in range(int(benchmark_cfg.horizon)):
        normalized_observation = normalizer.normalize_observation(env.observation())
        prediction = predict_action(
            params,
            fast_state,
            jnp.asarray(normalized_observation),
            model_cfg,
            read_enabled=bool(read_enabled),
            read_mode=str(read_mode),
            read_scale=float(read_scale),
        )
        action = np.asarray(jax.device_get(executable_action(prediction, model_cfg)))
        action_trace.append(action)
        _, done = env.step(action)
        if done:
            break
    return {
        'success': bool(env.success()),
        'final_distance': float(env.final_distance()),
        'rollout_length': len(action_trace),
    }
