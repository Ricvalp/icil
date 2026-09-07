from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from icil_jax_rlbench.data.hidden_goal import HiddenGoalEnvironment, StateNormalizer
from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    executable_action,
    predict_action,
)


@dataclass(frozen=True)
class StateRolloutTrace:
    episode_id: int
    observations: np.ndarray
    actions: np.ndarray
    success: bool
    final_distance: float

    @property
    def positions(self) -> np.ndarray:
        return self.observations[:, :2]


@dataclass(frozen=True)
class StateTaskVisualization:
    task_id: int
    wrong_task_id: int
    goal: np.ndarray
    wrong_goal: np.ndarray
    conditions: tuple[str, ...]
    correct_support_positions: tuple[np.ndarray, ...]
    wrong_support_positions: tuple[np.ndarray, ...]
    rollouts: Mapping[str, tuple[StateRolloutTrace, ...]]
    reference_actions: Mapping[str, np.ndarray]
    write_traces: Mapping[str, Mapping[str, np.ndarray]]
    fast_tensor_delta_norms: Mapping[str, Mapping[str, float]]
    grid_x: np.ndarray
    grid_y: np.ndarray
    vector_fields: Mapping[str, tuple[np.ndarray, np.ndarray]]
    world_limit: float
    success_radius: float


def policy_actions(
    params: Mapping[str, Any],
    fast_state: Any,
    normalized_observations: np.ndarray,
    model_cfg: FastWeightTTTConfig,
    *,
    read_enabled: bool,
    read_mode: str = 'absolute_gated',
    read_scale: float = 1.0,
) -> np.ndarray:
    prediction = predict_action(
        params,
        fast_state,
        jnp.asarray(normalized_observations),
        model_cfg,
        read_enabled=bool(read_enabled),
        read_mode=str(read_mode),
        read_scale=float(read_scale),
    )
    actions = executable_action(prediction, model_cfg)
    return np.asarray(jax.device_get(actions), dtype=np.float32)


def capture_rollout(
    params: Mapping[str, Any],
    fast_state: Any,
    *,
    goal: np.ndarray,
    episode_id: int,
    benchmark_cfg: Any,
    normalizer: StateNormalizer,
    model_cfg: FastWeightTTTConfig,
    read_enabled: bool,
    read_mode: str = 'absolute_gated',
    read_scale: float = 1.0,
) -> StateRolloutTrace:
    env = HiddenGoalEnvironment(benchmark_cfg, goal, int(episode_id))
    observations = [env.observation()]
    actions = []
    for _ in range(int(benchmark_cfg.horizon)):
        normalized = normalizer.normalize_observation(env.observation())
        action = policy_actions(
            params,
            fast_state,
            normalized,
            model_cfg,
            read_enabled=read_enabled,
            read_mode=read_mode,
            read_scale=read_scale,
        )
        next_observation, done = env.step(action)
        actions.append(action)
        observations.append(next_observation)
        if done:
            break
    return StateRolloutTrace(
        episode_id=int(episode_id),
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        success=bool(env.success()),
        final_distance=float(env.final_distance()),
    )


def support_position_traces(
    support: Mapping[str, np.ndarray], normalizer: StateNormalizer
) -> tuple[np.ndarray, ...]:
    traces = []
    for episode_index in range(int(support['observation'].shape[0])):
        valid = np.asarray(support['write_mask'][episode_index], dtype=np.bool_)
        count = int(np.sum(valid))
        if count < 1:
            continue
        observations = normalizer.denormalize_observation(
            support['observation'][episode_index, :count]
        )
        final_observation = normalizer.denormalize_observation(
            support['next_observation'][episode_index, count - 1]
        )
        positions = np.concatenate(
            [observations[:, :2], final_observation[None, :2]], axis=0
        )
        traces.append(positions.astype(np.float32))
    return tuple(traces)


def planar_vector_field(
    params: Mapping[str, Any],
    fast_state: Any,
    *,
    normalizer: StateNormalizer,
    model_cfg: FastWeightTTTConfig,
    world_limit: float,
    grid_size: int,
    phase: float,
    read_enabled: bool,
    read_mode: str = 'absolute_gated',
    read_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.linspace(
        -float(world_limit), float(world_limit), int(grid_size), dtype=np.float32
    )
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    observations = np.stack(
        [
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.zeros(grid_x.size, dtype=np.float32),
            np.full(grid_x.size, float(phase), dtype=np.float32),
        ],
        axis=-1,
    )
    normalized = normalizer.normalize_observation(observations)
    actions = policy_actions(
        params,
        fast_state,
        normalized,
        model_cfg,
        read_enabled=read_enabled,
        read_mode=read_mode,
        read_scale=read_scale,
    )
    field_x = actions[:, 0].reshape(grid_x.shape)
    field_y = actions[:, 1].reshape(grid_y.shape)
    return grid_x, grid_y, field_x, field_y


def fast_tensor_delta_norms(
    initial_fast_state: Any, fast_state: Any
) -> dict[str, float]:
    delta = jax.tree_util.tree_map(
        lambda current, initial: current - initial, fast_state, initial_fast_state
    )
    flat, _ = jax.tree_util.tree_flatten_with_path(delta)
    values = {}
    for path, tensor in flat:
        name = '/'.join(str(getattr(entry, 'key', entry)) for entry in path)
        values[name] = float(
            np.linalg.norm(np.asarray(jax.device_get(tensor), dtype=np.float32))
        )
    return values


__all__ = [
    'StateRolloutTrace',
    'StateTaskVisualization',
    'capture_rollout',
    'fast_tensor_delta_norms',
    'planar_vector_field',
    'policy_actions',
    'support_position_traces',
]
