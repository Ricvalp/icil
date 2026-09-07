from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from phi_mujoco.evaluation import PolicyInput
from phi_mujoco.offline import StandardNormalization

from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    initial_fast_state,
    predict_action,
)


@partial(
    jax.jit,
    static_argnames=('model_cfg', 'read_enabled', 'read_mode', 'read_scale'),
)
def _predict_normalized_action(
    params: Mapping[str, Any],
    fast_state: Any,
    observation: jax.Array,
    *,
    model_cfg: FastWeightTTTConfig,
    read_enabled: bool,
    read_mode: str,
    read_scale: float,
) -> jax.Array:
    return predict_action(
        params,
        fast_state,
        observation,
        model_cfg,
        read_enabled=bool(read_enabled),
        read_mode=str(read_mode),
        read_scale=float(read_scale),
    )


class MetaWorldJaxPolicy:
    """Expose a frozen JAX query policy to the public phi policy interface."""

    def __init__(
        self,
        *,
        integration,
        params: Mapping[str, Any],
        model_cfg: FastWeightTTTConfig,
        normalization: StandardNormalization,
        fast_state: Any | None = None,
        read_enabled: bool = False,
        read_mode: str = 'absolute_gated',
        read_scale: float = 1.0,
    ) -> None:
        self.integration = integration
        self.params = jax.tree_util.tree_map(jnp.asarray, params)
        self.model_cfg = model_cfg
        self.normalization = normalization
        self.fast_state = jax.tree_util.tree_map(
            jnp.asarray,
            initial_fast_state(self.params) if fast_state is None else fast_state,
        )
        self.read_enabled = bool(read_enabled)
        self.read_mode = str(read_mode)
        self.read_scale = float(read_scale)
        self._validate_model_input_dimension()

    @property
    def raw_observation_dim(self) -> int:
        return len(self.normalization.obs_mean)

    def _validate_model_input_dimension(self) -> None:
        if self.model_cfg.observation_dim != self.raw_observation_dim:
            raise ValueError(
                'Unconditioned policy model input must match the raw observation '
                'dimension.'
            )

    def _model_observation(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.normalization.normalize_observations(state), dtype=np.float32
        )

    def reset(self, *, integration, seed: int) -> None:
        del seed
        if integration != self.integration.spec:
            raise ValueError('Evaluation runner integration changed during a rollout.')

    def predict(self, inputs: PolicyInput) -> np.ndarray:
        if inputs.integration != self.integration.spec:
            raise ValueError('Policy input does not match the bound MetaWorld task.')
        if set(inputs.observations) != {'state'}:
            raise ValueError(
                "MetaWorld policy expects exactly the 'state' observation."
            )
        state = np.asarray(inputs.observations['state'], dtype=np.float32)
        if state.shape != (self.raw_observation_dim,):
            raise ValueError(
                f'Expected state shape {(self.raw_observation_dim,)}, got '
                f'{state.shape}.'
            )
        model_observation = self._model_observation(state)
        normalized_action = _predict_normalized_action(
            self.params,
            self.fast_state,
            jnp.asarray(model_observation),
            model_cfg=self.model_cfg,
            read_enabled=self.read_enabled,
            read_mode=self.read_mode,
            read_scale=self.read_scale,
        )
        action = self.normalization.denormalize_actions(
            np.asarray(jax.device_get(normalized_action))
        )
        return self.integration.project_action(
            np.asarray(action, dtype=np.float32),
            observations=inputs.observations,
        )


class MetaWorldConditionedJaxPolicy(MetaWorldJaxPolicy):
    """A query policy with one explicit context fixed for the whole rollout."""

    def __init__(self, *, context: np.ndarray, **kwargs) -> None:
        value = np.asarray(context, dtype=np.float32)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError('Policy context must be a finite one-dimensional array.')
        self.context = value
        super().__init__(**kwargs)

    def _validate_model_input_dimension(self) -> None:
        expected = self.raw_observation_dim + self.context.size
        if self.model_cfg.observation_dim != expected:
            raise ValueError(
                f'Conditioned model expects input dimension {expected}, got '
                f'{self.model_cfg.observation_dim}.'
            )

    def _model_observation(self, state: np.ndarray) -> np.ndarray:
        normalized_state = super()._model_observation(state)
        return np.concatenate((normalized_state, self.context), axis=0)


# Compatibility for existing Reach callers and checkpoints.
ML1ReachJaxPolicy = MetaWorldJaxPolicy

__all__ = [
    'ML1ReachJaxPolicy',
    'MetaWorldConditionedJaxPolicy',
    'MetaWorldJaxPolicy',
]
