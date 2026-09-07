from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import optax

from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    initial_fast_state,
    predict_action,
    robotics_action_loss,
    tree_l2_norm,
)
from icil_jax_rlbench.train.ttt_step import TTTTrainState


def query_only_objective(
    params,
    query: Mapping[str, jax.Array],
    model_cfg: FastWeightTTTConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Behavior cloning with no support input and no fast-memory READ."""

    prediction = predict_action(
        params,
        initial_fast_state(params),
        query['observation'],
        model_cfg,
        read_enabled=False,
    )
    loss, metrics = robotics_action_loss(
        prediction,
        query['action'],
        query['outer_loss_mask'],
        model_cfg,
    )
    return loss, {'loss': loss, **metrics}


def _clip_by_global_norm(tree, max_norm: float):
    norm = tree_l2_norm(tree)
    if float(max_norm) <= 0.0:
        return tree, norm
    scale = jnp.minimum(1.0, float(max_norm) / (norm + 1e-8))
    return jax.tree_util.tree_map(
        lambda value: value * scale.astype(value.dtype), tree
    ), norm


def create_query_only_train_step(
    optimizer: optax.GradientTransformation,
    model_cfg: FastWeightTTTConfig,
    *,
    slow_grad_clip_norm: float,
) -> Callable[
    [TTTTrainState, Mapping[str, jax.Array]],
    tuple[TTTTrainState, dict[str, jax.Array]],
]:
    def train_step(state: TTTTrainState, query: Mapping[str, jax.Array]):
        next_rng, _ = jax.random.split(state.rng)

        def loss_fn(params):
            return query_only_objective(params, query, model_cfg)

        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        gradients, raw_gradient_norm = _clip_by_global_norm(
            gradients, float(slow_grad_clip_norm)
        )
        updates, next_opt_state = optimizer.update(
            gradients, state.opt_state, state.params
        )
        next_state = state.replace(
            step=state.step + jnp.asarray(1, dtype=state.step.dtype),
            params=optax.apply_updates(state.params, updates),
            opt_state=next_opt_state,
            rng=next_rng,
        )
        return next_state, {
            **metrics,
            'slow_grad_norm': raw_gradient_norm,
            'slow_update_norm': tree_l2_norm(updates),
        }

    return jax.jit(train_step)


__all__ = ['create_query_only_train_step', 'query_only_objective']
