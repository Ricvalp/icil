from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from flax import struct
import jax
import jax.numpy as jnp
import optax

from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    TTTAdaptConfig,
    adapt_fast_state,
    initial_fast_state,
    inner_learning_rates,
    query_adaptation_effect_metrics,
    query_imitation_loss,
    tree_difference_norm,
    tree_l2_norm,
    write_loss,
    segment_support,
)


PyTree = Any


@dataclass(frozen=True)
class TTTStepConfig:
    slow_grad_clip_norm: float = 1.0
    outer_fast_drift_weight: float = 0.0


@struct.dataclass
class TTTTrainState:
    step: jax.Array
    params: PyTree
    opt_state: PyTree
    rng: jax.Array


def create_ttt_train_state(
    params: PyTree,
    optimizer: optax.GradientTransformation,
    rng: jax.Array,
    *,
    step: int = 0,
    opt_state: PyTree | None = None,
) -> TTTTrainState:
    return TTTTrainState(
        step=jnp.asarray(int(step), dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params) if opt_state is None else opt_state,
        rng=rng,
    )


def _clip_tree_by_global_norm(tree: PyTree, max_norm: float) -> Tuple[PyTree, jax.Array]:
    norm = tree_l2_norm(tree)
    if float(max_norm) <= 0.0:
        return tree, norm
    scale = jnp.minimum(1.0, float(max_norm) / (norm + 1e-8))
    return jax.tree_util.tree_map(lambda value: value * scale.astype(value.dtype), tree), norm


def _mean_tree(tree: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda value: jnp.mean(value, axis=0), tree)


def _task_meta_objective(
    params: Mapping[str, PyTree],
    support: Mapping[str, jax.Array],
    query: Mapping[str, jax.Array],
    model_cfg: FastWeightTTTConfig,
    adapt_cfg: TTTAdaptConfig,
    step_cfg: TTTStepConfig,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    fast_initial = initial_fast_state(params)
    before, before_metrics = query_imitation_loss(
        params, fast_initial, query, model_cfg, adapt_cfg
    )
    adapted, write_trace = adapt_fast_state(params, support, model_cfg, adapt_cfg)
    after, after_metrics = query_imitation_loss(
        params, adapted, query, model_cfg, adapt_cfg
    )
    fast_delta = tree_difference_norm(adapted, fast_initial)
    effect_metrics = query_adaptation_effect_metrics(
        params,
        fast_initial,
        adapted,
        query,
        model_cfg,
        adapt_cfg,
    )
    objective = after
    if float(step_cfg.outer_fast_drift_weight) > 0.0:
        objective = objective + float(step_cfg.outer_fast_drift_weight) * jnp.square(
            fast_delta
        )
    return objective, {
        'loss': objective,
        'query_loss_before': before,
        'query_loss_after': after,
        'improvement': before - after,
        'improvement_ratio': (before - after) / (jnp.abs(before) + 1e-8),
        'write_loss': jnp.mean(write_trace['write_loss']),
        'first_write_loss': write_trace['write_loss'][0],
        'last_write_loss': write_trace['write_loss'][-1],
        'fast_grad_norm': jnp.mean(write_trace['fast_grad_norm']),
        'fast_update_norm': jnp.mean(write_trace['fast_update_norm']),
        'fast_delta_norm': fast_delta,
        'fast_relative_delta_norm': fast_delta / (tree_l2_norm(fast_initial) + 1e-8),
        **effect_metrics,
        'translation_loss_before': before_metrics['translation_loss'],
        'translation_loss_after': after_metrics['translation_loss'],
        'gripper_loss_before': before_metrics['gripper_loss'],
        'gripper_loss_after': after_metrics['gripper_loss'],
        'translation_l1_after': after_metrics['translation_l1'],
        'gripper_accuracy_after': after_metrics['gripper_accuracy'],
        **{
            name: jnp.mean(value)
            for name, value in write_trace.items()
            if name.startswith('fast_tensor/')
        },
    }


def ttt_meta_objective(
    params: Mapping[str, PyTree],
    batch: Mapping[str, Mapping[str, jax.Array]],
    model_cfg: FastWeightTTTConfig,
    adapt_cfg: TTTAdaptConfig,
    step_cfg: TTTStepConfig,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    def one_task(support, query):
        return _task_meta_objective(
            params, support, query, model_cfg, adapt_cfg, step_cfg
        )

    losses, task_metrics = jax.vmap(one_task)(batch['support'], batch['query'])
    metrics = _mean_tree(task_metrics)
    loss = jnp.mean(losses)
    metrics['loss'] = loss
    rates = jax.tree_util.tree_leaves(inner_learning_rates(params, model_cfg))
    metrics['inner_lr_mean'] = jnp.mean(jnp.stack(rates))
    metrics['inner_lr_min'] = jnp.min(jnp.stack(rates))
    metrics['inner_lr_max'] = jnp.max(jnp.stack(rates))
    gate = jnp.tanh(params['read_gate'])
    metrics['read_gate_abs_mean'] = jnp.mean(jnp.abs(gate))
    metrics['read_gate_abs_max'] = jnp.max(jnp.abs(gate))
    metrics['fast_initial_norm'] = tree_l2_norm(initial_fast_state(params))
    return loss, metrics


def _gradient_group_metrics(grads: Mapping[str, PyTree]) -> Dict[str, jax.Array]:
    groups = (
        'support_encoder',
        'key_projection',
        'value_projection',
        'query_projection',
        'fast_init',
        'inner_lr_raw',
        'read_projection',
        'read_gate',
        'translation_head',
        'gripper_head',
    )
    return {
        f'meta_grad/{name}': tree_l2_norm(grads[name])
        for name in groups
        if name in grads
    }


def create_ttt_train_step(
    optimizer: optax.GradientTransformation,
    model_cfg: FastWeightTTTConfig,
    adapt_cfg: TTTAdaptConfig,
    step_cfg: TTTStepConfig,
    *,
    distributed: bool = False,
    jit: bool = True,
) -> Callable[[TTTTrainState, Mapping[str, PyTree]], Tuple[TTTTrainState, Dict[str, jax.Array]]]:
    axis_name = 'devices'

    def train_step(
        state: TTTTrainState, batch: Mapping[str, PyTree]
    ) -> Tuple[TTTTrainState, Dict[str, jax.Array]]:
        next_rng, _ = jax.random.split(state.rng)
        def loss_fn(params):
            return ttt_meta_objective(
                params, batch, model_cfg, adapt_cfg, step_cfg
            )

        (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        del loss
        gradient_metrics = _gradient_group_metrics(gradients)
        gradients, raw_gradient_norm = _clip_tree_by_global_norm(
            gradients, float(step_cfg.slow_grad_clip_norm)
        )
        if distributed:
            gradients = jax.lax.pmean(gradients, axis_name=axis_name)
            metrics = jax.lax.pmean(metrics, axis_name=axis_name)
            gradient_metrics = jax.lax.pmean(gradient_metrics, axis_name=axis_name)
            raw_gradient_norm = jax.lax.pmean(raw_gradient_norm, axis_name=axis_name)
        updates, next_opt_state = optimizer.update(
            gradients, state.opt_state, state.params
        )
        next_params = optax.apply_updates(state.params, updates)
        next_state = state.replace(
            step=state.step + jnp.asarray(1, dtype=state.step.dtype),
            params=next_params,
            opt_state=next_opt_state,
            rng=next_rng,
        )
        return next_state, {
            **metrics,
            **gradient_metrics,
            'slow_grad_norm': raw_gradient_norm,
            'slow_update_norm': tree_l2_norm(updates),
        }

    if distributed:
        if not bool(jit):
            raise ValueError('Distributed TTT training requires jit=True.')
        return jax.pmap(train_step, axis_name=axis_name)
    return jax.jit(train_step) if bool(jit) else train_step


def write_query_gradient_alignment(
    params: Mapping[str, PyTree],
    support: Mapping[str, jax.Array],
    query: Mapping[str, jax.Array],
    model_cfg: FastWeightTTTConfig,
    adapt_cfg: TTTAdaptConfig,
) -> Dict[str, jax.Array]:
    fast_initial = initial_fast_state(params)
    segments = segment_support(support, int(adapt_cfg.write_segment_size))
    first_segment = jax.tree_util.tree_map(lambda value: value[0], segments)
    write_gradient = jax.grad(write_loss, argnums=1)(
        params,
        fast_initial,
        first_segment,
        model_cfg,
        adapt_cfg,
        fast_initial,
    )

    def query_loss_for_fast(fast_state: PyTree) -> jax.Array:
        loss, _ = query_imitation_loss(
            params, fast_state, query, model_cfg, adapt_cfg
        )
        return loss

    query_gradient = jax.grad(query_loss_for_fast)(fast_initial)
    write_leaves = jax.tree_util.tree_leaves(write_gradient)
    query_leaves = jax.tree_util.tree_leaves(query_gradient)
    dot = sum(
        jnp.vdot(write.astype(jnp.float32), query.astype(jnp.float32))
        for write, query in zip(write_leaves, query_leaves)
    )
    write_norm = tree_l2_norm(write_gradient)
    query_norm = tree_l2_norm(query_gradient)
    # The actual WRITE update is -grad(write), hence the sign inversion.
    update_query_cosine = -dot / (write_norm * query_norm + 1e-8)
    return {
        'write_query_gradient_cosine': dot / (write_norm * query_norm + 1e-8),
        'update_query_gradient_cosine': update_query_cosine,
        'oracle_query_grad_norm': query_norm,
        'diagnostic_write_grad_norm': write_norm,
    }
