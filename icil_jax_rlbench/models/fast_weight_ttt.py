from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp


PyTree = Any
READ_MODES = ('absolute_gated', 'delta')


@dataclass(frozen=True)
class FastWeightTTTConfig:
    observation_dim: int = 4
    action_dim: int = 3
    translation_dim: int = 2
    hidden_dim: int = 64
    fast_dim: int = 32
    fast_hidden_dim: int = 64
    fast_model: str = 'mlp'  # linear | mlp
    gate_init: float = 1e-3
    inner_lr_init: float = 3e-2
    inner_lr_min: float = 1e-5
    translation_output: str = 'tanh'  # tanh | linear
    translation_loss_weight: float = 1.0
    gripper_loss_weight: float = 0.25
    translation_huber_delta: float = 0.1
    gripper_loss: str = 'binary_cross_entropy'  # binary_cross_entropy | huber
    gripper_huber_delta: float = 0.1


@dataclass(frozen=True)
class TTTAdaptConfig:
    write_objective: str = 'kvb'  # kvb | action_bc
    write_segment_size: int = 4
    write_steps_per_segment: int = 1
    first_order: bool = False
    fast_grad_clip_norm: float = 1.0
    fast_update_clip_norm: float = 0.0
    fast_drift_weight: float = 0.0
    write_enabled: bool = True
    read_enabled: bool = True
    read_mode: str = 'absolute_gated'
    read_scale: float = 1.0


def _linear_init(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    *,
    scale: float = 1.0,
    zero_kernel: bool = False,
) -> Dict[str, jax.Array]:
    if zero_kernel:
        kernel = jnp.zeros((int(input_dim), int(output_dim)), dtype=jnp.float32)
    else:
        std = float(scale) * jnp.sqrt(2.0 / max(1, int(input_dim)))
        kernel = std * jax.random.normal(
            rng, (int(input_dim), int(output_dim)), dtype=jnp.float32
        )
    return {
        'kernel': kernel,
        'bias': jnp.zeros((int(output_dim),), dtype=jnp.float32),
    }


def _linear(params: Mapping[str, jax.Array], x: jax.Array) -> jax.Array:
    return jnp.matmul(x, params['kernel']) + params['bias']


def _layer_norm(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + float(eps))


def _l2_normalize(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    squared_norm = jnp.sum(
        jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True
    )
    inverse_norm = jax.lax.rsqrt(
        jnp.maximum(squared_norm, float(eps) * float(eps))
    )
    return x * inverse_norm.astype(x.dtype)


def _inverse_softplus(value: float) -> jax.Array:
    value = max(float(value), 1e-8)
    return jnp.asarray(jnp.log(jnp.expm1(value)), dtype=jnp.float32)


def init_fast_weight_ttt_params(
    rng: jax.Array,
    cfg: FastWeightTTTConfig,
) -> Dict[str, PyTree]:
    if str(cfg.fast_model) not in ('linear', 'mlp'):
        raise ValueError("fast_model must be 'linear' or 'mlp'.")
    if str(cfg.translation_output) not in ('tanh', 'linear'):
        raise ValueError("translation_output must be 'tanh' or 'linear'.")
    if str(cfg.gripper_loss) not in ('binary_cross_entropy', 'huber'):
        raise ValueError(
            "gripper_loss must be 'binary_cross_entropy' or 'huber'."
        )
    if int(cfg.action_dim) != int(cfg.translation_dim) + 1:
        raise ValueError('The state policy expects translation components plus one gripper logit.')
    keys = iter(jax.random.split(rng, 16))
    support_input_dim = (
        int(cfg.observation_dim) + int(cfg.action_dim) + int(cfg.observation_dim)
    )
    params: Dict[str, PyTree] = {
        'support_encoder': {
            'fc1': _linear_init(next(keys), support_input_dim, int(cfg.hidden_dim)),
            'fc2': _linear_init(next(keys), int(cfg.hidden_dim), int(cfg.hidden_dim)),
        },
        'query_encoder': {
            'fc1': _linear_init(next(keys), int(cfg.observation_dim), int(cfg.hidden_dim)),
            'fc2': _linear_init(next(keys), int(cfg.hidden_dim), int(cfg.hidden_dim)),
        },
        'key_projection': _linear_init(next(keys), int(cfg.hidden_dim), int(cfg.fast_dim)),
        'value_projection': _linear_init(next(keys), int(cfg.hidden_dim), int(cfg.fast_dim)),
        'query_projection': _linear_init(next(keys), int(cfg.hidden_dim), int(cfg.fast_dim)),
        'read_projection': _linear_init(next(keys), int(cfg.fast_dim), int(cfg.hidden_dim)),
        'translation_head': _linear_init(
            next(keys), int(cfg.hidden_dim), int(cfg.translation_dim), scale=0.1
        ),
        'gripper_head': _linear_init(next(keys), int(cfg.hidden_dim), 1, scale=0.1),
        'read_gate': jnp.full(
            (int(cfg.hidden_dim),), float(cfg.gate_init), dtype=jnp.float32
        ),
    }
    if str(cfg.fast_model) == 'linear':
        fast_init = {
            'linear': _linear_init(next(keys), int(cfg.fast_dim), int(cfg.fast_dim), scale=0.1)
        }
    else:
        fast_init = {
            'fc1': _linear_init(
                next(keys), int(cfg.fast_dim), int(cfg.fast_hidden_dim), scale=0.1
            ),
            'fc2': _linear_init(
                next(keys), int(cfg.fast_hidden_dim), int(cfg.fast_dim), scale=0.1
            ),
        }
    params['fast_init'] = fast_init
    raw_rate = _inverse_softplus(
        max(float(cfg.inner_lr_init) - float(cfg.inner_lr_min), 1e-8)
    )
    params['inner_lr_raw'] = jax.tree_util.tree_map(
        lambda _: raw_rate, fast_init
    )
    return params


def initial_fast_state(params: Mapping[str, PyTree]) -> PyTree:
    return jax.tree_util.tree_map(lambda x: x, params['fast_init'])


def inner_learning_rates(
    params: Mapping[str, PyTree], cfg: FastWeightTTTConfig
) -> PyTree:
    return jax.tree_util.tree_map(
        lambda raw: jax.nn.softplus(raw) + float(cfg.inner_lr_min),
        params['inner_lr_raw'],
    )


def fast_model_apply(
    fast_state: Mapping[str, PyTree], x: jax.Array, cfg: FastWeightTTTConfig
) -> jax.Array:
    if str(cfg.fast_model) == 'linear':
        return _linear(fast_state['linear'], x)
    hidden = jax.nn.gelu(_linear(fast_state['fc1'], x))
    return _linear(fast_state['fc2'], hidden)


def _encode_query(params: Mapping[str, PyTree], observation: jax.Array) -> jax.Array:
    hidden = jax.nn.gelu(_linear(params['query_encoder']['fc1'], observation))
    return jax.nn.gelu(_linear(params['query_encoder']['fc2'], hidden))


def encode_support_evidence(
    params: Mapping[str, PyTree],
    observation: jax.Array,
    action: jax.Array,
    next_observation: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    transition = next_observation - observation
    evidence = jnp.concatenate([observation, action, transition], axis=-1)
    hidden = jax.nn.gelu(_linear(params['support_encoder']['fc1'], evidence))
    hidden = jax.nn.gelu(_linear(params['support_encoder']['fc2'], hidden))
    return project_key_value(params, hidden)


def project_key_value(
    params: Mapping[str, PyTree], hidden: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Project an encoded support event into WRITE keys and values."""

    normalized = _layer_norm(hidden)
    key = _l2_normalize(_linear(params['key_projection'], normalized))
    value = _l2_normalize(_linear(params['value_projection'], normalized))
    return key, value


def fast_read_residual(
    params: Mapping[str, PyTree],
    fast_state: PyTree,
    hidden: jax.Array,
    cfg: FastWeightTTTConfig,
    *,
    read_enabled: bool = True,
    read_mode: str = 'absolute_gated',
    read_scale: float = 1.0,
) -> jax.Array:
    """Return the fast adapter residual injected into slow query features."""

    if not bool(read_enabled):
        return jnp.zeros_like(hidden)
    query = _l2_normalize(
        _linear(params['query_projection'], _layer_norm(hidden))
    )
    memory = fast_model_apply(fast_state, query, cfg)
    if str(read_mode) == 'delta':
        initial_memory = fast_model_apply(initial_fast_state(params), query, cfg)
        adapted_read = _linear(params['read_projection'], memory)
        initial_read = _linear(params['read_projection'], initial_memory)
        return float(read_scale) * (adapted_read - initial_read)
    elif str(read_mode) != 'absolute_gated':
        raise ValueError(f'read_mode must be one of {READ_MODES}.')
    read = _linear(params['read_projection'], memory)
    return jnp.tanh(params['read_gate']) * read


def read_fast_memory(
    params: Mapping[str, PyTree],
    fast_state: PyTree,
    hidden: jax.Array,
    cfg: FastWeightTTTConfig,
    *,
    read_enabled: bool = True,
    read_mode: str = 'absolute_gated',
    read_scale: float = 1.0,
) -> jax.Array:
    """READ from fast weights and inject a residual into slow features."""

    return hidden + fast_read_residual(
        params,
        fast_state,
        hidden,
        cfg,
        read_enabled=read_enabled,
        read_mode=read_mode,
        read_scale=read_scale,
    )


def predict_action(
    params: Mapping[str, PyTree],
    fast_state: PyTree,
    observation: jax.Array,
    cfg: FastWeightTTTConfig,
    *,
    read_enabled: bool = True,
    read_mode: str = 'absolute_gated',
    read_scale: float = 1.0,
) -> jax.Array:
    """Predict from query observation and fast state only.

    There is intentionally no support argument. Query actions are outputs and
    cannot enter either the query projection or the READ operation.
    """

    hidden = _encode_query(params, observation)
    hidden = read_fast_memory(
        params,
        fast_state,
        hidden,
        cfg,
        read_enabled=read_enabled,
        read_mode=read_mode,
        read_scale=read_scale,
    )
    translation = _linear(params['translation_head'], hidden)
    if str(cfg.translation_output) == 'tanh':
        translation = jnp.tanh(translation)
    gripper_logit = _linear(params['gripper_head'], hidden)
    return jnp.concatenate([translation, gripper_logit], axis=-1)


def executable_action(prediction: jax.Array, cfg: FastWeightTTTConfig) -> jax.Array:
    translation = jnp.clip(prediction[..., : int(cfg.translation_dim)], -1.0, 1.0)
    gripper_prediction = prediction[..., int(cfg.translation_dim) :]
    if str(cfg.gripper_loss) == 'binary_cross_entropy':
        gripper = jax.nn.sigmoid(gripper_prediction)
    else:
        gripper = jnp.clip(gripper_prediction, -1.0, 1.0)
    return jnp.concatenate([translation, gripper], axis=-1)


def _masked_mean(value: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(jnp.float32)
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    numerator = jnp.sum(value.astype(jnp.float32) * mask)
    denominator = jnp.sum(jnp.ones_like(value, dtype=jnp.float32) * mask)
    return numerator / jnp.maximum(denominator, 1.0)


def _huber_element(error: jax.Array, delta: float) -> jax.Array:
    absolute = jnp.abs(error)
    return jnp.where(
        absolute <= float(delta),
        0.5 * jnp.square(error) / max(float(delta), 1e-6),
        absolute - 0.5 * float(delta),
    )


def robotics_action_loss(
    prediction: jax.Array,
    target: jax.Array,
    mask: jax.Array,
    cfg: FastWeightTTTConfig,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    translation_prediction = prediction[..., : int(cfg.translation_dim)]
    translation_target = target[..., : int(cfg.translation_dim)]
    error = translation_prediction - translation_target
    translation_element = _huber_element(error, float(cfg.translation_huber_delta))
    translation_loss = _masked_mean(translation_element, mask)
    gripper_prediction = prediction[..., int(cfg.translation_dim)]
    gripper_target = target[..., int(cfg.translation_dim)]
    gripper_error = gripper_prediction - gripper_target
    if str(cfg.gripper_loss) == 'binary_cross_entropy':
        gripper_element = jnp.maximum(gripper_prediction, 0.0) - (
            gripper_prediction * gripper_target
        ) + jnp.log1p(jnp.exp(-jnp.abs(gripper_prediction)))
        gripper_accuracy = _masked_mean(
            (
                (jax.nn.sigmoid(gripper_prediction) >= 0.5)
                == (gripper_target >= 0.5)
            ).astype(jnp.float32),
            mask,
        )
    elif str(cfg.gripper_loss) == 'huber':
        gripper_element = _huber_element(
            gripper_error, float(cfg.gripper_huber_delta)
        )
        gripper_accuracy = _masked_mean(
            (jnp.abs(gripper_error) <= float(cfg.gripper_huber_delta)).astype(
                jnp.float32
            ),
            mask,
        )
    else:
        raise ValueError(
            "gripper_loss must be 'binary_cross_entropy' or 'huber'."
        )
    gripper_loss = _masked_mean(gripper_element, mask)
    total = (
        float(cfg.translation_loss_weight) * translation_loss
        + float(cfg.gripper_loss_weight) * gripper_loss
    )
    return total, {
        'translation_loss': translation_loss,
        'gripper_loss': gripper_loss,
        'translation_l1': _masked_mean(jnp.abs(error), mask),
        'gripper_l1': _masked_mean(jnp.abs(gripper_error), mask),
        'gripper_accuracy': gripper_accuracy,
    }


def tree_l2_norm(tree: PyTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves)
    )


def tree_difference_norm(left: PyTree, right: PyTree) -> jax.Array:
    return tree_l2_norm(
        jax.tree_util.tree_map(lambda x, y: x - y, left, right)
    )


def _clip_each_fast_tensor(tree: PyTree, max_norm: float) -> PyTree:
    if float(max_norm) <= 0.0:
        return tree

    def clip(value: jax.Array) -> jax.Array:
        squared_norm = jnp.sum(jnp.square(value.astype(jnp.float32)))
        norm = jnp.sqrt(jnp.maximum(squared_norm, 1e-16))
        scale = jnp.minimum(1.0, float(max_norm) / (norm + 1e-8))
        return value * scale.astype(value.dtype)

    return jax.tree_util.tree_map(clip, tree)


def _path_name(path) -> str:
    return '/'.join(str(getattr(entry, 'key', entry)) for entry in path)


def _per_tensor_norms(tree: PyTree, prefix: str) -> Dict[str, jax.Array]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {
        f'{prefix}/{_path_name(path)}': jnp.linalg.norm(value.astype(jnp.float32))
        for path, value in flat
    }


def write_loss(
    params: Mapping[str, PyTree],
    fast_state: PyTree,
    segment: Mapping[str, jax.Array],
    model_cfg: FastWeightTTTConfig,
    adapt_cfg: TTTAdaptConfig,
    fast_initial: PyTree,
) -> jax.Array:
    mask = segment['write_mask']
    objective = str(adapt_cfg.write_objective)
    if objective == 'kvb':
        key, value = encode_support_evidence(
            params,
            segment['observation'],
            segment['action'],
            segment['next_observation'],
        )
        reconstruction = fast_model_apply(fast_state, key, model_cfg)
        loss = _masked_mean(jnp.square(reconstruction - value), mask)
    elif objective == 'action_bc':
        prediction = predict_action(
            params,
            fast_state,
            segment['observation'],
            model_cfg,
            read_enabled=bool(adapt_cfg.read_enabled),
            read_mode=str(adapt_cfg.read_mode),
            read_scale=float(adapt_cfg.read_scale),
        )
        loss, _ = robotics_action_loss(
            prediction, segment['action'], mask, model_cfg
        )
    else:
        raise ValueError("write_objective must be 'kvb' or 'action_bc'.")
    if float(adapt_cfg.fast_drift_weight) > 0.0:
        loss = loss + float(adapt_cfg.fast_drift_weight) * jnp.square(
            tree_difference_norm(fast_state, fast_initial)
        )
    return loss


def _flatten_support(support: Mapping[str, jax.Array]) -> Dict[str, jax.Array]:
    required = ('observation', 'action', 'next_observation', 'write_mask')
    flattened = {}
    for name in required:
        value = support[name]
        flattened[name] = value.reshape((-1,) + value.shape[2:])
    return flattened


def segment_support(
    support: Mapping[str, jax.Array], segment_size: int
) -> Dict[str, jax.Array]:
    if int(segment_size) <= 0:
        raise ValueError('write_segment_size must be positive.')
    flat = _flatten_support(support)
    count = int(flat['write_mask'].shape[0])
    padded_count = ((count + int(segment_size) - 1) // int(segment_size)) * int(
        segment_size
    )
    pad = padded_count - count
    segmented = {}
    for name, value in flat.items():
        pad_width = [(0, pad)] + [(0, 0)] * (value.ndim - 1)
        padded = jnp.pad(value, pad_width)
        segmented[name] = padded.reshape(
            (padded_count // int(segment_size), int(segment_size)) + value.shape[1:]
        )
    return segmented


def adapt_fast_state(
    params: Mapping[str, PyTree],
    support: Mapping[str, jax.Array],
    model_cfg: FastWeightTTTConfig,
    adapt_cfg: TTTAdaptConfig,
) -> Tuple[PyTree, Dict[str, jax.Array]]:
    if int(adapt_cfg.write_steps_per_segment) <= 0:
        raise ValueError('write_steps_per_segment must be positive.')
    fast_initial = initial_fast_state(params)
    segments = segment_support(support, int(adapt_cfg.write_segment_size))
    segment_count = int(segments['write_mask'].shape[0])
    if not bool(adapt_cfg.write_enabled):
        zeros = jnp.zeros((segment_count,), dtype=jnp.float32)
        metrics = {
            'write_loss': zeros,
            'fast_grad_norm': zeros,
            'fast_update_norm': zeros,
            'fast_delta_norm': zeros,
        }
        for name in _per_tensor_norms(fast_initial, 'fast_tensor/grad_norm'):
            metrics[name] = zeros
        for name in _per_tensor_norms(fast_initial, 'fast_tensor/update_norm'):
            metrics[name] = zeros
        for name in _per_tensor_norms(fast_initial, 'fast_tensor/delta_norm'):
            metrics[name] = zeros
        return fast_initial, metrics
    learning_rates = inner_learning_rates(params, model_cfg)

    def segment_step(fast_state: PyTree, segment: Mapping[str, jax.Array]):
        def one_update(state: PyTree):
            loss, gradient = jax.value_and_grad(write_loss, argnums=1)(
                params,
                state,
                segment,
                model_cfg,
                adapt_cfg,
                fast_initial,
            )
            raw_gradient_norm = tree_l2_norm(gradient)
            gradient = _clip_each_fast_tensor(
                gradient, float(adapt_cfg.fast_grad_clip_norm)
            )
            if bool(adapt_cfg.first_order):
                gradient = jax.tree_util.tree_map(jax.lax.stop_gradient, gradient)
            update = jax.tree_util.tree_map(
                lambda rate, grad: -rate.astype(grad.dtype) * grad,
                learning_rates,
                gradient,
            )
            update = _clip_each_fast_tensor(
                update, float(adapt_cfg.fast_update_clip_norm)
            )
            next_state = jax.tree_util.tree_map(
                lambda value, delta: value + delta, state, update
            )
            metrics = {
                'write_loss': loss,
                'fast_grad_norm': raw_gradient_norm,
                'fast_update_norm': tree_l2_norm(update),
                **_per_tensor_norms(gradient, 'fast_tensor/grad_norm'),
                **_per_tensor_norms(update, 'fast_tensor/update_norm'),
            }
            delta = jax.tree_util.tree_map(
                lambda value, initial: value - initial, next_state, fast_initial
            )
            metrics.update(_per_tensor_norms(delta, 'fast_tensor/delta_norm'))
            return next_state, metrics

        state, accumulated = one_update(fast_state)
        for _ in range(1, int(adapt_cfg.write_steps_per_segment)):
            state, metrics = one_update(state)
            accumulated = jax.tree_util.tree_map(
                lambda left, right: left + right, accumulated, metrics
            )
        inverse_steps = jnp.asarray(
            1.0 / float(adapt_cfg.write_steps_per_segment), dtype=jnp.float32
        )
        accumulated = jax.tree_util.tree_map(
            lambda value: value * inverse_steps, accumulated
        )
        accumulated['fast_delta_norm'] = tree_difference_norm(state, fast_initial)
        return state, accumulated

    adapted, trace = jax.lax.scan(segment_step, fast_initial, segments)
    return adapted, trace


def adapt_encoded_support(
    params: Mapping[str, PyTree],
    support_registers: jax.Array,
    support_mask: jax.Array,
    model_cfg: FastWeightTTTConfig,
    adapt_cfg: TTTAdaptConfig,
) -> Tuple[PyTree, Dict[str, jax.Array]]:
    """Apply the same KVB WRITE rule to pre-encoded visual event registers.

    Inputs are [segments, registers, hidden_dim]. This is the gated bridge used
    by the RLBench space-time encoder; it deliberately has no query action or
    direct support-to-decoder output.
    """

    if str(adapt_cfg.write_objective) != 'kvb':
        raise ValueError('Encoded support currently implements only KVB WRITE.')
    if int(adapt_cfg.write_steps_per_segment) <= 0:
        raise ValueError('write_steps_per_segment must be positive.')
    if support_registers.ndim != 3:
        raise ValueError(
            'support_registers must be [segments, registers, hidden_dim], got '
            f'{support_registers.shape}.'
        )
    fast_initial = initial_fast_state(params)
    rates = inner_learning_rates(params, model_cfg)

    def segment_step(fast_state, segment):
        registers, mask = segment

        def loss_fn(state):
            key, value = project_key_value(params, registers)
            reconstruction = fast_model_apply(state, key, model_cfg)
            loss = _masked_mean(jnp.square(reconstruction - value), mask)
            if float(adapt_cfg.fast_drift_weight) > 0.0:
                loss = loss + float(adapt_cfg.fast_drift_weight) * jnp.square(
                    tree_difference_norm(state, fast_initial)
                )
            return loss

        state = fast_state
        accumulated = None
        for _ in range(int(adapt_cfg.write_steps_per_segment)):
            loss, gradient = jax.value_and_grad(loss_fn)(state)
            raw_gradient_norm = tree_l2_norm(gradient)
            gradient = _clip_each_fast_tensor(
                gradient, float(adapt_cfg.fast_grad_clip_norm)
            )
            if bool(adapt_cfg.first_order):
                gradient = jax.tree_util.tree_map(jax.lax.stop_gradient, gradient)
            update = jax.tree_util.tree_map(
                lambda rate, grad: -rate.astype(grad.dtype) * grad,
                rates,
                gradient,
            )
            update = _clip_each_fast_tensor(
                update, float(adapt_cfg.fast_update_clip_norm)
            )
            state = jax.tree_util.tree_map(
                lambda value, delta: value + delta, state, update
            )
            metrics = {
                'write_loss': loss,
                'fast_grad_norm': raw_gradient_norm,
                'fast_update_norm': tree_l2_norm(update),
            }
            accumulated = (
                metrics
                if accumulated is None
                else jax.tree_util.tree_map(
                    lambda left, right: left + right, accumulated, metrics
                )
            )
        inverse = 1.0 / max(1, int(adapt_cfg.write_steps_per_segment))
        accumulated = jax.tree_util.tree_map(lambda value: value * inverse, accumulated)
        accumulated['fast_delta_norm'] = tree_difference_norm(state, fast_initial)
        return state, accumulated

    if not bool(adapt_cfg.write_enabled):
        count = int(support_registers.shape[0])
        zeros = jnp.zeros((count,), dtype=jnp.float32)
        return fast_initial, {
            'write_loss': zeros,
            'fast_grad_norm': zeros,
            'fast_update_norm': zeros,
            'fast_delta_norm': zeros,
        }
    return jax.lax.scan(
        segment_step, fast_initial, (support_registers, support_mask)
    )


def query_imitation_loss(
    params: Mapping[str, PyTree],
    fast_state: PyTree,
    query: Mapping[str, jax.Array],
    model_cfg: FastWeightTTTConfig,
    adapt_cfg: TTTAdaptConfig,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    prediction = predict_action(
        params,
        fast_state,
        query['observation'],
        model_cfg,
        read_enabled=bool(adapt_cfg.read_enabled),
        read_mode=str(adapt_cfg.read_mode),
        read_scale=float(adapt_cfg.read_scale),
    )
    return robotics_action_loss(
        prediction,
        query['action'],
        query['outer_loss_mask'],
        model_cfg,
    )


def query_adaptation_effect_metrics(
    params: Mapping[str, PyTree],
    initial_state: PyTree,
    adapted_state: PyTree,
    query: Mapping[str, jax.Array],
    model_cfg: FastWeightTTTConfig,
    adapt_cfg: TTTAdaptConfig,
) -> Dict[str, jax.Array]:
    """Measure how much a fast update changes READ features and predictions."""

    hidden = _encode_query(params, query['observation'])
    read_kwargs = {
        'read_enabled': bool(adapt_cfg.read_enabled),
        'read_mode': str(adapt_cfg.read_mode),
        'read_scale': float(adapt_cfg.read_scale),
    }
    initial_read = fast_read_residual(
        params, initial_state, hidden, model_cfg, **read_kwargs
    )
    adapted_read = fast_read_residual(
        params, adapted_state, hidden, model_cfg, **read_kwargs
    )
    initial_prediction = predict_action(
        params, initial_state, query['observation'], model_cfg, **read_kwargs
    )
    adapted_prediction = predict_action(
        params, adapted_state, query['observation'], model_cfg, **read_kwargs
    )
    mask = query['outer_loss_mask']
    return {
        'read_delta_rms': jnp.sqrt(
            _masked_mean(jnp.square(adapted_read - initial_read), mask)
        ),
        'prediction_delta_rms': jnp.sqrt(
            _masked_mean(
                jnp.square(adapted_prediction - initial_prediction), mask
            )
        ),
    }


def fast_state_effective_rank(fast_state: PyTree) -> jax.Array:
    matrices = [leaf for leaf in jax.tree_util.tree_leaves(fast_state) if leaf.ndim == 2]
    if not matrices:
        return jnp.asarray(0.0, dtype=jnp.float32)
    ranks = []
    for matrix in matrices:
        singular_values = jnp.linalg.svd(matrix.astype(jnp.float32), compute_uv=False)
        probability = singular_values / jnp.maximum(jnp.sum(singular_values), 1e-8)
        entropy = -jnp.sum(
            jnp.where(probability > 0.0, probability * jnp.log(probability + 1e-8), 0.0)
        )
        ranks.append(jnp.exp(entropy))
    return jnp.mean(jnp.stack(ranks))


def named_fast_tensor_metrics(
    params: Mapping[str, PyTree],
    fast_state: PyTree,
    cfg: FastWeightTTTConfig,
    prefix: str = 'fast',
) -> Dict[str, jax.Array]:
    initial = initial_fast_state(params)
    rates = inner_learning_rates(params, cfg)
    flat_state, _ = jax.tree_util.tree_flatten_with_path(fast_state)
    flat_initial, _ = jax.tree_util.tree_flatten_with_path(initial)
    flat_rates, _ = jax.tree_util.tree_flatten_with_path(rates)
    out: Dict[str, jax.Array] = {}
    for (path, value), (_, initial_value), (_, rate) in zip(
        flat_state, flat_initial, flat_rates
    ):
        name = _path_name(path)
        out[f'{prefix}/{name}_norm'] = jnp.linalg.norm(value.astype(jnp.float32))
        out[f'{prefix}/{name}_delta_norm'] = jnp.linalg.norm(
            (value - initial_value).astype(jnp.float32)
        )
        out[f'{prefix}/{name}_lr'] = rate
    return out
