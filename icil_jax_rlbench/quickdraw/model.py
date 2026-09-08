"""Causal sketch policies using the shared functional fast-weight adapter.

A uses an autoregressive diagonal-Gaussian mixture coordinate likelihood, plus
Bernoulli incoming-pen and STOP events. B uses the same GRU backbone with a
deterministic coordinate/action head. WRITE evidence never enters TTT READ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    TTTAdaptConfig,
    _linear,
    _linear_init,
    adapt_encoded_support,
    apply_fast_gradient,
    fast_drift_penalty,
    fast_model_apply,
    fast_read_residual,
    init_fast_adapter_params,
    initial_fast_state,
    inner_learning_rates,
    project_key_value,
    tree_difference_norm,
    tree_l2_norm,
)


MODEL_TYPES = (
    'ttt_kvb_full', 'explicit_context', 'no_support',
    'ttt_supervised_write', 'ttt_kvb_first_order',
)
ArrayTree = Any


@dataclass(frozen=True)
class SketchModelConfig:
    experiment: str = 'a'
    model_type: str = 'ttt_kvb_full'
    hidden_dim: int = 48
    fast_dim: int = 16
    fast_hidden_dim: int = 24
    max_steps: int = 128
    mixture_components: int = 5
    segment_size: int = 16
    inner_lr_init: float = 0.03
    inner_lr_min: float = 1e-5
    write_steps_per_segment: int = 1
    fast_grad_clip_norm: float = 1.0
    fast_update_clip_norm: float = 0.0
    fast_drift_weight: float = 0.0
    read_scale: float = 1.0
    motion_bound: float = 0.1
    pen_loss_weight: float = 0.25
    stop_loss_weight: float = 0.25

    def __post_init__(self):
        if self.experiment not in ('a', 'b1', 'b2'):
            raise ValueError('experiment must be a, b1 or b2.')
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f'model_type must be one of {MODEL_TYPES}.')
        for name in ('hidden_dim', 'fast_dim', 'fast_hidden_dim', 'max_steps',
                     'mixture_components', 'segment_size', 'write_steps_per_segment'):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f'{name} must be positive.')
        if self.motion_bound <= 0 or self.inner_lr_init <= 0:
            raise ValueError('motion_bound and inner_lr_init must be positive.')

    def fast_config(self) -> FastWeightTTTConfig:
        return FastWeightTTTConfig(
            hidden_dim=self.hidden_dim, fast_dim=self.fast_dim,
            fast_hidden_dim=self.fast_hidden_dim, inner_lr_init=self.inner_lr_init,
            inner_lr_min=self.inner_lr_min,
        )

    def adapt_config(self) -> TTTAdaptConfig:
        return TTTAdaptConfig(
            write_segment_size=self.segment_size,
            write_steps_per_segment=self.write_steps_per_segment,
            first_order=self.model_type == 'ttt_kvb_first_order',
            fast_grad_clip_norm=self.fast_grad_clip_norm,
            fast_update_clip_norm=self.fast_update_clip_norm,
            fast_drift_weight=self.fast_drift_weight,
            read_mode='delta', read_scale=self.read_scale,
        )


def _gru_init(key, input_dim, hidden_dim):
    left, right = jax.random.split(key)
    return {
        'input': _linear_init(left, input_dim, 3 * hidden_dim, scale=0.4),
        'hidden': _linear_init(right, hidden_dim, 3 * hidden_dim, scale=0.4),
    }


def _gru_step(params, carry, inputs):
    xi, xr, xn = jnp.split(_linear(params['input'], inputs), 3, axis=-1)
    hi, hr, hn = jnp.split(_linear(params['hidden'], carry), 3, axis=-1)
    update = jax.nn.sigmoid(xi + hi)
    reset = jax.nn.sigmoid(xr + hr)
    candidate = jnp.tanh(xn + reset * hn)
    return (1.0 - update) * carry + update * candidate


def init_sketch_params(key: jax.Array, cfg: SketchModelConfig) -> ArrayTree:
    query_key, output_key, support_key, adapter_key, context_key = jax.random.split(key, 5)
    output_dim = 5 * cfg.mixture_components + 2 if cfg.experiment == 'a' else 4
    params = {
        'query_encoder': _gru_init(query_key, 13, cfg.hidden_dim),
        'output_head': _linear_init(output_key, cfg.hidden_dim, output_dim, scale=0.1),
    }
    if cfg.model_type.startswith('ttt_'):
        params['support_encoder'] = _gru_init(support_key, 16, cfg.hidden_dim)
        params.update(init_fast_adapter_params(adapter_key, cfg.fast_config()))
    else:
        # Comparators have no trainable fast adapter; this keeps reset uniform.
        params['fast_init'] = {}
        if cfg.model_type == 'explicit_context':
            params['context_encoder'] = _gru_init(support_key, 16, cfg.hidden_dim)
            params['context_projection'] = _linear_init(
                context_key, cfg.hidden_dim, cfg.hidden_dim, scale=0.4
            )
    return params


def _section_shape(section: Mapping[str, jax.Array]):
    if section['tokens'].ndim != 3 or section['tokens'].shape[-1] != 4:
        raise ValueError('A single-task section needs tokens [demo,time,4].')
    demos, steps = section['tokens'].shape[:2]
    if section['event_mask'].shape != (demos, steps):
        raise ValueError('event_mask must have shape [demo,time].')
    if section['point_mask'].shape != (demos, steps):
        raise ValueError('point_mask must have shape [demo,time].')
    if section['frame'].shape != (demos, 4):
        raise ValueError('frame must have shape [demo,4].')
    if section['state'].shape != (demos, steps, 3):
        raise ValueError('state must have shape [demo,time,3].')
    return demos, steps


def _support_registers(params, support, cfg, encoder_name):
    demos, steps = _section_shape(support)
    mask = support['event_mask'].astype(bool)
    points = support['point_mask'].astype(bool)
    # Public demonstrated waypoint boundaries distinguish program knots from
    # B2 travel/interpolation. Query READ never consumes this support channel.
    knots = support.get('knot_mask', points)
    if knots.shape != points.shape:
        raise ValueError('Support knot_mask must have shape [demo,time].')
    knots = (knots.astype(bool) & points & mask)[..., None]
    tokens = jnp.where(mask[..., None], support['tokens'], 0.0)
    state = jnp.where(mask[..., None], support['state'], 0.0)
    positions = state[..., :2] if cfg.experiment == 'b2' else tokens[..., :2]
    previous = jnp.concatenate([jnp.zeros_like(positions[:, :1]), positions[:, :-1]], axis=1)
    previous_valid = jnp.concatenate([jnp.zeros_like(points[:, :1]), points[:, :-1]], axis=1)
    transitions = jnp.where((points & previous_valid)[..., None], positions - previous, 0.0)
    elapsed = jnp.broadcast_to(jnp.arange(steps)[None, :, None] / float(cfg.max_steps), (demos, steps, 1))
    start = jnp.broadcast_to((jnp.arange(steps) == 0)[None, :, None], (demos, steps, 1))
    frames = jnp.broadcast_to(support['frame'][:, None, :], (demos, steps, 4))
    evidence = jnp.concatenate([tokens, transitions, frames, state, elapsed, start, knots], axis=-1)
    evidence = jnp.where(mask[..., None], evidence, 0.0)

    def encode_drawing(events, valid):
        def step(carry, item):
            event, keep = item
            proposed = _gru_step(params[encoder_name], carry, event)
            carry = jnp.where(keep, proposed, carry)
            return carry, jnp.where(keep, carry, 0.0)
        return jax.lax.scan(step, jnp.zeros((cfg.hidden_dim,), dtype=events.dtype), (events, valid))[1]

    # Reset recurrence at each drawing; masked padding cannot create transitions.
    return jax.vmap(encode_drawing)(evidence, mask), mask


def encode_context(params, support, cfg: SketchModelConfig):
    """The separately named feed-forward support path for its own comparator."""
    if cfg.model_type != 'explicit_context':
        raise ValueError('Direct support context is exclusive to explicit_context.')
    registers, mask = _support_registers(params, support, cfg, 'context_encoder')
    return {'registers': registers.reshape((-1, cfg.hidden_dim)), 'mask': mask.reshape(-1)}


def _check_context(cfg, context):
    if cfg.model_type != 'explicit_context' and context is not None:
        raise ValueError('Only explicit_context may receive support context.')


def _read_hidden(params, fast_state, hidden, cfg, context):
    _check_context(cfg, context)
    if cfg.model_type.startswith('ttt_'):
        return hidden + fast_read_residual(
            params, fast_state, hidden, cfg.fast_config(),
            read_mode='delta', read_scale=cfg.read_scale,
        )
    if cfg.model_type == 'explicit_context' and context is not None:
        scores = jnp.einsum('...h,nh->...n', hidden, context['registers']) / jnp.sqrt(float(cfg.hidden_dim))
        weights = jax.nn.softmax(jnp.where(context['mask'], scores, -1e9), axis=-1)
        weights = weights * context['mask']
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-8)
        attended = jnp.einsum('...n,nh->...h', weights, context['registers'])
        residual = _linear(params['context_projection'], attended)
        return hidden + jnp.where(jnp.any(context['mask']), residual, 0.0)
    return hidden


def _distribution(params, fast_state, hidden, cfg, context=None):
    raw = _linear(params['output_head'], _read_hidden(params, fast_state, hidden, cfg, context))
    if cfg.experiment == 'a':
        count = cfg.mixture_components
        return {
            'mixture_logits': raw[..., :count],
            'xy_mean': raw[..., count:3 * count].reshape(raw.shape[:-1] + (count, 2)),
            'xy_log_scale': -1.5 + 2.5 * jnp.tanh(raw[..., 3 * count:5 * count].reshape(raw.shape[:-1] + (count, 2))),
            'pen_logit': raw[..., -2], 'stop_logit': raw[..., -1],
        }
    xy = raw[..., :2]
    if cfg.experiment == 'b2':
        xy = cfg.motion_bound * xy * jax.lax.rsqrt(1.0 + jnp.sum(xy ** 2, axis=-1, keepdims=True))
    return {'xy': xy, 'pen_logit': raw[..., 2], 'stop_logit': raw[..., 3]}


def initial_query_carry(cfg: SketchModelConfig) -> jax.Array:
    return jnp.zeros((cfg.hidden_dim,), dtype=jnp.float32)


def _query_input(previous_token, timestep, frame, state, cfg):
    # Time is a fixed public clock, never divided by a target's true length.
    clock = jnp.asarray([timestep / float(cfg.max_steps), timestep == 0], dtype=previous_token.dtype)
    return jnp.concatenate([previous_token, frame, state, clock], axis=-1)


def query_step(params, fast_state, cfg: SketchModelConfig, carry, previous_token,
               timestep, frame, state, context=None):
    """One causal READ; B2 supplies the actual current environment state."""
    hidden = _gru_step(params['query_encoder'], carry,
                       _query_input(previous_token, timestep, frame, state, cfg))
    return hidden, _distribution(params, fast_state, hidden, cfg, context)


def _query_hidden(params, query, cfg):
    demos, steps = _section_shape(query)
    tokens = query['tokens']
    history = jnp.concatenate([jnp.zeros_like(tokens[:, :1]), tokens[:, :-1]], axis=1)

    def drawing(previous, frame, states):
        def step(carry, item):
            token, state, timestep = item
            inputs = _query_input(token, timestep, frame, state, cfg)
            hidden = _gru_step(params['query_encoder'], carry, inputs)
            return hidden, hidden
        return jax.lax.scan(step, jnp.zeros((cfg.hidden_dim,), dtype=tokens.dtype),
                            (previous, states, jnp.arange(steps)))[1]
    # Teacher forcing exposes only the previous event; masks never enter READ.
    return jax.vmap(drawing)(history, query['frame'], query['state'])


def teacher_forced_distribution(params, fast_state, query, cfg, context=None):
    """Training/diagnostic API; generation uses query_step without target data."""
    return _distribution(params, fast_state, _query_hidden(params, query, cfg), cfg, context)


def _masked_mean(value, mask):
    return jnp.sum(jnp.where(mask, value, 0.0)) / jnp.maximum(jnp.sum(mask), 1)


def _prediction_loss(distribution, tokens, event_mask, point_mask, cfg):
    points = point_mask.astype(bool)
    events = event_mask.astype(bool)
    targets = jnp.where(events[..., None], tokens, 0.0)
    if cfg.experiment == 'a':
        residual = (targets[..., None, :2] - distribution['xy_mean']) * jnp.exp(-distribution['xy_log_scale'])
        log_component = -0.5 * jnp.sum(residual ** 2 + 2.0 * distribution['xy_log_scale'] + jnp.log(2.0 * jnp.pi), axis=-1)
        coordinate_element = -jax.scipy.special.logsumexp(jax.nn.log_softmax(distribution['mixture_logits'], axis=-1) + log_component, axis=-1)
        mean = jnp.sum(jax.nn.softmax(distribution['mixture_logits'], axis=-1)[..., None] * distribution['xy_mean'], axis=-2)
    else:
        mean = distribution['xy']
        coordinate_element = jnp.mean((mean - targets[..., :2]) ** 2, axis=-1)
    coordinate_loss = _masked_mean(coordinate_element, points)
    pen_loss = _masked_mean(jax.nn.softplus(distribution['pen_logit']) - targets[..., 2] * distribution['pen_logit'], points)
    stop_loss = _masked_mean(jax.nn.softplus(distribution['stop_logit']) - targets[..., 3] * distribution['stop_logit'], events)
    loss = coordinate_loss + cfg.pen_loss_weight * pen_loss + cfg.stop_loss_weight * stop_loss
    return loss, {
        'loss': loss, 'coordinate_loss': coordinate_loss,
        'pen_loss': pen_loss, 'stop_loss': stop_loss,
        'coordinate_mse': _masked_mean(jnp.mean((mean - targets[..., :2]) ** 2, axis=-1), points),
        'pen_accuracy': _masked_mean((distribution['pen_logit'] >= 0) == (targets[..., 2] >= 0.5), points),
        'stop_accuracy': _masked_mean((distribution['stop_logit'] >= 0) == (targets[..., 3] >= 0.5), events),
        'query_events': jnp.sum(events), 'query_points': jnp.sum(points),
    }


def query_loss(params, fast_state, query, cfg: SketchModelConfig, context=None):
    distribution = teacher_forced_distribution(params, fast_state, query, cfg, context)
    return _prediction_loss(distribution, query['tokens'], query['event_mask'], query['point_mask'], cfg)


def _segment_drawings(value, segment_size):
    demos, steps = value.shape[:2]
    per_drawing = (steps + segment_size - 1) // segment_size
    padding = per_drawing * segment_size - steps
    padded = jnp.pad(value, [(0, 0), (0, padding)] + [(0, 0)] * (value.ndim - 2))
    # A partial last segment is closed before the next demonstration starts.
    return padded.reshape((demos * per_drawing, segment_size) + value.shape[2:])


def _empty_trace():
    return {name: jnp.zeros((1,), dtype=jnp.float32) for name in
            ('write_loss', 'fast_grad_norm', 'fast_update_norm', 'fast_delta_norm')}


def adapt_sketch(params, support, cfg: SketchModelConfig, mode=None):
    if mode is not None and mode != cfg.model_type:
        raise ValueError('Adaptation mode must match the checkpoint model_type.')
    if cfg.model_type in ('no_support', 'explicit_context'):
        return initial_fast_state(params), _empty_trace()
    if cfg.model_type != 'ttt_supervised_write':
        registers, mask = _support_registers(params, support, cfg, 'support_encoder')
        return adapt_encoded_support(
            params, _segment_drawings(registers, cfg.segment_size),
            _segment_drawings(mask, cfg.segment_size), cfg.fast_config(), cfg.adapt_config(),
        )
    hidden = _query_hidden(params, support, cfg)
    segments = tuple(_segment_drawings(value, cfg.segment_size) for value in
                     (hidden, support['tokens'], support['event_mask'], support['point_mask']))
    initial = initial_fast_state(params)
    rates = inner_learning_rates(params, cfg.fast_config())

    def segment_step(state, item):
        features, tokens, events, points = item
        def loss_fn(fast):
            loss, _ = _prediction_loss(_distribution(params, fast, features, cfg), tokens, events, points, cfg)
            loss = loss + cfg.fast_drift_weight * fast_drift_penalty(fast, initial)
            return jnp.where(jnp.any(events), loss, jnp.zeros_like(loss))
        metrics = None
        for _ in range(cfg.write_steps_per_segment):
            loss, gradient = jax.value_and_grad(loss_fn)(state)
            raw_norm = tree_l2_norm(gradient)
            state, _, update = apply_fast_gradient(state, gradient, rates, cfg.adapt_config())
            values = {'write_loss': loss, 'fast_grad_norm': raw_norm, 'fast_update_norm': tree_l2_norm(update)}
            metrics = values if metrics is None else jax.tree_util.tree_map(lambda x, y: x + y, metrics, values)
        metrics = jax.tree_util.tree_map(lambda x: x / cfg.write_steps_per_segment, metrics)
        metrics['fast_delta_norm'] = tree_difference_norm(state, initial)
        return state, metrics
    return jax.lax.scan(segment_step, initial, segments)


def meta_objective(params, batch, cfg: SketchModelConfig, mode=None, key=None):
    """Task-isolated query-only objective; analytic AR NLL uses no noise key."""
    del key
    def task(support, query):
        fast, trace = adapt_sketch(params, support, cfg, mode)
        context = encode_context(params, support, cfg) if cfg.model_type == 'explicit_context' else None
        loss, metrics = query_loss(params, fast, query, cfg, context)
        metrics.update({name: jnp.mean(value) for name, value in trace.items()})
        metrics['fast_delta_norm'] = tree_difference_norm(fast, initial_fast_state(params))
        metrics['support_events'] = jnp.sum(support['event_mask'])
        segment_mask = _segment_drawings(support['event_mask'], cfg.segment_size)
        metrics['write_steps'] = jnp.sum(jnp.any(segment_mask, axis=-1)) * cfg.write_steps_per_segment if cfg.model_type.startswith('ttt_') else jnp.asarray(0)
        return loss, metrics
    losses, metrics = jax.vmap(task)(batch['support'], batch['query'])
    return jnp.mean(losses), jax.tree_util.tree_map(lambda x: jnp.mean(x.astype(jnp.float32)), metrics)


def first_write_diagnostics(params, support, query, cfg: SketchModelConfig):
    """Offline oracle diagnostics; query targets never enter adapt_sketch.

    Functional vectors use a fixed teacher-forced query history supplied by the
    caller. Compare tasks with the same public probes for response geometry.
    Positive local_improvement_prediction means the first actual update aligns
    with descent on the query objective; raw gradient cosines are separate.
    """
    if not cfg.model_type.startswith('ttt_'):
        raise ValueError('WRITE diagnostics require a TTT model.')
    initial = initial_fast_state(params)
    masks = _segment_drawings(support['event_mask'], cfg.segment_size)
    index = jnp.argmax(jnp.any(masks, axis=-1))
    mask = masks[index]
    if cfg.model_type == 'ttt_supervised_write':
        features = _segment_drawings(_query_hidden(params, support, cfg), cfg.segment_size)[index]
        tokens = _segment_drawings(support['tokens'], cfg.segment_size)[index]
        points = _segment_drawings(support['point_mask'], cfg.segment_size)[index]
        def write_fn(state):
            loss, _ = _prediction_loss(_distribution(params, state, features, cfg), tokens, mask, points, cfg)
            loss = loss + cfg.fast_drift_weight * fast_drift_penalty(state, initial)
            return jnp.where(jnp.any(mask), loss, jnp.zeros_like(loss))
    else:
        registers, _ = _support_registers(params, support, cfg, 'support_encoder')
        registers = _segment_drawings(registers, cfg.segment_size)[index]
        key, value = project_key_value(params, registers)
        def write_fn(state):
            reconstruction = fast_model_apply(state, key, cfg.fast_config())
            loss = _masked_mean(jnp.mean((reconstruction - value) ** 2, axis=-1), mask)
            loss = loss + cfg.fast_drift_weight * fast_drift_penalty(state, initial)
            return jnp.where(jnp.any(mask), loss, jnp.zeros_like(loss))
    write_loss, raw_gradient = jax.value_and_grad(write_fn)(initial)
    first, clipped_gradient, update = apply_fast_gradient(
        initial, raw_gradient, inner_learning_rates(params, cfg.fast_config()), cfg.adapt_config()
    )
    final, _ = adapt_sketch(params, support, cfg)
    before, query_gradient = jax.value_and_grad(lambda state: query_loss(params, state, query, cfg)[0])(initial)
    after_first = query_loss(params, first, query, cfg)[0]
    after_final = query_loss(params, final, query, cfg)[0]
    def dot(left, right):
        return sum(jnp.sum(x * y) for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)))
    def cosine(left, right):
        return dot(left, right) / jnp.maximum(tree_l2_norm(left) * tree_l2_norm(right), 1e-12)
    def vector(tree):
        return jnp.concatenate([value.reshape(-1) for value in jax.tree_util.tree_leaves(tree)])
    hidden = _query_hidden(params, query, cfg)
    baseline_distribution = _distribution(params, initial, hidden, cfg)
    final_distribution = _distribution(params, final, hidden, cfg)
    tokens = jnp.where(support['event_mask'][..., None], support['tokens'], 0.)
    count = jnp.maximum(jnp.sum(support['event_mask']), 1)
    mean = jnp.sum(tokens, axis=(0, 1)) / count
    variance = jnp.sum(jnp.where(support['event_mask'][..., None], (tokens - mean) ** 2, 0.), axis=(0, 1)) / count
    return {
        'raw_support_statistics': jnp.concatenate([mean, variance]),
        'raw_write_gradient': raw_gradient,
        'clipped_write_gradient': clipped_gradient,
        'first_update': update,
        'final_fast_delta': jax.tree_util.tree_map(lambda w, w0: w - w0, final, initial),
        'query_gradient': query_gradient,
        'write_loss': write_loss,
        'query_loss_before': before,
        'query_loss_after_first_update': after_first,
        'query_loss_after_adaptation': after_final,
        'first_query_improvement': before - after_first,
        'local_improvement_prediction': -dot(query_gradient, update),
        'raw_write_gradient_query_cosine': cosine(raw_gradient, query_gradient),
        'first_update_query_gradient_cosine': cosine(update, query_gradient),
        'functional_before': vector(baseline_distribution),
        'functional_delta': vector(final_distribution) - vector(baseline_distribution),
        'read_delta': fast_read_residual(params, final, hidden, cfg.fast_config(), read_mode='delta', read_scale=cfg.read_scale),
    }


def sample_token(distribution, cfg: SketchModelConfig, key, deterministic=False):
    mixture_key, coordinate_key, pen_key, stop_key = jax.random.split(key, 4)
    if cfg.experiment == 'a':
        index = jnp.argmax(distribution['mixture_logits']) if deterministic else jax.random.categorical(mixture_key, distribution['mixture_logits'])
        xy = distribution['xy_mean'][index]
        if not deterministic:
            xy = xy + jnp.exp(distribution['xy_log_scale'][index]) * jax.random.normal(coordinate_key, (2,))
    else:
        xy = distribution['xy']
    if deterministic or cfg.experiment != 'a':
        pen = distribution['pen_logit'] >= 0
        stop = distribution['stop_logit'] >= 0
    else:
        pen = jax.random.bernoulli(pen_key, jax.nn.sigmoid(distribution['pen_logit']))
        stop = jax.random.bernoulli(stop_key, jax.nn.sigmoid(distribution['stop_logit']))
    return jnp.concatenate([jnp.where(stop, jnp.zeros_like(xy), xy), jnp.asarray([pen & ~stop, stop], dtype=xy.dtype)])


def generate(params, fast_state, cfg: SketchModelConfig, key, frame=None,
             context=None, initial_state=None, deterministic=False):
    """Empty-prefix generation from public inputs and a frozen fast state.

    B2 must use query_step with the pen environment to obtain executed states.
    There is intentionally no target, label, target-length, or support argument.
    """
    _check_context(cfg, context)
    if cfg.experiment == 'b2':
        raise ValueError('B2 requires an environment rollout using query_step.')
    frame = jnp.asarray([0., 0., 0., 1.], dtype=jnp.float32) if frame is None else jnp.asarray(frame)
    state = jnp.zeros((3,), dtype=jnp.float32) if initial_state is None else jnp.asarray(initial_state)
    def step(carry, timestep):
        recurrent, previous, stopped = carry
        recurrent, distribution = query_step(params, fast_state, cfg, recurrent, previous,
                                               timestep, frame, state, context)
        token = sample_token(distribution, cfg, jax.random.fold_in(key, timestep), deterministic)
        token = token.at[2].set(jnp.where(timestep == 0, 0., token[2]))
        token = jnp.where(stopped, jnp.zeros_like(token), token)
        event = ~stopped
        point = event & (token[3] < 0.5)
        stopped = stopped | (token[3] >= 0.5)
        return (recurrent, token, stopped), (token, event, point)
    (_, _, stopped), (tokens, events, points) = jax.lax.scan(
        step, (initial_query_carry(cfg), jnp.zeros((4,), dtype=jnp.float32), jnp.asarray(False)),
        jnp.arange(cfg.max_steps),
    )
    return {'tokens': tokens, 'event_mask': events, 'point_mask': points,
            'length': jnp.sum(points), 'stopped': stopped}
