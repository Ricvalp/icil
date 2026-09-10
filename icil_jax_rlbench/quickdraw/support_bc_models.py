"""Second-order fast adaptation with a causal support behavior-cloning WRITE.

The query architecture and fast READ match KVB. Demonstrations affect queries
only through task-local gradient updates; no direct support memory is retained.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

from icil_jax_rlbench.models.fast_weight_ttt import (
    apply_fast_gradient, fast_drift_penalty, fast_read_residual,
    initial_fast_state, inner_learning_rates, tree_difference_norm, tree_l2_norm,
)
from . import kvb_models
from .kvb_models import KVBModelConfig, _Backbone, _CausalBlock
from .supervised_models import (
    _check_query, _check_support, _dense, _distribution, _example_mean,
    _generation_result, _norm, _rngs, _sample_distribution, _sequence_mean,
)


@dataclass(frozen=True)
class SupportBCModelConfig(KVBModelConfig):
    def adapt_config(self):
        return replace(super().adapt_config(), write_objective='action_bc')


class _BCBackbone(_Backbone):
    """Reuse KVB's causal decoder and head without its bidirectional writer."""

    def setup(self):
        cfg = self.cfg
        self.query_input = _dense(cfg, cfg.hidden_dim, 'query_input')
        self.query_position = self.param('query_position', nn.initializers.normal(0.02),
                                         (cfg.max_steps, cfg.hidden_dim), jnp.float32)
        self.decoder_blocks = tuple(_CausalBlock(cfg, name=f'decoder_{i}')
                                    for i in range(cfg.decoder_layers))
        self.output_norm = _norm(cfg, 'output_norm')
        self.output_head = nn.Dense(
            5 * cfg.mixture_components + 2, dtype=jnp.float32, param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(0.02), name='output_head')

    def __call__(self, inputs):
        hidden, _ = self.decode_hidden(inputs)
        return self.head(hidden)


def init_model(key, cfg: SupportBCModelConfig, support_count=4):
    # Keep every active parameter identical to the matched KVB initialization.
    # The temporary bidirectional writer is discarded, never optimized or saved.
    params = kvb_models.init_model(key, cfg, support_count=support_count)
    return {
        'backbone': {name: value for name, value in params['backbone'].items()
                     if not name.startswith('context')},
        'adapter': {name: value for name, value in params['adapter'].items()
                    if name not in ('key_projection', 'value_projection')},
    }


def _prediction_metrics(raw, target, points, events, cfg):
    """The same pooled Gaussian-mixture/pen/STOP objective for WRITE and READ."""
    dist = _distribution(raw, cfg)
    residual = (target[..., None, :2] - dist['xy_mean']) * jnp.exp(-dist['xy_log_scale'])
    component = -0.5 * jnp.sum(
        residual ** 2 + 2 * dist['xy_log_scale'] + jnp.log(2 * jnp.pi), axis=-1)
    coordinate = -jax.scipy.special.logsumexp(
        jax.nn.log_softmax(dist['mixture_logits']) + component, axis=-1)
    mean = jnp.sum(jax.nn.softmax(dist['mixture_logits'])[..., None] * dist['xy_mean'], axis=-2)
    metrics = {
        'coordinate_loss': _sequence_mean(coordinate, points),
        'pen_loss': _sequence_mean(jax.nn.softplus(dist['pen_logit']) - target[..., 2] * dist['pen_logit'], points),
        'stop_loss': _sequence_mean(jax.nn.softplus(dist['stop_logit']) - target[..., 3] * dist['stop_logit'], events),
        'coordinate_mse': _sequence_mean(jnp.mean((mean - target[..., :2]) ** 2, axis=-1), points),
        'pen_accuracy': _sequence_mean((dist['pen_logit'] >= 0) == (target[..., 2] >= .5), points),
        'stop_accuracy': _sequence_mean((dist['stop_logit'] >= 0) == (target[..., 3] >= .5), events),
    }
    metrics['loss'] = (metrics['coordinate_loss'] + cfg.pen_loss_weight * metrics['pen_loss']
                       + cfg.stop_loss_weight * metrics['stop_loss'])
    return metrics


def encode_support(params, support_tokens, support_mask, cfg, key=None, *, training=False):
    """Cache causal features once, retaining the gradient graph and dropout draw.

    Each demo starts with its own zero/BOS history and position sequence. Only
    the resulting features and target masks are pooled across its K demos.
    """
    _check_support(support_tokens, support_mask, cfg)
    batch, count, steps, _ = support_tokens.shape
    events = support_mask.astype(bool)
    target = jnp.where(events[..., None], support_tokens.astype(jnp.float32), 0.0)
    points = events & (target[..., 3] < .5)
    # [B,K,T,4] -> [B*K,T,4] keeps causal history inside each demonstration.
    sequences = target.reshape(batch * count, steps, 4)
    history = jnp.concatenate((jnp.zeros_like(sequences[:, :1]), sequences[:, :-1]), axis=1)
    model = _BCBackbone(cfg)
    hidden, _ = model.apply({'params': params['backbone']}, history,
        training=training, rngs=_rngs(key, training), method=model.decode_hidden)
    return {
        'hidden': hidden.reshape(batch, count * steps, cfg.hidden_dim),
        'target': target.reshape(batch, count * steps, 4),
        'point_mask': points.reshape(batch, count * steps),
        'event_mask': events.reshape(batch, count * steps),
    }


def _read_encoded(params, fast_state, hidden, cfg):
    residual = fast_read_residual(params['adapter'], fast_state, hidden,
        cfg.fast_config(), read_mode='delta', read_scale=cfg.read_scale)
    model = _BCBackbone(cfg)
    raw = model.apply({'params': params['backbone']}, hidden + residual, method=model.head)
    return raw, residual


def support_write_loss(params, fast_state, encoded_task, cfg):
    """Pure support likelihood for one task; no query tokens or task pooling."""
    raw, _ = _read_encoded(params, fast_state, encoded_task['hidden'], cfg)
    return _prediction_metrics(raw[None], encoded_task['target'][None],
        encoded_task['point_mask'][None], encoded_task['event_mask'][None], cfg)['loss'][0]


def adapt_support(params, support_tokens, support_mask, cfg, key=None, *, training=False):
    """Reuse the same pooled support batch for exactly inner_steps GD updates."""
    encoded = encode_support(params, support_tokens, support_mask, cfg, key, training=training)
    initial = initial_fast_state(params['adapter'])
    rates = inner_learning_rates(params['adapter'], cfg.fast_config())
    adapt_cfg = cfg.adapt_config()

    def adapt_task(task):
        active = jnp.any(task['event_mask'])

        def objective(state):
            value = support_write_loss(params, state, task, cfg)
            if cfg.fast_drift_weight > 0:
                value += cfg.fast_drift_weight * fast_drift_penalty(state, initial)
            # Empty support is an exact no-op, including the drift regularizer.
            return jnp.where(active, value, jnp.zeros_like(value))

        state, accumulated = initial, None
        for _ in range(cfg.inner_steps):
            value, gradient = jax.value_and_grad(objective)(state)
            raw_norm = tree_l2_norm(gradient)
            state, _, update = apply_fast_gradient(state, gradient, rates, adapt_cfg)
            metrics = {'write_loss': value, 'fast_grad_norm': raw_norm,
                       'fast_update_norm': tree_l2_norm(update)}
            accumulated = metrics if accumulated is None else jax.tree.map(jnp.add, accumulated, metrics)
        metrics = jax.tree.map(lambda value: value / cfg.inner_steps, accumulated)
        metrics.update(write_loss_before=support_write_loss(params, initial, task, cfg),
                       write_loss_after=support_write_loss(params, state, task, cfg),
                       fast_delta_norm=tree_difference_norm(state, initial))
        return state, metrics

    # [B,...] is the task axis; every task starts independently at the learned W0.
    return jax.vmap(adapt_task)(encoded)


def _raw_from_history(params, fast_state, history, cfg, key=None, *, training=False,
                      cache=None, position=None):
    model = _BCBackbone(cfg)
    hidden, updated = model.apply({'params': params['backbone']}, history,
        training=training, cache=cache, position=position,
        rngs=_rngs(key, training), method=model.decode_hidden)
    raw, residual = jax.vmap(lambda state, features:
        _read_encoded(params, state, features, cfg))(fast_state, hidden)
    return raw, updated, residual


def autoregressive_distribution(params, fast_state, query_tokens, cfg, key=None, *, training=False):
    """Frozen READ from permitted causal query history only."""
    _check_query(query_tokens, cfg)
    history = jnp.concatenate((jnp.zeros_like(query_tokens[:, :1]), query_tokens[:, :-1]), axis=1)
    raw, _, _ = _raw_from_history(params, fast_state, history, cfg, key, training=training)
    return _distribution(raw, cfg)


def loss(params, batch: Mapping, cfg: SupportBCModelConfig, key, training=True):
    """Outer independent-query likelihood, differentiated through support BC."""
    target = batch['query_tokens'].astype(jnp.float32)
    _check_query(target, cfg)
    valid = batch.get('example_mask', jnp.ones((target.shape[0],), dtype=bool)).astype(bool)
    events = batch['query_mask'].astype(bool)
    points = batch['query_point_mask'].astype(bool) & events
    if events.shape != target.shape[:-1] or points.shape != events.shape or valid.shape != target.shape[:1]:
        raise ValueError('Query/example masks have incompatible shapes.')
    target = jnp.where(events[..., None] & valid[:, None, None], target, 0.0)
    support_key, query_key = jax.random.split(key)
    support_mask = batch['support_mask'].astype(bool) & valid[:, None, None]
    fast_state, trace = adapt_support(params, batch['support_tokens'], support_mask,
        cfg, support_key, training=training)
    history = jnp.concatenate((jnp.zeros_like(target[:, :1]), target[:, :-1]), axis=1)
    raw, _, delta = _raw_from_history(params, fast_state, history, cfg, query_key, training=training)
    per_example = _prediction_metrics(raw, target, points, events, cfg)
    per_example.update(trace)
    per_example['read_delta_rms'] = jnp.sqrt(_sequence_mean(jnp.mean(delta ** 2, axis=-1), events))
    rates = jnp.stack(jax.tree.leaves(inner_learning_rates(params['adapter'], cfg.fast_config())))
    per_example['inner_lr_mean'] = jnp.full(valid.shape, jnp.mean(rates))
    metrics = {name: _example_mean(value, valid) for name, value in per_example.items()}
    return metrics['loss'], metrics


def generate_from_fast_state(params, fast_state, cfg, key, *, deterministic=False):
    """Decode with cached causal attention and unchanged adapted task state."""
    batch = jax.tree.leaves(fast_state)[0].shape[0]
    shape = (batch, cfg.num_heads, cfg.max_steps, cfg.hidden_dim // cfg.num_heads)
    cache = tuple((jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32))
                  for _ in range(cfg.decoder_layers))

    def step(carry, position):
        previous, kv_cache, rng = carry
        rng, sample_key = jax.random.split(rng)
        raw, updated, _ = _raw_from_history(params, fast_state, previous[:, None], cfg,
                                           cache=kv_cache, position=position)
        token = _sample_distribution(_distribution(raw[:, 0], cfg), sample_key, deterministic)
        return (token, updated, rng), token

    _, tokens = jax.lax.scan(step,
        (jnp.zeros((batch, 4), jnp.float32), cache, key), jnp.arange(cfg.max_steps))
    tokens = tokens.transpose(1, 0, 2)
    return _generation_result(tokens, tokens)


def generate(params, support_tokens, support_mask, cfg: SupportBCModelConfig, key, *, deterministic=False):
    fast_state, _ = adapt_support(params, support_tokens, support_mask, cfg)
    return generate_from_fast_state(params, fast_state, cfg, key, deterministic=deterministic)
