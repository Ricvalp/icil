"""Autoregressive sketch Transformer with task-local, second-order KVB WRITE.

The support Transformer runs once per task. All valid demonstration events form
one pooled WRITE batch, reused for every inner step. Query decoding can access
only slow parameters, the resulting fast state, and causal query history.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig, TTTAdaptConfig, adapt_encoded_support,
    fast_model_apply, fast_read_residual, init_fast_adapter_params,
    initial_fast_state, inner_learning_rates, project_key_value,
)
from .supervised_models import (
    SupervisedModelConfig, _Attention, _ContextBlock, _MLP, _check_query,
    _check_support, _dense, _distribution, _example_mean, _generation_result,
    _norm, _rngs, _sample_distribution, _sequence_mean,
)


@dataclass(frozen=True)
class KVBModelConfig(SupervisedModelConfig):
    dtype: str = 'float32'
    fast_dim: int = 64
    fast_hidden_dim: int = 128
    inner_steps: int = 3
    inner_lr_init: float = 0.03
    inner_lr_min: float = 1e-5
    first_order: bool = False
    fast_grad_clip_norm: float = 1.0
    fast_update_clip_norm: float = 0.0
    fast_drift_weight: float = 0.0
    read_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.architecture != 'autoregressive':
            raise ValueError('KVB currently supports only the autoregressive Transformer.')
        if self.dtype != 'float32':
            raise ValueError('KVB uses float32 for the Transformer and second-order updates.')
        for name in ('fast_dim', 'fast_hidden_dim', 'inner_steps'):
            value = getattr(self, name)
            if int(value) != value or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        for name in ('inner_lr_init', 'inner_lr_min', 'fast_grad_clip_norm',
                     'fast_update_clip_norm', 'fast_drift_weight', 'read_scale'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
        if self.inner_lr_init <= self.inner_lr_min:
            raise ValueError('inner_lr_init must be greater than inner_lr_min.')
        if not isinstance(self.first_order, bool):
            raise ValueError('first_order must be a boolean FOMAML ablation flag.')

    def fast_config(self):
        return FastWeightTTTConfig(
            hidden_dim=self.hidden_dim, fast_dim=self.fast_dim,
            fast_hidden_dim=self.fast_hidden_dim, fast_model='mlp',
            inner_lr_init=self.inner_lr_init, inner_lr_min=self.inner_lr_min,
        )

    def adapt_config(self):
        return TTTAdaptConfig(
            write_objective='kvb', write_steps_per_segment=self.inner_steps,
            first_order=self.first_order, fast_grad_clip_norm=self.fast_grad_clip_norm,
            fast_update_clip_norm=self.fast_update_clip_norm,
            fast_drift_weight=self.fast_drift_weight,
            read_mode='delta', read_scale=self.read_scale,
        )


class _CausalBlock(nn.Module):
    cfg: KVBModelConfig

    @nn.compact
    def __call__(self, value, mask, *, training, cache=None, position=None):
        normalized = _norm(self.cfg, 'self_norm')(value)
        residual, updated = _Attention(self.cfg, name='self_attention')(
            normalized, normalized, mask, training=training,
            cache=cache, position=position)
        value += nn.Dropout(self.cfg.dropout)(residual, deterministic=not training)
        residual = _MLP(self.cfg, name='mlp')(
            _norm(self.cfg, 'mlp_norm')(value), training=training)
        return value + nn.Dropout(self.cfg.dropout)(
            residual, deterministic=not training), updated


class _Backbone(nn.Module):
    cfg: KVBModelConfig

    def setup(self):
        cfg = self.cfg
        self.context_input = _dense(cfg, cfg.hidden_dim, 'context_input')
        self.query_input = _dense(cfg, cfg.hidden_dim, 'query_input')
        self.context_position = self.param('context_position', nn.initializers.normal(0.02),
                                           (cfg.max_steps, cfg.hidden_dim), jnp.float32)
        self.query_position = self.param('query_position', nn.initializers.normal(0.02),
                                         (cfg.max_steps, cfg.hidden_dim), jnp.float32)
        self.context_blocks = tuple(_ContextBlock(cfg, name=f'context_{i}')
                                    for i in range(cfg.context_layers))
        self.decoder_blocks = tuple(_CausalBlock(cfg, name=f'decoder_{i}')
                                    for i in range(cfg.decoder_layers))
        self.context_norm = _norm(cfg, 'context_norm')
        self.output_norm = _norm(cfg, 'output_norm')
        self.output_head = nn.Dense(
            5 * cfg.mixture_components + 2, dtype=jnp.float32, param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(0.02), name='output_head')

    def encode(self, support_tokens, support_mask, *, training=False):
        _check_support(support_tokens, support_mask, self.cfg)
        batch, count, steps, _ = support_tokens.shape
        # Each demonstration has its own positions and attention mask. Pool only
        # after encoding: [B,K,T,4] -> [B*K,T,H] -> [B,K*T,H].
        mask = support_mask.astype(bool).reshape(batch * count, steps)
        tokens = jnp.where(support_mask[..., None], support_tokens, 0.0)
        value = self.context_input(tokens.reshape(batch * count, steps, 4))
        value += self.context_position[None, :steps]
        value = jnp.where(mask[..., None], value, 0.0)
        for block in self.context_blocks:
            value = block(value, mask, training=training)
        value = jnp.where(mask[..., None], self.context_norm(value), 0.0)
        return value.reshape(batch, count * steps, self.cfg.hidden_dim), mask.reshape(batch, count * steps)

    def decode_hidden(self, inputs, *, training=False, cache=None, position=None):
        steps = inputs.shape[1]
        if cache is None:
            positions = self.query_position[None, :steps]
            mask = jnp.tril(jnp.ones((1, 1, steps, steps), dtype=bool))
        else:
            positions = jax.lax.dynamic_slice(
                self.query_position, (position, 0), (1, self.cfg.hidden_dim))[None]
            mask = (jnp.arange(self.cfg.max_steps) <= position)[None, None, None, :]
        value = self.query_input(inputs) + positions
        updated = []
        for i, block in enumerate(self.decoder_blocks):
            value, layer_cache = block(
                value, mask, training=training,
                cache=None if cache is None else cache[i], position=position)
            updated.append(layer_cache)
        return self.output_norm(value), tuple(updated)

    def head(self, hidden):
        return self.output_head(hidden)

    def __call__(self, support_tokens, support_mask, inputs):
        # Initialize both independent encoders; support never enters decoding.
        self.encode(support_tokens, support_mask)
        hidden, _ = self.decode_hidden(inputs)
        return self.head(hidden)


def init_model(key, cfg: KVBModelConfig, support_count=4):
    if int(support_count) != support_count or support_count < 1:
        raise ValueError('support_count must be a positive integer.')
    backbone_key, adapter_key = jax.random.split(key)
    tokens = jnp.zeros((1, support_count, cfg.max_steps, 4), jnp.float32)
    backbone = _Backbone(cfg).init(
        backbone_key, tokens, jnp.ones(tokens.shape[:-1], dtype=bool), tokens[:, 0])['params']
    adapter = init_fast_adapter_params(adapter_key, cfg.fast_config())
    # Delta READ has no learned gate. W0 and the positive per-tensor rates remain
    # slow parameters; task-adapted states are returned separately, never stored.
    adapter.pop('read_gate')
    return {'backbone': backbone, 'adapter': adapter}


def encode_support(params, support_tokens, support_mask, cfg, key=None, *, training=False):
    """Encode each support sketch once, without detaching its WRITE features."""
    model = _Backbone(cfg)
    return model.apply(
        {'params': params['backbone']}, support_tokens, support_mask,
        training=training, rngs=_rngs(key, training), method=model.encode)


def adapt_support(params, support_tokens, support_mask, cfg, key=None, *, training=False):
    """Return batched fast states and scalar-per-task adaptation diagnostics."""
    encoded, mask = encode_support(params, support_tokens, support_mask, cfg, key, training=training)
    adapter, fast_cfg, adapt_cfg = params['adapter'], cfg.fast_config(), cfg.adapt_config()

    def adapt_task(registers, valid):
        # A SINGLE segment contains every valid event from all K demos. The
        # existing writer reuses this same batch for exactly inner_steps updates.
        state, trace = adapt_encoded_support(
            adapter, registers[None], valid[None], fast_cfg, adapt_cfg)
        keys, values = project_key_value(adapter, registers)

        def reconstruction(fast):
            squared = jnp.mean((fast_model_apply(fast, keys, fast_cfg) - values) ** 2, axis=-1)
            return jnp.sum(jnp.where(valid, squared, 0.0)) / jnp.maximum(jnp.sum(valid), 1)

        metrics = {name: value[0] for name, value in trace.items()}
        metrics['write_loss_before'] = reconstruction(initial_fast_state(adapter))
        metrics['write_loss_after'] = reconstruction(state)
        return state, metrics

    # Every task starts at the shared learned W0; no support is averaged across B.
    return jax.vmap(adapt_task)(encoded, mask)


def _raw_from_history(params, fast_state, history, cfg, key=None, *, training=False,
                      cache=None, position=None):
    model, variables = _Backbone(cfg), {'params': params['backbone']}
    hidden, updated = model.apply(
        variables, history, training=training, cache=cache, position=position,
        rngs=_rngs(key, training), method=model.decode_hidden)
    residual = jax.vmap(lambda state, query: fast_read_residual(
        params['adapter'], state, query, cfg.fast_config(),
        read_mode='delta', read_scale=cfg.read_scale))(fast_state, hidden)
    raw = model.apply(variables, hidden + residual, method=model.head)
    return raw, updated, residual


def autoregressive_distribution(params, fast_state, query_tokens, cfg, key=None, *, training=False):
    """READ from frozen fast state and strictly earlier query events only."""
    _check_query(query_tokens, cfg)
    history = jnp.concatenate((jnp.zeros_like(query_tokens[:, :1]), query_tokens[:, :-1]), axis=1)
    raw, _, _ = _raw_from_history(params, fast_state, history, cfg, key, training=training)
    return _distribution(raw, cfg)


def loss(params, batch: Mapping, cfg: KVBModelConfig, key, training=True):
    """Independent query likelihood; WRITE is never added to the outer loss."""
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
    fast_state, trace = adapt_support(
        params, batch['support_tokens'], support_mask, cfg, support_key, training=training)
    history = jnp.concatenate((jnp.zeros_like(target[:, :1]), target[:, :-1]), axis=1)
    raw, _, read_delta = _raw_from_history(
        params, fast_state, history, cfg, query_key, training=training)
    dist = _distribution(raw, cfg)
    residual = (target[..., None, :2] - dist['xy_mean']) * jnp.exp(-dist['xy_log_scale'])
    component = -0.5 * jnp.sum(residual ** 2 + 2 * dist['xy_log_scale'] + jnp.log(2 * jnp.pi), axis=-1)
    coordinate = -jax.scipy.special.logsumexp(
        jax.nn.log_softmax(dist['mixture_logits']) + component, axis=-1)
    mean = jnp.sum(jax.nn.softmax(dist['mixture_logits'])[..., None] * dist['xy_mean'], axis=-2)
    per_example = {
        'coordinate_loss': _sequence_mean(coordinate, points),
        'pen_loss': _sequence_mean(jax.nn.softplus(dist['pen_logit']) - target[..., 2] * dist['pen_logit'], points),
        'stop_loss': _sequence_mean(jax.nn.softplus(dist['stop_logit']) - target[..., 3] * dist['stop_logit'], events),
        'coordinate_mse': _sequence_mean(jnp.mean((mean - target[..., :2]) ** 2, axis=-1), points),
        'pen_accuracy': _sequence_mean((dist['pen_logit'] >= 0) == (target[..., 2] >= .5), points),
        'stop_accuracy': _sequence_mean((dist['stop_logit'] >= 0) == (target[..., 3] >= .5), events),
        'read_delta_rms': jnp.sqrt(_sequence_mean(jnp.mean(read_delta ** 2, axis=-1), events)),
        **trace,
    }
    per_example['loss'] = (per_example['coordinate_loss'] + cfg.pen_loss_weight * per_example['pen_loss'] +
                           cfg.stop_loss_weight * per_example['stop_loss'])
    rates = jnp.stack(jax.tree_util.tree_leaves(inner_learning_rates(params['adapter'], cfg.fast_config())))
    per_example['inner_lr_mean'] = jnp.full(valid.shape, jnp.mean(rates))
    metrics = {name: _example_mean(value, valid) for name, value in per_example.items()}
    return metrics['loss'], metrics


def generate_from_fast_state(params, fast_state, cfg, key, *, deterministic=False):
    """Generate an entire sketch with an unchanged, already-adapted fast state."""
    batch = jax.tree_util.tree_leaves(fast_state)[0].shape[0]
    shape = (batch, cfg.num_heads, cfg.max_steps, cfg.hidden_dim // cfg.num_heads)
    cache = tuple((jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32))
                  for _ in range(cfg.decoder_layers))

    def step(carry, position):
        previous, kv_cache, rng = carry
        rng, sample_key = jax.random.split(rng)
        raw, updated, _ = _raw_from_history(
            params, fast_state, previous[:, None], cfg, cache=kv_cache, position=position)
        token = _sample_distribution(_distribution(raw[:, 0], cfg), sample_key, deterministic)
        return (token, updated, rng), token

    _, tokens = jax.lax.scan(
        step, (jnp.zeros((batch, 4), jnp.float32), cache, key), jnp.arange(cfg.max_steps))
    tokens = tokens.transpose(1, 0, 2)
    return _generation_result(tokens, tokens)


def generate(params, support_tokens, support_mask, cfg: KVBModelConfig, key, *, deterministic=False):
    """Adapt once on support, then generate from its frozen task-local state."""
    fast_state, _ = adapt_support(params, support_tokens, support_mask, cfg)
    return generate_from_fast_state(params, fast_state, cfg, key, deterministic=deterministic)
