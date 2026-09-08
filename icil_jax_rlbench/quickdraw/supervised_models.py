"""Supervised sketch Transformers with explicit demonstration cross-attention.

These policies implement ordinary in-context imitation, separately from the
fast-weight experiments. Tokens are absolute x/y, incoming pen, and STOP. No
retrieval embeddings, category identities, query lengths, or fast state enter
generation. Diffusion predicts noise over a fixed horizon, including absorbing
STOP padding; its denoiser never receives a clean query validity mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SupervisedModelConfig:
    architecture: str = 'autoregressive'
    hidden_dim: int = 256
    num_heads: int = 8
    context_layers: int = 4
    decoder_layers: int = 6
    mlp_ratio: int = 4
    max_steps: int = 128
    mixture_components: int = 10
    dropout: float = 0.1
    diffusion_steps: int = 100
    dtype: str = 'bfloat16'
    pen_loss_weight: float = 0.25
    stop_loss_weight: float = 0.25

    def __post_init__(self):
        if self.architecture not in ('autoregressive', 'diffusion'):
            raise ValueError('architecture must be autoregressive or diffusion.')
        for name in ('hidden_dim', 'num_heads', 'context_layers', 'decoder_layers',
                     'mlp_ratio', 'max_steps', 'mixture_components', 'diffusion_steps'):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if self.hidden_dim % self.num_heads:
            raise ValueError('hidden_dim must be divisible by num_heads.')
        if self.dtype not in ('float32', 'bfloat16'):
            raise ValueError('dtype must be float32 or bfloat16.')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1).')
        if self.pen_loss_weight < 0 or self.stop_loss_weight < 0:
            raise ValueError('Event loss weights must be nonnegative.')

    @property
    def compute_dtype(self):
        return jnp.float32 if self.dtype == 'float32' else jnp.bfloat16


def _dense(cfg, features, name, *, use_bias=True):
    return nn.Dense(features, dtype=cfg.compute_dtype, param_dtype=jnp.float32,
                    use_bias=use_bias, name=name)


def _norm(cfg, name):
    return nn.LayerNorm(dtype=cfg.compute_dtype, param_dtype=jnp.float32, name=name)


class _Attention(nn.Module):
    cfg: SupervisedModelConfig

    def setup(self):
        self.q = _dense(self.cfg, self.cfg.hidden_dim, 'q', use_bias=False)
        self.k = _dense(self.cfg, self.cfg.hidden_dim, 'k', use_bias=False)
        self.v = _dense(self.cfg, self.cfg.hidden_dim, 'v', use_bias=False)
        self.out = _dense(self.cfg, self.cfg.hidden_dim, 'out', use_bias=False)
        self.dropout = nn.Dropout(self.cfg.dropout)

    def _heads(self, value):
        # [batch,time,hidden] -> [batch,head,time,head_dim].
        return value.reshape(value.shape[:2] +
                             (self.cfg.num_heads, self.cfg.hidden_dim // self.cfg.num_heads)).transpose(0, 2, 1, 3)

    def project_kv(self, memory):
        return self._heads(self.k(memory)), self._heads(self.v(memory))

    def __call__(self, query, memory, mask, *, training, projected=None,
                 cache=None, position=None):
        q = self._heads(self.q(query))
        k, v = self.project_kv(memory) if projected is None else projected
        updated = None
        if cache is not None:
            # Generation inserts one event into each layer's fixed-size KV cache.
            k = jax.lax.dynamic_update_slice(cache[0], k, (0, 0, position, 0))
            v = jax.lax.dynamic_update_slice(cache[1], v, (0, 0, position, 0))
            updated = (k, v)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q.astype(jnp.float32),
                            k.astype(jnp.float32)) * (q.shape[-1] ** -0.5)
        mask = jnp.broadcast_to(mask, scores.shape)
        weights = jax.nn.softmax(jnp.where(mask, scores, -1e30), axis=-1)
        # An entirely masked support is a well-defined zero context.
        weights = jnp.where(mask, weights, 0.0)
        weights /= jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-8)
        weights = self.dropout(weights.astype(self.cfg.compute_dtype), deterministic=not training)
        value = jnp.einsum('bhqk,bhkd->bhqd', weights, v)
        value = value.transpose(0, 2, 1, 3).reshape(query.shape[:2] + (self.cfg.hidden_dim,))
        return self.out(value), updated


class _MLP(nn.Module):
    cfg: SupervisedModelConfig

    @nn.compact
    def __call__(self, value, *, training):
        value = nn.gelu(_dense(self.cfg, self.cfg.hidden_dim * self.cfg.mlp_ratio, 'up')(value))
        value = nn.Dropout(self.cfg.dropout)(value, deterministic=not training)
        return _dense(self.cfg, self.cfg.hidden_dim, 'down')(value)


class _ContextBlock(nn.Module):
    cfg: SupervisedModelConfig

    @nn.compact
    def __call__(self, value, mask, *, training):
        normalized = _norm(self.cfg, 'attention_norm')(value)
        residual, _ = _Attention(self.cfg, name='attention')(
            normalized, normalized, mask[:, None, None, :], training=training)
        value += nn.Dropout(self.cfg.dropout)(residual, deterministic=not training)
        residual = _MLP(self.cfg, name='mlp')(_norm(self.cfg, 'mlp_norm')(value), training=training)
        value += nn.Dropout(self.cfg.dropout)(residual, deterministic=not training)
        return jnp.where(mask[..., None], value, 0.0)


class _DecoderBlock(nn.Module):
    cfg: SupervisedModelConfig

    def setup(self):
        self.self_norm = _norm(self.cfg, 'self_norm')
        self.cross_norm = _norm(self.cfg, 'cross_norm')
        self.mlp_norm = _norm(self.cfg, 'mlp_norm')
        self.self_attention = _Attention(self.cfg, name='self_attention')
        self.cross_attention = _Attention(self.cfg, name='cross_attention')
        self.mlp = _MLP(self.cfg, name='mlp')
        self.dropout = nn.Dropout(self.cfg.dropout)
        if self.cfg.architecture == 'diffusion':
            self.time_affine = _dense(self.cfg, 6 * self.cfg.hidden_dim, 'time_affine')

    def project_memory(self, memory):
        return self.cross_attention.project_kv(memory)

    def __call__(self, value, memory, memory_mask, self_mask, *, training,
                 time_condition=None, cache=None, cross_cache=None, position=None):
        shifts = (0.0,) * 6 if time_condition is None else jnp.split(
            self.time_affine(nn.silu(time_condition))[:, None, :], 6, axis=-1)
        shift_s, scale_s, shift_c, scale_c, shift_m, scale_m = shifts
        normalized = self.self_norm(value) * (1 + scale_s) + shift_s
        residual, updated = self.self_attention(
            normalized, normalized, self_mask, training=training,
            cache=cache, position=position)
        value += self.dropout(residual, deterministic=not training)
        normalized = self.cross_norm(value) * (1 + scale_c) + shift_c
        residual, _ = self.cross_attention(
            normalized, memory, memory_mask[:, None, None, :], training=training,
            projected=cross_cache)
        value += self.dropout(residual, deterministic=not training)
        normalized = self.mlp_norm(value) * (1 + scale_m) + shift_m
        return value + self.dropout(self.mlp(normalized, training=training),
                                    deterministic=not training), updated


def _time_embedding(timesteps, width):
    half = (width + 1) // 2
    frequencies = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / max(half - 1, 1))
    angles = timesteps.astype(jnp.float32)[:, None] * frequencies[None, :]
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)[:, :width]


class _Policy(nn.Module):
    cfg: SupervisedModelConfig

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
        self.decoder_blocks = tuple(_DecoderBlock(cfg, name=f'decoder_{i}')
                                    for i in range(cfg.decoder_layers))
        self.context_norm = _norm(cfg, 'context_norm')
        self.output_norm = _norm(cfg, 'output_norm')
        output_dim = 5 * cfg.mixture_components + 2 if cfg.architecture == 'autoregressive' else 4
        # Output distributions/noise are evaluated in float32, including on GPU.
        self.output_head = nn.Dense(output_dim, dtype=jnp.float32, param_dtype=jnp.float32,
                                    kernel_init=nn.initializers.normal(0.02), name='output_head')
        if cfg.architecture == 'diffusion':
            self.time_up = _dense(cfg, cfg.hidden_dim * 4, 'time_up')
            self.time_down = _dense(cfg, cfg.hidden_dim, 'time_down')

    def encode(self, support_tokens, support_mask, *, training=False):
        _check_support(support_tokens, support_mask, self.cfg)
        batch, count, steps, _ = support_tokens.shape
        # Sketches have independent positions and bidirectional self-attention;
        # flattening only after encoding makes demonstration order immaterial.
        mask = support_mask.astype(bool).reshape(batch * count, steps)
        tokens = jnp.where(support_mask[..., None], support_tokens, 0.0).reshape(batch * count, steps, 4)
        value = self.context_input(tokens) + self.context_position[None, :steps].astype(self.cfg.compute_dtype)
        value = jnp.where(mask[..., None], value, 0.0)
        for block in self.context_blocks:
            value = block(value, mask, training=training)
        value = jnp.where(mask[..., None], self.context_norm(value), 0.0)
        return value.reshape(batch, count * steps, self.cfg.hidden_dim), mask.reshape(batch, count * steps)

    def project_memory(self, memory):
        return tuple(block.project_memory(memory) for block in self.decoder_blocks)

    def decode(self, inputs, memory, memory_mask, *, training=False,
               timesteps=None, cache=None, cross_cache=None, position=None):
        steps = inputs.shape[1]
        if cache is None:
            position_embedding = self.query_position[None, :steps]
            self_mask = jnp.ones((1, 1, steps, steps), dtype=bool)
            if self.cfg.architecture == 'autoregressive':
                self_mask = jnp.tril(self_mask)
        else:
            position_embedding = jax.lax.dynamic_slice(
                self.query_position, (position, 0), (1, self.cfg.hidden_dim))[None]
            self_mask = (jnp.arange(self.cfg.max_steps) <= position)[None, None, None, :]
        value = self.query_input(inputs) + position_embedding.astype(self.cfg.compute_dtype)
        condition = None
        if self.cfg.architecture == 'diffusion':
            if timesteps is None or cache is not None:
                raise ValueError('Diffusion requires timesteps and never uses a causal cache.')
            condition = self.time_down(nn.silu(self.time_up(_time_embedding(timesteps, self.cfg.hidden_dim))))
        updated = []
        for i, block in enumerate(self.decoder_blocks):
            value, layer_cache = block(
                value, memory, memory_mask, self_mask, training=training,
                time_condition=condition, cache=None if cache is None else cache[i],
                cross_cache=None if cross_cache is None else cross_cache[i], position=position)
            updated.append(layer_cache)
        return self.output_head(self.output_norm(value)), tuple(updated)

    def __call__(self, support_tokens, support_mask, inputs, *, training=False, timesteps=None):
        memory, memory_mask = self.encode(support_tokens, support_mask, training=training)
        return self.decode(inputs, memory, memory_mask, training=training, timesteps=timesteps)[0]


def _check_support(tokens, mask, cfg):
    if tokens.ndim != 4 or tokens.shape[-1] != 4 or tokens.shape[1] < 1:
        raise ValueError('support_tokens must have shape [batch,K,time,4], K >= 1.')
    if mask.shape != tokens.shape[:-1] or tokens.shape[2] > cfg.max_steps:
        raise ValueError('support_mask must match support tokens within max_steps.')


def _check_query(query, cfg):
    if query.ndim != 3 or query.shape[-2:] != (cfg.max_steps, 4):
        raise ValueError('query actions must have shape [batch,max_steps,4].')


def init_model(key, cfg: SupervisedModelConfig, support_count=4):
    if int(support_count) < 1:
        raise ValueError('support_count must be positive.')
    tokens = jnp.zeros((1, support_count, cfg.max_steps, 4), jnp.float32)
    return _Policy(cfg).init(
        key, tokens, jnp.ones(tokens.shape[:-1], dtype=bool), tokens[:, 0],
        timesteps=jnp.zeros((1,), jnp.int32) if cfg.architecture == 'diffusion' else None)['params']


def _rngs(key, training):
    if training and key is None:
        raise ValueError('Training requires an explicit dropout RNG key.')
    return {'dropout': key} if training else None


def _distribution(raw, cfg):
    count = cfg.mixture_components
    return {
        'mixture_logits': raw[..., :count],
        'xy_mean': raw[..., count:3 * count].reshape(raw.shape[:-1] + (count, 2)),
        'xy_log_scale': -1.5 + 2.5 * jnp.tanh(raw[..., 3 * count:5 * count].reshape(raw.shape[:-1] + (count, 2))),
        'pen_logit': raw[..., -2], 'stop_logit': raw[..., -1],
    }


def autoregressive_distribution(params, support_tokens, support_mask, query_tokens,
                                cfg, key=None, *, training=False):
    """Parallel teacher forcing: output t sees query events strictly before t."""
    if cfg.architecture != 'autoregressive':
        raise ValueError('This likelihood belongs to the autoregressive policy.')
    _check_query(query_tokens, cfg)
    history = jnp.concatenate((jnp.zeros_like(query_tokens[:, :1]), query_tokens[:, :-1]), axis=1)
    raw = _Policy(cfg).apply({'params': params}, support_tokens, support_mask, history,
                             training=training, rngs=_rngs(key, training))
    return _distribution(raw, cfg)


def diffusion_denoise(params, support_tokens, support_mask, noisy_actions,
                      timesteps, cfg, key=None, *, training=False):
    """Predict epsilon from noisy actions, public noise time, and demonstrations."""
    if cfg.architecture != 'diffusion':
        raise ValueError('This denoiser belongs to the diffusion policy.')
    _check_query(noisy_actions, cfg)
    if timesteps.shape != (noisy_actions.shape[0],):
        raise ValueError('timesteps must have shape [batch].')
    return _Policy(cfg).apply({'params': params}, support_tokens, support_mask, noisy_actions,
                              timesteps=timesteps, training=training, rngs=_rngs(key, training))


def diffusion_schedule(cfg):
    """Cosine alpha-bar schedule with capped terminal beta (DDPM indexing 0..S-1)."""
    times = jnp.arange(cfg.diffusion_steps + 1, dtype=jnp.float32) / cfg.diffusion_steps
    cumulative = jnp.cos((times + 0.008) / 1.008 * jnp.pi / 2) ** 2
    cumulative /= cumulative[0]
    beta = jnp.clip(1.0 - cumulative[1:] / cumulative[:-1], 1e-5, 0.999)
    alpha = 1.0 - beta
    alpha_bar = jnp.cumprod(alpha)
    previous = jnp.concatenate((jnp.ones((1,)), alpha_bar[:-1]))
    posterior_variance = beta * (1.0 - previous) / (1.0 - alpha_bar)
    return {'beta': beta, 'alpha': alpha, 'alpha_bar': alpha_bar,
            'previous_alpha_bar': previous, 'posterior_variance': posterior_variance}


def diffusion_targets(query_tokens, query_mask):
    """Train padding as repeated STOP actions; mask is used only to create labels."""
    absorbing = jnp.asarray([0.0, 0.0, 0.0, 1.0], jnp.float32)
    target = jnp.where(query_mask[..., None], query_tokens, absorbing)
    return target.at[..., 2:].set(2.0 * target[..., 2:] - 1.0)


def _example_mean(values, mask):
    weights = mask.astype(jnp.float32)
    return jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(jnp.sum(weights), 1.0)


def _sequence_mean(values, mask):
    return jnp.sum(jnp.where(mask, values, 0.0), axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1)


def loss(params, batch: Mapping, cfg: SupervisedModelConfig, key, training=True):
    """Ordinary supervised objective; every reported metric is an example mean."""
    target = batch['query_tokens'].astype(jnp.float32)
    _check_query(target, cfg)
    valid = batch.get('example_mask', jnp.ones((target.shape[0],), dtype=bool)).astype(bool)
    events = batch['query_mask'].astype(bool)
    points = batch['query_point_mask'].astype(bool) & events
    if events.shape != target.shape[:-1] or points.shape != events.shape or valid.shape != target.shape[:1]:
        raise ValueError('Query/example masks have incompatible shapes.')
    target = jnp.where(events[..., None] & valid[:, None, None], target, 0.0)
    if cfg.architecture == 'autoregressive':
        dist = autoregressive_distribution(params, batch['support_tokens'], batch['support_mask'],
                                           target, cfg, key, training=training)
        residual = (target[..., None, :2] - dist['xy_mean']) * jnp.exp(-dist['xy_log_scale'])
        component = -0.5 * jnp.sum(residual ** 2 + 2 * dist['xy_log_scale'] + jnp.log(2 * jnp.pi), axis=-1)
        coordinate = -jax.scipy.special.logsumexp(jax.nn.log_softmax(dist['mixture_logits']) + component, axis=-1)
        mean = jnp.sum(jax.nn.softmax(dist['mixture_logits'])[..., None] * dist['xy_mean'], axis=-2)
        per_example = {
            'coordinate_loss': _sequence_mean(coordinate, points),
            'pen_loss': _sequence_mean(jax.nn.softplus(dist['pen_logit']) - target[..., 2] * dist['pen_logit'], points),
            'stop_loss': _sequence_mean(jax.nn.softplus(dist['stop_logit']) - target[..., 3] * dist['stop_logit'], events),
            'coordinate_mse': _sequence_mean(jnp.mean((mean - target[..., :2]) ** 2, axis=-1), points),
            'pen_accuracy': _sequence_mean((dist['pen_logit'] >= 0) == (target[..., 2] >= .5), points),
            'stop_accuracy': _sequence_mean((dist['stop_logit'] >= 0) == (target[..., 3] >= .5), events),
        }
        per_example['loss'] = (per_example['coordinate_loss'] + cfg.pen_loss_weight * per_example['pen_loss'] +
                               cfg.stop_loss_weight * per_example['stop_loss'])
    else:
        time_key, noise_key, dropout_key = jax.random.split(key, 3)
        times = jax.random.randint(time_key, (target.shape[0],), 0, cfg.diffusion_steps)
        clean = diffusion_targets(target, events)
        noise = jax.random.normal(noise_key, clean.shape)
        alpha_bar = diffusion_schedule(cfg)['alpha_bar'][times, None, None]
        noisy = jnp.sqrt(alpha_bar) * clean + jnp.sqrt(1.0 - alpha_bar) * noise
        predicted = diffusion_denoise(params, batch['support_tokens'], batch['support_mask'],
                                      noisy, times, cfg, dropout_key, training=training)
        squared = (predicted - noise) ** 2
        per_example = {'noise_loss': jnp.mean(squared, axis=(1, 2)),
                       'noise_xy_loss': jnp.mean(squared[..., :2], axis=(1, 2)),
                       'noise_event_loss': jnp.mean(squared[..., 2:], axis=(1, 2))}
        per_example['loss'] = per_example['noise_loss']
    metrics = {name: _example_mean(value, valid) for name, value in per_example.items()}
    return metrics['loss'], metrics


def _sample_distribution(dist, key, deterministic):
    mix_key, xy_key, pen_key, stop_key = jax.random.split(key, 4)
    component = jnp.argmax(dist['mixture_logits'], axis=-1) if deterministic else jax.random.categorical(mix_key, dist['mixture_logits'])
    mean = jnp.take_along_axis(dist['xy_mean'], component[..., None, None], axis=-2)[..., 0, :]
    scale = jnp.take_along_axis(dist['xy_log_scale'], component[..., None, None], axis=-2)[..., 0, :]
    xy = mean if deterministic else mean + jnp.exp(scale) * jax.random.normal(xy_key, mean.shape)
    if deterministic:
        pen, stop = dist['pen_logit'] >= 0, dist['stop_logit'] >= 0
    else:
        pen = jax.random.bernoulli(pen_key, jax.nn.sigmoid(dist['pen_logit']))
        stop = jax.random.bernoulli(stop_key, jax.nn.sigmoid(dist['stop_logit']))
    return jnp.concatenate((xy, pen[..., None].astype(jnp.float32), stop[..., None].astype(jnp.float32)), axis=-1)


def _generation_result(tokens, raw_actions):
    stop_flags = tokens[..., 3] >= .5
    stopped = jnp.any(stop_flags, axis=1)
    first_stop = jnp.argmax(stop_flags, axis=1)
    length = jnp.where(stopped, first_stop, tokens.shape[1])
    positions = jnp.arange(tokens.shape[1])[None, :]
    point_mask = positions < length[:, None]
    event_mask = point_mask | (stopped[:, None] & (positions == length[:, None]))
    return {'tokens': tokens, 'raw_actions': raw_actions, 'event_mask': event_mask,
            'point_mask': point_mask, 'length': length, 'stopped': stopped}


def generate(params, support_tokens, support_mask, cfg: SupervisedModelConfig, key,
             *, deterministic=False):
    """Generate from support alone, retaining missing STOP and unbounded geometry.

    AR caches each layer's self/cross keys and values, avoiding repeated prefix
    decoding. Diffusion uses all declared DDPM steps. Its deterministic option
    disables reverse-step noise but still initializes from the supplied RNG key.
    Neither architecture forces a final STOP or clips coordinate failures.
    """
    _check_support(support_tokens, support_mask, cfg)
    model, variables = _Policy(cfg), {'params': params}
    batch = support_tokens.shape[0]
    memory, memory_mask = model.apply(variables, support_tokens, support_mask, method=model.encode)
    cross_cache = model.apply(variables, memory, method=model.project_memory)
    if cfg.architecture == 'autoregressive':
        cache_shape = (batch, cfg.num_heads, cfg.max_steps, cfg.hidden_dim // cfg.num_heads)
        cache = tuple((jnp.zeros(cache_shape, cfg.compute_dtype), jnp.zeros(cache_shape, cfg.compute_dtype))
                      for _ in range(cfg.decoder_layers))

        def step(carry, position):
            previous, kv_cache, rng = carry
            rng, sample_key = jax.random.split(rng)
            raw, updated = model.apply(variables, previous[:, None], memory, memory_mask,
                                       cache=kv_cache, cross_cache=cross_cache, position=position,
                                       method=model.decode)
            token = _sample_distribution(_distribution(raw[:, 0], cfg), sample_key, deterministic)
            return (token, updated, rng), token

        _, tokens = jax.lax.scan(step, (jnp.zeros((batch, 4)), cache, key), jnp.arange(cfg.max_steps))
        tokens = tokens.transpose(1, 0, 2)
        return _generation_result(tokens, tokens)

    schedule = diffusion_schedule(cfg)
    start_key, reverse_key = jax.random.split(key)
    initial = jax.random.normal(start_key, (batch, cfg.max_steps, 4))

    def reverse_step(carry, timestep):
        noisy, rng = carry
        rng, noise_key = jax.random.split(rng)
        predicted, _ = model.apply(variables, noisy, memory, memory_mask,
                                   timesteps=jnp.full((batch,), timestep), cross_cache=cross_cache,
                                   method=model.decode)
        beta, alpha, alpha_bar = (schedule[name][timestep] for name in ('beta', 'alpha', 'alpha_bar'))
        mean = (noisy - beta / jnp.sqrt(1.0 - alpha_bar) * predicted) / jnp.sqrt(alpha)
        noise = jnp.zeros_like(noisy) if deterministic else jax.random.normal(noise_key, noisy.shape)
        sample = mean + jnp.sqrt(schedule['posterior_variance'][timestep]) * noise
        return (sample, rng), None

    (raw, _), _ = jax.lax.scan(reverse_step, (initial, reverse_key), jnp.arange(cfg.diffusion_steps - 1, -1, -1))
    # Save continuous event predictions as well as the declared thresholded tokens.
    raw_actions = raw.at[..., 2:].set((raw[..., 2:] + 1.0) / 2.0)
    tokens = raw.at[..., 2:].set((raw[..., 2:] >= 0).astype(jnp.float32))
    return _generation_result(tokens, raw_actions)
