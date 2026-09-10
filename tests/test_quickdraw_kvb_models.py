from dataclasses import replace
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from icil_jax_rlbench.models.fast_weight_ttt import initial_fast_state
from icil_jax_rlbench.quickdraw.kvb_models import (
    KVBModelConfig, _raw_from_history, adapt_support,
    autoregressive_distribution, generate, generate_from_fast_state, init_model, loss,
)
from icil_jax_rlbench.quickdraw.supervised_models import _distribution


def _config(**kwargs):
    return replace(KVBModelConfig(
        hidden_dim=8, num_heads=2, context_layers=1, decoder_layers=1,
        mlp_ratio=2, max_steps=5, mixture_components=2, dropout=0.,
        fast_dim=4, fast_hidden_dim=8), **kwargs)


def _batch():
    tokens = jnp.asarray([
        [[-.3, -.1, 0, 0], [.1, .2, 1, 0], [.2, .3, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
        [[.3, .1, 0, 0], [-.1, -.2, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
    ], dtype=jnp.float32)
    mask = jnp.arange(5)[None] < jnp.asarray([4, 3])[:, None]
    return {
        'support_tokens': jnp.stack((tokens, tokens.at[..., :2].multiply(-1)), axis=1),
        'support_mask': jnp.stack((mask, mask), axis=1),
        'query_tokens': tokens, 'query_mask': mask,
        'query_point_mask': mask & (tokens[..., 3] < .5),
        'example_mask': jnp.ones((2,), dtype=bool),
    }


@pytest.mark.parametrize('kwargs', [
    {'architecture': 'diffusion'}, {'dtype': 'bfloat16'}, {'inner_steps': 0},
    {'fast_dim': 1.5}, {'inner_lr_init': 1e-6}, {'inner_lr_min': -1.},
    {'fast_grad_clip_norm': -1.}, {'fast_drift_weight': float('nan')},
    {'first_order': 'false'},
])
def test_kvb_configuration_rejects_invalid_scientific_settings(kwargs):
    with pytest.raises(ValueError):
        _config(**kwargs)


def test_kvb_default_fast_capacity_and_float32_parameters():
    cfg = _config(fast_dim=64, fast_hidden_dim=128)
    params = init_model(jax.random.key(1), cfg)
    assert sum(leaf.size for leaf in jax.tree.leaves(params['adapter']['fast_init'])) == 16576
    assert cfg.inner_steps == 3 and cfg.first_order is False
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
    assert 'cross_attention' not in repr(jax.tree.structure(params['backbone']))
    assert 'read_gate' not in params['adapter']


def test_causal_parallel_likelihood_matches_cached_decoder():
    cfg, batch = _config(), _batch()
    params = init_model(jax.random.key(2), cfg, support_count=2)
    state, _ = adapt_support(params, batch['support_tokens'], batch['support_mask'], cfg)
    target = batch['query_tokens']
    baseline = autoregressive_distribution(params, state, target, cfg)
    changed = autoregressive_distribution(params, state, target.at[:, 2:].set(19.), cfg)
    for name in baseline:
        np.testing.assert_array_equal(baseline[name][:, :3], changed[name][:, :3])
    assert not np.allclose(baseline['xy_mean'][:, 3:], changed['xy_mean'][:, 3:])

    shape = (2, cfg.num_heads, cfg.max_steps, cfg.hidden_dim // cfg.num_heads)
    cache = tuple((jnp.zeros(shape), jnp.zeros(shape)) for _ in range(cfg.decoder_layers))
    previous = jnp.zeros((2, 1, 4))
    for position in range(cfg.max_steps):
        raw, cache, _ = _raw_from_history(
            params, state, previous, cfg, cache=cache, position=jnp.asarray(position))
        actual = _distribution(raw[:, 0], cfg)
        for name in baseline:
            np.testing.assert_allclose(actual[name], baseline[name][:, position], atol=1e-6, rtol=1e-5)
        previous = target[:, position:position + 1]


def test_support_padding_and_demo_order_do_not_change_adaptation():
    cfg, batch = _config(), _batch()
    params = init_model(jax.random.key(3), cfg, support_count=2)
    tokens, mask = batch['support_tokens'], batch['support_mask']
    adapt = jax.jit(lambda tokens, mask: adapt_support(params, tokens, mask, cfg))
    baseline, _ = adapt(tokens, mask)
    padded, _ = adapt(jnp.where(mask[..., None], tokens, jnp.nan), mask)
    reordered, _ = adapt(tokens[:, ::-1], mask[:, ::-1])
    altered, _ = adapt(tokens.at[..., :2].add(.7), mask)
    for first, second in zip(jax.tree.leaves(baseline), jax.tree.leaves(padded)):
        np.testing.assert_array_equal(first, second)
    for first, second in zip(jax.tree.leaves(baseline), jax.tree.leaves(reordered)):
        np.testing.assert_allclose(first, second, atol=1e-7, rtol=1e-6)
    assert any(not np.allclose(first, second, atol=1e-7, rtol=1e-6)
               for first, second in zip(jax.tree.leaves(baseline), jax.tree.leaves(altered)))


def test_frozen_read_has_no_support_encoder_or_write_projection_dependency():
    cfg, batch = _config(), _batch()
    params = init_model(jax.random.key(4), cfg, support_count=2)
    state, _ = adapt_support(params, batch['support_tokens'], batch['support_mask'], cfg)
    baseline = autoregressive_distribution(params, state, batch['query_tokens'], cfg)
    changed_backbone = dict(params['backbone'])
    for name in changed_backbone:
        if name.startswith('context'):
            changed_backbone[name] = jax.tree.map(jnp.zeros_like, changed_backbone[name])
    changed_adapter = dict(params['adapter'])
    for name in ('key_projection', 'value_projection'):
        changed_adapter[name] = jax.tree.map(jnp.zeros_like, changed_adapter[name])
    changed = dict(params, backbone=changed_backbone, adapter=changed_adapter)
    after = autoregressive_distribution(changed, state, batch['query_tokens'], cfg)
    for name in baseline:
        np.testing.assert_array_equal(baseline[name], after[name])
    assert 'support_tokens' not in inspect.signature(autoregressive_distribution).parameters
    assert 'support_tokens' not in inspect.signature(generate_from_fast_state).parameters
    initial = jax.tree.map(lambda x: jnp.broadcast_to(x, (2,) + x.shape), initial_fast_state(params['adapter']))
    _, _, delta = _raw_from_history(params, initial, batch['query_tokens'], cfg)
    np.testing.assert_array_equal(delta, jnp.zeros_like(delta))


def test_generation_adapts_once_then_reuses_frozen_state_and_rng():
    cfg, batch = _config(), _batch()
    params = init_model(jax.random.key(5), cfg, support_count=2)
    state, _ = adapt_support(params, batch['support_tokens'], batch['support_mask'], cfg)
    key = jax.random.key(7)
    sample = jax.jit(lambda key: generate(
        params, batch['support_tokens'], batch['support_mask'], cfg, key))
    actual, repeated, other = sample(key), sample(key), sample(jax.random.key(8))
    frozen = jax.jit(lambda state: generate_from_fast_state(params, state, cfg, key))(state)
    assert actual['tokens'].shape == (2, cfg.max_steps, 4)
    assert np.isfinite(actual['raw_actions']).all()
    assert not np.array_equal(actual['raw_actions'], other['raw_actions'])
    for name in actual:
        np.testing.assert_array_equal(actual[name], repeated[name])
        np.testing.assert_allclose(actual[name], frozen[name], atol=1e-6, rtol=1e-6)
    for row in range(2):
        stops = np.flatnonzero(np.asarray(actual['tokens'][row, :, 3]) >= .5)
        assert int(actual['length'][row]) == (int(stops[0]) if len(stops) else cfg.max_steps)
        assert bool(actual['stopped'][row]) == bool(len(stops))


def test_second_order_dropout_is_explicit_and_reproducible():
    cfg, batch = _config(dropout=.2), _batch()
    params = init_model(jax.random.key(6), cfg, support_count=2)
    objective = jax.jit(lambda key: loss(params, batch, cfg, key)[0])
    first, repeat, other = objective(jax.random.key(1)), objective(jax.random.key(1)), objective(jax.random.key(2))
    assert np.isfinite(first)
    np.testing.assert_array_equal(first, repeat)
    assert float(first) != float(other)
    with pytest.raises(ValueError, match='dropout RNG'):
        adapt_support(params, batch['support_tokens'], batch['support_mask'], cfg, training=True)
