"""Matched query architecture and task-local support behavior-cloning WRITE."""

from dataclasses import asdict
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from icil_jax_rlbench.models.fast_weight_ttt import (
    apply_fast_gradient, fast_drift_penalty, initial_fast_state, inner_learning_rates,
)
from icil_jax_rlbench.quickdraw import kvb_models as kvb, support_bc_models as bc
from icil_jax_rlbench.quickdraw.supervised_models import _distribution
from test_quickdraw_kvb_models import _batch, _config as _kvb_config


def _config(**kwargs):
    return bc.SupportBCModelConfig(**asdict(_kvb_config(**kwargs)))


def _equal_trees(left, right, *, exact=False):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        if exact:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)


def test_active_initial_parameters_and_unadapted_query_predictions_match_kvb():
    cfg, batch = _config(), _batch()
    key = jax.random.key(12)
    original = kvb.init_model(key, _kvb_config(), support_count=2)
    params = bc.init_model(key, cfg, support_count=2)
    assert cfg.adapt_config().write_objective == 'action_bc'
    assert cfg.inner_steps == 3 and not cfg.first_order
    assert not any(name.startswith('context') for name in params['backbone'])
    assert set(params['adapter']) == {'fast_init', 'inner_lr_raw', 'query_projection', 'read_projection'}
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(params))
    for section in params:
        for name, value in params[section].items():
            _equal_trees(value, original[section][name], exact=True)
    initial = jax.tree.map(lambda value: jnp.broadcast_to(value, (2,) + value.shape),
                           initial_fast_state(params['adapter']))
    actual = bc.autoregressive_distribution(params, initial, batch['query_tokens'], cfg)
    expected = kvb.autoregressive_distribution(original, initial, batch['query_tokens'], _kvb_config())
    _equal_trees(actual, expected, exact=True)


def test_support_features_are_causal_and_restart_at_every_demonstration():
    cfg, batch = _config(), _batch()
    params = bc.init_model(jax.random.key(2), cfg, support_count=2)
    tokens, mask = batch['support_tokens'], batch['support_mask']
    original = bc.encode_support(params, tokens, mask, cfg)
    changed_tokens = tokens.at[:, 0, 1, :2].add(.9)
    changed = bc.encode_support(params, changed_tokens, mask, cfg)
    # Flattening happens after per-demo causal encoding, so the second demo
    # cannot observe the first one's changed labels or position/history.
    np.testing.assert_array_equal(original['hidden'][:, cfg.max_steps:], changed['hidden'][:, cfg.max_steps:])
    np.testing.assert_array_equal(original['hidden'][:, :2], changed['hidden'][:, :2])
    assert not np.allclose(original['hidden'][:, 2:cfg.max_steps], changed['hidden'][:, 2:cfg.max_steps])
    np.testing.assert_array_equal(original['hidden'][:, 0], original['hidden'][:, cfg.max_steps])
    assert 'query_tokens' not in inspect.signature(bc.encode_support).parameters
    assert 'query_tokens' not in inspect.signature(bc.adapt_support).parameters


@pytest.mark.parametrize('steps', [1, 3, 5])
def test_pooled_updates_reuse_cached_support_features_and_dropout(steps):
    cfg, batch = _config(inner_steps=steps, dropout=.2, fast_drift_weight=.03), _batch()
    params = bc.init_model(jax.random.key(3), cfg, support_count=2)
    key = jax.random.key(4)
    cached = bc.encode_support(params, batch['support_tokens'], batch['support_mask'], cfg, key, training=True)
    task = jax.tree.map(lambda value: value[0], cached)
    initial = initial_fast_state(params['adapter'])
    rates = inner_learning_rates(params['adapter'], cfg.fast_config())

    def objective(state):
        return bc.support_write_loss(params, state, task, cfg) + cfg.fast_drift_weight * fast_drift_penalty(state, initial)

    expected = initial
    for _ in range(steps):
        gradient = jax.grad(objective)(expected)
        expected, _, _ = apply_fast_gradient(expected, gradient, rates, cfg.adapt_config())
    actual, trace = jax.jit(lambda: bc.adapt_support(params, batch['support_tokens'],
        batch['support_mask'], cfg, key, training=True))()
    _equal_trees(expected, jax.tree.map(lambda value: value[0], actual))
    assert float(trace['write_loss_after'][0]) < float(trace['write_loss_before'][0])
    assert float(trace['fast_delta_norm'][0]) > 0
    with pytest.raises(ValueError, match='dropout RNG'):
        bc.adapt_support(params, batch['support_tokens'], batch['support_mask'], cfg, training=True)


def test_task_isolation_padding_and_empty_support_noop_including_drift():
    cfg, batch = _config(fast_drift_weight=1.), _batch()
    params = bc.init_model(jax.random.key(5), cfg, support_count=2)
    tokens, mask = batch['support_tokens'], batch['support_mask']
    adapt = jax.jit(lambda tokens, mask: bc.adapt_support(params, tokens, mask, cfg))
    original, _ = adapt(tokens, mask)
    padded, _ = adapt(jnp.where(mask[..., None], tokens, jnp.nan), mask)
    _equal_trees(original, padded, exact=True)
    changed, _ = adapt(tokens.at[1, ..., :2].add(.8), mask)
    _equal_trees(jax.tree.map(lambda value: value[0], original),
                 jax.tree.map(lambda value: value[0], changed), exact=True)
    assert any(not np.allclose(a[1], b[1], atol=1e-7, rtol=1e-6)
               for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(changed)))
    reordered, _ = adapt(tokens[:, ::-1], mask[:, ::-1])
    _equal_trees(original, reordered)
    empty, trace = adapt(jnp.full_like(tokens, jnp.nan), jnp.zeros_like(mask))
    expected = jax.tree.map(lambda value: jnp.broadcast_to(value, (2,) + value.shape),
                            initial_fast_state(params['adapter']))
    _equal_trees(empty, expected, exact=True)
    for value in trace.values():
        np.testing.assert_array_equal(value, jnp.zeros_like(value))


def test_query_loss_has_finite_full_gradients_and_ignores_padding():
    cfg, batch = _config(), _batch()
    params = bc.init_model(jax.random.key(6), cfg, support_count=2)
    objective = jax.jit(jax.value_and_grad(lambda params, batch:
        bc.loss(params, batch, cfg, jax.random.key(7), training=False), has_aux=True))
    (value, metrics), grads = objective(params, batch)
    assert np.isfinite(value) and all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(grads))
    assert np.isfinite(np.asarray(list(metrics.values()))).all()
    assert float(metrics['write_loss_after']) < float(metrics['write_loss_before'])
    assert sum(float(jnp.sum(jnp.abs(value))) for value in jax.tree.leaves(grads['adapter']['inner_lr_raw'])) > 0
    padded = dict(batch,
        support_tokens=jnp.where(batch['support_mask'][..., None], batch['support_tokens'], jnp.nan),
        query_tokens=jnp.where(batch['query_mask'][..., None], batch['query_tokens'], jnp.nan))
    (padded_value, _), padded_grads = objective(params, padded)
    np.testing.assert_array_equal(value, padded_value)
    _equal_trees(grads, padded_grads, exact=True)


def test_cached_read_matches_causal_likelihood_and_generation_freezes_fast_state():
    cfg, batch = _config(), _batch()
    params = bc.init_model(jax.random.key(8), cfg, support_count=2)
    state, _ = bc.adapt_support(params, batch['support_tokens'], batch['support_mask'], cfg)
    target = batch['query_tokens']
    distribution = bc.autoregressive_distribution(params, state, target, cfg)
    changed = bc.autoregressive_distribution(params, state, target.at[:, 2:].set(9.), cfg)
    for name in distribution:
        np.testing.assert_array_equal(distribution[name][:, :3], changed[name][:, :3])
    shape = (2, cfg.num_heads, cfg.max_steps, cfg.hidden_dim // cfg.num_heads)
    cache = tuple((jnp.zeros(shape), jnp.zeros(shape)) for _ in range(cfg.decoder_layers))
    previous = jnp.zeros((2, 1, 4))
    for position in range(cfg.max_steps):
        raw, cache, _ = bc._raw_from_history(params, state, previous, cfg,
            cache=cache, position=jnp.asarray(position))
        actual = _distribution(raw[:, 0], cfg)
        for name in distribution:
            np.testing.assert_allclose(actual[name], distribution[name][:, position], atol=1e-6, rtol=1e-5)
        previous = target[:, position:position + 1]
    key = jax.random.key(9)
    generated = jax.jit(lambda: bc.generate(params, batch['support_tokens'], batch['support_mask'], cfg, key))()
    frozen = jax.jit(lambda: bc.generate_from_fast_state(params, state, cfg, key))()
    _equal_trees(generated, frozen)
    assert generated['tokens'].shape == (2, cfg.max_steps, 4)
    assert np.isfinite(generated['raw_actions']).all()
    assert 'support_tokens' not in inspect.signature(bc.generate_from_fast_state).parameters
    assert 'support_tokens' not in inspect.signature(bc.autoregressive_distribution).parameters
