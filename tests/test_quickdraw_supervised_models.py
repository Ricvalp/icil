from dataclasses import replace
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw.supervised_models import (
    SupervisedModelConfig, _Policy, _distribution, autoregressive_distribution,
    diffusion_denoise, diffusion_schedule, diffusion_targets, generate, init_model, loss,
)


def _config(architecture='autoregressive', **kwargs):
    return SupervisedModelConfig(architecture=architecture, hidden_dim=12, num_heads=3,
                                 context_layers=1, decoder_layers=1, mlp_ratio=2,
                                 max_steps=5, mixture_components=2, diffusion_steps=4,
                                 dropout=0.0, dtype='float32', **kwargs)


def _batch():
    drawings = jnp.asarray([
        [[-.3, -.1, 0, 0], [.1, .2, 1, 0], [.2, .3, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
        [[.3, .1, 0, 0], [-.1, -.2, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
    ], dtype=jnp.float32)
    mask = jnp.arange(5)[None] < jnp.asarray([4, 3])[:, None]
    return {'support_tokens': jnp.stack((drawings, drawings.at[..., :2].multiply(-1)), axis=1),
            'support_mask': jnp.stack((mask, mask), axis=1),
            'query_tokens': drawings, 'query_mask': mask,
            'query_point_mask': mask & (drawings[..., 3] < .5),
            'example_mask': jnp.asarray([True, True])}


@pytest.mark.parametrize('architecture', ['autoregressive', 'diffusion'])
def test_supervised_models_have_finite_gradients_and_improve_fixed_batch(architecture):
    cfg, batch = _config(architecture), _batch()
    params = init_model(jax.random.key(5), cfg, support_count=2)
    objective = jax.jit(jax.value_and_grad(lambda p: loss(p, batch, cfg, jax.random.key(7), training=False)[0]))
    before, gradient = objective(params)
    assert np.isfinite(before)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(gradient))
    assert float(jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(gradient)))) > 0
    updated = jax.tree.map(lambda p, g: p - 1e-4 * g, params, gradient)
    after, _ = objective(updated)
    assert float(after) < float(before)
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(params))
    assert 'fast' not in repr(jax.tree.structure(params)).lower()


def test_autoregressive_future_causality_and_cached_decoder_match():
    cfg, batch = _config(), _batch()
    params = init_model(jax.random.key(2), cfg, support_count=2)
    targets = batch['query_tokens']
    changed = targets.at[:, 2:, :].set(19.0)
    first = autoregressive_distribution(params, batch['support_tokens'], batch['support_mask'], targets, cfg)
    second = autoregressive_distribution(params, batch['support_tokens'], batch['support_mask'], changed, cfg)
    for name in first:
        np.testing.assert_array_equal(first[name][:, :3], second[name][:, :3])
    assert not np.allclose(first['xy_mean'][:, 3:], second['xy_mean'][:, 3:])

    policy, variables = _Policy(cfg), {'params': params}
    memory, memory_mask = policy.apply(variables, batch['support_tokens'], batch['support_mask'], method=policy.encode)
    cross_cache = policy.apply(variables, memory, method=policy.project_memory)
    shape = (2, cfg.num_heads, cfg.max_steps, cfg.hidden_dim // cfg.num_heads)
    cache = tuple((jnp.zeros(shape), jnp.zeros(shape)) for _ in range(cfg.decoder_layers))
    previous = jnp.zeros((2, 1, 4))
    for i in range(cfg.max_steps):
        raw, cache = policy.apply(variables, previous, memory, memory_mask, cache=cache,
                                   cross_cache=cross_cache, position=jnp.asarray(i), method=policy.decode)
        actual = _distribution(raw[:, 0], cfg)
        for name in first:
            np.testing.assert_allclose(actual[name], first[name][:, i], atol=1e-6, rtol=1e-5)
        previous = targets[:, i:i + 1]


@pytest.mark.parametrize('architecture', ['autoregressive', 'diffusion'])
def test_context_is_used_but_padding_and_demonstration_order_are_not(architecture):
    cfg, batch = _config(architecture), _batch()
    params = init_model(jax.random.key(2), cfg, support_count=2)

    def predict(tokens, mask):
        if architecture == 'autoregressive':
            return autoregressive_distribution(params, tokens, mask, batch['query_tokens'], cfg)['xy_mean']
        return diffusion_denoise(params, tokens, mask, batch['query_tokens'], jnp.asarray([1, 2]), cfg)

    tokens, mask = batch['support_tokens'], batch['support_mask']
    baseline = predict(tokens, mask)
    altered_padding = jnp.where(mask[..., None], tokens, jnp.nan)
    np.testing.assert_array_equal(baseline, predict(altered_padding, mask))
    np.testing.assert_allclose(baseline, predict(tokens[:, ::-1], mask[:, ::-1]), atol=1e-6, rtol=1e-5)
    changed_context = predict(tokens.at[..., :2].add(.7), mask)
    assert not np.allclose(baseline, changed_context, atol=1e-6)
    empty = predict(jnp.full_like(tokens, jnp.nan), jnp.zeros_like(mask))
    assert np.isfinite(empty).all()


@pytest.mark.parametrize('architecture', ['autoregressive', 'diffusion'])
def test_generation_is_support_only_keyed_and_retains_output_failures(architecture):
    cfg, batch = _config(architecture), _batch()
    params = init_model(jax.random.key(2), cfg, support_count=2)
    sample = jax.jit(lambda key: generate(params, batch['support_tokens'], batch['support_mask'], cfg, key))
    output = sample(jax.random.key(3))
    repeated = sample(jax.random.key(3))
    other = sample(jax.random.key(4))
    assert output['tokens'].shape == (2, cfg.max_steps, 4)
    assert output['event_mask'].shape == (2, cfg.max_steps)
    assert np.isfinite(output['raw_actions']).all()
    assert not np.array_equal(output['raw_actions'], other['raw_actions'])
    for name in output:
        np.testing.assert_array_equal(output[name], repeated[name])
    for row in range(2):
        stops = np.flatnonzero(np.asarray(output['tokens'][row, :, 3]) >= .5)
        assert int(output['length'][row]) == (int(stops[0]) if len(stops) else cfg.max_steps)
        assert bool(output['stopped'][row]) == bool(len(stops))
    assert set(inspect.signature(generate).parameters) == {
        'params', 'support_tokens', 'support_mask', 'cfg', 'key', 'deterministic'}


def test_diffusion_full_horizon_labels_and_timestep_conditioning():
    cfg, batch = _config('diffusion'), _batch()
    params = init_model(jax.random.key(2), cfg, support_count=2)
    target = diffusion_targets(batch['query_tokens'], batch['query_mask'])
    np.testing.assert_array_equal(target[1, 2:], jnp.asarray([[0, 0, -1, 1]] * 3))
    dirty_padding = jnp.where(batch['query_mask'][..., None], batch['query_tokens'], 99)
    np.testing.assert_array_equal(target, diffusion_targets(dirty_padding, batch['query_mask']))
    first = diffusion_denoise(params, batch['support_tokens'], batch['support_mask'], target, jnp.zeros((2,), jnp.int32), cfg)
    last = diffusion_denoise(params, batch['support_tokens'], batch['support_mask'], target, jnp.full((2,), 3), cfg)
    assert not np.allclose(first, last)
    assert set(inspect.signature(diffusion_denoise).parameters) == {
        'params', 'support_tokens', 'support_mask', 'noisy_actions', 'timesteps', 'cfg', 'key', 'training'}
    schedule = diffusion_schedule(cfg)
    assert float(schedule['posterior_variance'][0]) == 0
    assert np.all(np.diff(np.asarray(schedule['alpha_bar'])) < 0)


@pytest.mark.parametrize('architecture', ['autoregressive', 'diffusion'])
def test_padding_examples_have_zero_objective_gradient(architecture):
    cfg, batch = _config(architecture), _batch()
    params = init_model(jax.random.key(9), cfg, support_count=2)
    empty = dict(batch, example_mask=jnp.zeros((2,), dtype=bool))
    (value, metrics), gradient = jax.value_and_grad(loss, has_aux=True)(
        params, empty, cfg, jax.random.key(7), training=False)
    assert float(value) == 0.0
    assert all(float(metric) == 0.0 for metric in metrics.values())
    assert all(np.all(np.asarray(leaf) == 0) for leaf in jax.tree.leaves(gradient))


def test_bfloat16_activations_keep_float32_parameters_and_explicit_dropout_rng():
    cfg = replace(_config(), dtype='bfloat16', dropout=.2)
    batch = _batch()
    params = init_model(jax.random.key(2), cfg, support_count=2)
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(params))
    one = loss(params, batch, cfg, jax.random.key(1))[0]
    repeated = loss(params, batch, cfg, jax.random.key(1))[0]
    other = loss(params, batch, cfg, jax.random.key(3))[0]
    np.testing.assert_array_equal(one, repeated)
    assert np.isfinite(one)
    assert float(one) != float(other)
