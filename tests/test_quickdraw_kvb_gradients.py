"""Independent tiny-fixture checks of the sketch KVB meta-gradient graph."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from icil_jax_rlbench.models import fast_weight_ttt as generic
from icil_jax_rlbench.quickdraw import kvb_models as kvb


def _config(**changes):
    return kvb.KVBModelConfig(hidden_dim=8, num_heads=2, context_layers=1,
        decoder_layers=1, mlp_ratio=2, max_steps=4, mixture_components=2,
        dropout=0., dtype='float32', fast_dim=4, fast_hidden_dim=8,
        inner_steps=3, inner_lr_init=.2, fast_grad_clip_norm=0.,
        fast_update_clip_norm=0., **changes)


def _batch():
    first = jnp.asarray([
        [[-.6, -.2, 0, 0], [-.1, .3, 1, 0], [.7, -.4, 0, 0], [0, 0, 0, 1]],
        [[.5, .6, 0, 0], [-.2, .1, 1, 0], [-.6, -.5, 1, 0], [0, 0, 0, 1]],
    ], jnp.float32)
    second = first.at[..., :2].multiply(jnp.asarray([-.8, .7]))
    second = second.at[:, 2, :].set(jnp.asarray([0., 0., 0., 1.])).at[:, 3, :].set(0.)
    supports = jnp.stack((first, second), axis=1)
    support_mask = jnp.broadcast_to(jnp.asarray([[True] * 4, [True, True, True, False]]), (2, 2, 4))
    query = jnp.asarray([
        [[-.4, -.6, 0, 0], [.2, .4, 1, 0], [.6, -.1, 1, 0], [0, 0, 0, 1]],
        [[.6, .4, 0, 0], [-.4, -.3, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
    ], jnp.float32)
    events = jnp.asarray([[True] * 4, [True, True, True, False]])
    return {'support_tokens': supports, 'support_mask': support_mask,
            'query_tokens': query, 'query_mask': events,
            'query_point_mask': events & (query[..., 3] < .5),
            'example_mask': jnp.ones(2, bool)}


@pytest.fixture(scope='module')
def setup():
    config, batch = _config(), _batch()
    params = kvb.init_model(jax.random.key(43), config, support_count=2)
    return config, batch, params


def _norm(tree):
    return float(np.sqrt(sum(np.sum(np.asarray(value) ** 2) for value in jax.tree.leaves(tree))))


def _assert_tree_close(actual, expected, *, exact=False):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        if exact:
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-6)


def _objective(params, batch, config):
    return kvb.loss(params, batch, config, jax.random.key(19), training=False)[0]


def _replace_leaf(tree, path, value):
    if not path:
        return value
    result = dict(tree)
    result[path[0]] = _replace_leaf(tree[path[0]], path[1:], value)
    return result


def _largest_derivative(tree, prefix=()):
    candidates = []
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(value)
        index = np.unravel_index(int(np.argmax(np.abs(array))), array.shape)
        candidates.append((abs(float(array[index])),
                           prefix + tuple(entry.key for entry in path), index, float(array[index])))
    return max(candidates, key=lambda item: item[0])


def _leaf(tree, path):
    for name in path:
        tree = tree[name]
    return tree


def test_three_steps_reuse_the_same_complete_pooled_support_batch(setup):
    config, batch, params = setup
    encoded, masks = kvb.encode_support(params, batch['support_tokens'], batch['support_mask'], config)
    assert encoded.shape == (2, 2 * 4, config.hidden_dim)
    np.testing.assert_array_equal(masks, batch['support_mask'].reshape(2, -1))
    adapter, fast_config = params['adapter'], config.fast_config()
    rates = generic.inner_learning_rates(adapter, fast_config)
    reference, reference_losses = [], []
    for task in range(2):
        keys, values = generic.project_key_value(adapter, encoded[task])

        def write(state):
            residual = generic.fast_model_apply(state, keys, fast_config) - values
            squared_per_point = jnp.sum(residual ** 2, axis=-1)
            return jnp.sum(jnp.where(masks[task], squared_per_point, 0.)) / (
                config.fast_dim * jnp.maximum(jnp.sum(masks[task]), 1))

        state = generic.initial_fast_state(adapter)
        losses = []
        for _ in range(3):
            value, gradient = jax.value_and_grad(write)(state)
            # Independent reference: no generic adaptation/update helper.
            state = jax.tree.map(lambda value, grad, rate: value - rate * grad,
                                 state, gradient, rates)
            losses.append(value)
        reference.append(state)
        reference_losses.append(jnp.mean(jnp.stack(losses)))
    expected = jax.tree.map(lambda *values: jnp.stack(values), *reference)
    eager, trace = kvb.adapt_support(params, batch['support_tokens'], batch['support_mask'], config)
    compiled, compiled_trace = jax.jit(lambda tokens, mask: kvb.adapt_support(
        params, tokens, mask, config))(batch['support_tokens'], batch['support_mask'])
    _assert_tree_close(eager, expected)
    _assert_tree_close(compiled, expected)
    _assert_tree_close(compiled_trace, trace)
    np.testing.assert_allclose(trace['write_loss'], reference_losses, rtol=2e-5, atol=2e-6)
    # This fixture distinguishes three updates from silently taking just one.
    once, _ = kvb.adapt_support(params, batch['support_tokens'], batch['support_mask'],
                               replace(config, inner_steps=1))
    assert any(not np.allclose(one, three, atol=1e-6)
               for one, three in zip(jax.tree.leaves(once), jax.tree.leaves(eager)))


def test_full_and_first_order_have_the_expected_write_only_gradient_paths(setup):
    config, batch, params = setup
    objective = jax.jit(jax.value_and_grad(_objective), static_argnums=2)
    full_value, full = objective(params, batch, config)
    first_value, first = objective(params, batch, replace(config, first_order=True))
    np.testing.assert_allclose(full_value, first_value, rtol=1e-6, atol=1e-7)
    context_names = [name for name in params['backbone'] if name.startswith('context')]
    assert context_names
    for name in context_names:
        assert _norm(full['backbone'][name]) > 1e-9, name
        assert _norm(first['backbone'][name]) == 0., name
    for name in ('key_projection', 'value_projection'):
        assert _norm(full['adapter'][name]) > 1e-9, name
        assert _norm(first['adapter'][name]) == 0., name
    for name in ('fast_init', 'inner_lr_raw', 'query_projection', 'read_projection'):
        assert _norm(full['adapter'][name]) > 1e-9, name
    # Stopping inner gradients does not sever direct learned-rate or READ paths.
    assert _norm(first['adapter']['inner_lr_raw']) > 1e-9
    assert _norm(first['adapter']['query_projection']) > 1e-9
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(full))
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(first))


def test_three_step_second_order_derivatives_match_finite_differences(setup):
    config, batch, params = setup
    objective = jax.jit(lambda parameters: _objective(parameters, batch, config))
    value, gradient = jax.jit(jax.value_and_grad(objective))(params)
    np.testing.assert_allclose(value, _objective(params, batch, config), rtol=2e-6, atol=1e-6)
    groups = [('adapter', name) for name in
              ('key_projection', 'value_projection', 'fast_init', 'inner_lr_raw', 'read_projection')]
    groups.append(('backbone', 'context_input'))
    for group in groups:
        magnitude, path, index, analytic = _largest_derivative(_leaf(gradient, group), group)
        assert magnitude > 1e-7, group
        original = _leaf(params, path)
        epsilon = .01
        positive = _replace_leaf(params, path, original.at[index].add(epsilon))
        negative = _replace_leaf(params, path, original.at[index].add(-epsilon))
        numerical = (float(objective(positive)) - float(objective(negative))) / (2 * epsilon)
        np.testing.assert_allclose(analytic, numerical, rtol=.04, atol=1e-5,
                                   err_msg=f'Outer derivative through three updates: {path}{index}')


def test_three_update_outer_loss_has_no_support_write_penalty(setup):
    config, batch, params = setup
    fast_state, trace = kvb.adapt_support(params, batch['support_tokens'], batch['support_mask'],
                                         config, jax.random.key(19), training=False)
    distribution = kvb.autoregressive_distribution(params, fast_state, batch['query_tokens'], config)
    target = batch['query_tokens']
    # Compute the declared query density independently of the model's loss helper.
    standardized = (target[..., None, :2] - distribution['xy_mean']) / jnp.exp(distribution['xy_log_scale'])
    components = (-.5 * jnp.sum(standardized ** 2, axis=-1)
                  - jnp.sum(distribution['xy_log_scale'], axis=-1) - jnp.log(2 * jnp.pi))
    coordinate_nll = -jax.scipy.special.logsumexp(
        jax.nn.log_softmax(distribution['mixture_logits']) + components, axis=-1)
    pen_bce = jnp.logaddexp(0., distribution['pen_logit']) - target[..., 2] * distribution['pen_logit']
    stop_bce = jnp.logaddexp(0., distribution['stop_logit']) - target[..., 3] * distribution['stop_logit']
    expected = []
    for task in range(2):
        points, events = batch['query_point_mask'][task], batch['query_mask'][task]
        expected.append(jnp.mean(coordinate_nll[task][points])
                        + config.pen_loss_weight * jnp.mean(pen_bce[task][points])
                        + config.stop_loss_weight * jnp.mean(stop_bce[task][events]))
    np.testing.assert_allclose(_objective(params, batch, config), jnp.mean(jnp.stack(expected)),
                               rtol=2e-6, atol=1e-6)
    # Even a nonempty, reconstructable support must not provide an outer loss
    # when every query event has been removed from supervision.
    empty_query = {**batch, 'query_mask': jnp.zeros_like(batch['query_mask']),
                   'query_point_mask': jnp.zeros_like(batch['query_point_mask'])}
    value, gradient = jax.jit(jax.value_and_grad(_objective), static_argnums=2)(params, empty_query, config)
    assert float(value) == 0.
    assert all(np.isfinite(value).all() and not np.any(value) for value in jax.tree.leaves(gradient))
    initial = generic.initial_fast_state(params['adapter'])
    assert any(np.any(np.asarray(value) != np.asarray(origin)[None])
               for value, origin in zip(jax.tree.leaves(fast_state), jax.tree.leaves(initial)))
    assert np.asarray(trace['write_loss']).max() > 0.


def test_task_resets_isolation_and_padded_support_are_exact_noops(setup):
    config, batch, params = setup
    adapt = jax.jit(lambda tokens, mask: kvb.adapt_support(
        params, tokens, mask, config, jax.random.key(3), training=False))
    full_state, _ = adapt(batch['support_tokens'], batch['support_mask'])
    repeated, _ = adapt(batch['support_tokens'], batch['support_mask'])
    _assert_tree_close(full_state, repeated, exact=True)
    for task in range(2):
        single, _ = adapt(batch['support_tokens'][task:task + 1], batch['support_mask'][task:task + 1])
        _assert_tree_close(jax.tree.map(lambda value: value[task:task + 1], full_state), single)
    changed_tokens = batch['support_tokens'].at[0, :, :, :2].multiply(-1.2)
    changed, _ = adapt(changed_tokens, batch['support_mask'])
    _assert_tree_close(jax.tree.map(lambda value: value[1], full_state),
                       jax.tree.map(lambda value: value[1], changed), exact=True)
    assert any(not np.allclose(left[0], right[0], atol=1e-7)
               for left, right in zip(jax.tree.leaves(full_state), jax.tree.leaves(changed)))
    # Drift must not cause updates on a task whose entire support is padding.
    empty_config = replace(config, fast_drift_weight=.5)
    empty_state, trace = jax.jit(lambda tokens, mask: kvb.adapt_support(
        params, tokens, mask, empty_config, jax.random.key(5), training=False))(
            jnp.full_like(batch['support_tokens'], jnp.nan), jnp.zeros_like(batch['support_mask']))
    initial = jax.tree.map(lambda value: jnp.broadcast_to(value, (2,) + value.shape),
                          generic.initial_fast_state(params['adapter']))
    _assert_tree_close(empty_state, initial, exact=True)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(trace))
    assert not np.any(np.asarray(trace['fast_delta_norm']))
    origin_distribution = kvb.autoregressive_distribution(params, initial, batch['query_tokens'], config)
    no_read_distribution = kvb.autoregressive_distribution(
        params, initial, batch['query_tokens'], replace(config, read_scale=0.))
    _assert_tree_close(origin_distribution, no_read_distribution, exact=True)
    adapted_distribution = kvb.autoregressive_distribution(params, full_state, batch['query_tokens'], config)
    assert not np.allclose(origin_distribution['xy_mean'], adapted_distribution['xy_mean'], atol=1e-7)
