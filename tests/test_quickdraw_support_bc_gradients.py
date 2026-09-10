"""Independent gradient and masking checks for pooled support-BC WRITE."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from icil_jax_rlbench.models import fast_weight_ttt as generic
from icil_jax_rlbench.quickdraw import support_bc_models as bc
from icil_jax_rlbench.quickdraw.supervised_models import _distribution
from test_quickdraw_kvb_gradients import (
    _assert_tree_close, _batch, _largest_derivative, _leaf, _norm, _replace_leaf,
)


@pytest.fixture(scope='module')
def setup():
    cfg = bc.SupportBCModelConfig(hidden_dim=8, num_heads=2, context_layers=1,
        decoder_layers=1, mlp_ratio=2, max_steps=4, mixture_components=2,
        dropout=0., fast_dim=4, fast_hidden_dim=8, inner_steps=3,
        inner_lr_init=.4, fast_grad_clip_norm=0., fast_update_clip_norm=0.)
    params = bc.init_model(jax.random.key(43), cfg, support_count=2)
    # Make the inner update signal resolvable by float32 central differences.
    params = _replace_leaf(params, ('backbone', 'output_head', 'kernel'),
                           params['backbone']['output_head']['kernel'] * 5.)
    return cfg, _batch(), params


def _independent_nll(dist, target, points, events, cfg):
    standardized = (target[..., None, :2] - dist['xy_mean']) / jnp.exp(dist['xy_log_scale'])
    components = (-.5 * jnp.sum(standardized ** 2, axis=-1)
                  - jnp.sum(dist['xy_log_scale'], axis=-1) - jnp.log(2 * jnp.pi))
    coordinate = -jax.scipy.special.logsumexp(
        jax.nn.log_softmax(dist['mixture_logits']) + components, axis=-1)
    pen = jnp.logaddexp(0., dist['pen_logit']) - target[..., 2] * dist['pen_logit']
    stop = jnp.logaddexp(0., dist['stop_logit']) - target[..., 3] * dist['stop_logit']

    def mean(value, mask):
        return jnp.sum(jnp.where(mask, value, 0.)) / jnp.maximum(jnp.sum(mask), 1)

    return mean(coordinate, points) + cfg.pen_loss_weight * mean(pen, points) + cfg.stop_loss_weight * mean(stop, events)


def _objective(params, batch, cfg):
    return bc.loss(params, batch, cfg, jax.random.key(19), training=False)[0]


def test_three_updates_match_manual_pooled_support_likelihood(setup):
    cfg, batch, params = setup
    encoded = bc.encode_support(params, batch['support_tokens'], batch['support_mask'], cfg)
    assert encoded['hidden'].shape == (2, 8, cfg.hidden_dim)
    np.testing.assert_array_equal(encoded['event_mask'], batch['support_mask'].reshape(2, -1))
    np.testing.assert_array_equal(jnp.sum(encoded['event_mask'] & ~encoded['point_mask'], axis=1), [2, 2])
    assert int(encoded['point_mask'][0].sum()) == 5  # Unequal demo lengths exercise pooling.
    rates = generic.inner_learning_rates(params['adapter'], cfg.fast_config())
    states, average_losses = [], []
    for task in range(2):
        evidence = jax.tree.map(lambda value: value[task], encoded)

        def write(state):
            raw, _ = bc._read_encoded(params, state, evidence['hidden'], cfg)
            return _independent_nll(_distribution(raw, cfg), evidence['target'],
                                   evidence['point_mask'], evidence['event_mask'], cfg)

        state = generic.initial_fast_state(params['adapter'])
        np.testing.assert_allclose(write(state), bc.support_write_loss(params, state, evidence, cfg),
                                   rtol=2e-6, atol=1e-6)
        losses = []
        for _ in range(3):
            value, grad = jax.value_and_grad(write)(state)
            state = jax.tree.map(lambda value, gradient, rate: value - rate * gradient,
                                 state, grad, rates)
            losses.append(value)
        states.append(state)
        average_losses.append(jnp.mean(jnp.stack(losses)))
    expected = jax.tree.map(lambda *values: jnp.stack(values), *states)
    eager, trace = bc.adapt_support(params, batch['support_tokens'], batch['support_mask'], cfg)
    compiled, compiled_trace = jax.jit(lambda tokens, mask: bc.adapt_support(
        params, tokens, mask, cfg))(batch['support_tokens'], batch['support_mask'])
    _assert_tree_close(eager, expected)
    _assert_tree_close(compiled, expected)
    _assert_tree_close(compiled_trace, trace)
    np.testing.assert_allclose(trace['write_loss'], average_losses, rtol=2e-5, atol=2e-6)
    once, _ = bc.adapt_support(params, batch['support_tokens'], batch['support_mask'],
                              replace(cfg, inner_steps=1))
    assert _norm(jax.tree.map(jnp.subtract, eager, once)) > 1e-4


def test_second_order_and_fomaml_distinguish_support_evidence_gradient_paths(setup):
    cfg, batch, params = setup

    @jax.jit
    def evidence_gradient(tokens):
        def objective(evidence, first_order):
            return _objective(params, {**batch, 'support_tokens': evidence},
                              replace(cfg, first_order=first_order))
        return (jax.value_and_grad(lambda value: objective(value, False))(tokens),
                jax.value_and_grad(lambda value: objective(value, True))(tokens))

    (full_value, full_evidence), (first_value, first_evidence) = evidence_gradient(batch['support_tokens'])
    np.testing.assert_array_equal(full_value, first_value)
    assert _norm(full_evidence) > 1e-5
    np.testing.assert_array_equal(first_evidence, jnp.zeros_like(first_evidence))
    assert not np.any(np.asarray(full_evidence)[~np.asarray(batch['support_mask'])])
    grad_fn = jax.jit(jax.grad(_objective), static_argnums=2)
    full, first = grad_fn(params, batch, cfg), grad_fn(params, batch, replace(cfg, first_order=True))
    # The shared decoder keeps direct query gradients in both methods; it is not
    # a WRITE-only encoder whose entire FOMAML gradient should disappear.
    for path in (('backbone', 'query_input'), ('backbone', 'decoder_0')):
        assert _norm(_leaf(full, path)) > 1e-6
        assert _norm(_leaf(first, path)) > 1e-6
        assert _norm(jax.tree.map(jnp.subtract, _leaf(full, path), _leaf(first, path))) > 1e-6
    for name in ('fast_init', 'inner_lr_raw'):
        assert _norm(full['adapter'][name]) > 1e-6
        assert _norm(first['adapter'][name]) > 1e-7
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(full))
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(first))


def test_full_three_step_outer_derivatives_match_finite_differences(setup):
    cfg, batch, params = setup
    objective = jax.jit(lambda parameters: _objective(parameters, batch, cfg))
    gradient = jax.jit(jax.grad(objective))(params)
    for group in (('adapter', 'fast_init'), ('adapter', 'inner_lr_raw'),
                  ('backbone', 'query_input'), ('backbone', 'decoder_0')):
        magnitude, path, index, analytic = _largest_derivative(_leaf(gradient, group), group)
        assert magnitude > 1e-6, group
        original = _leaf(params, path)
        # BOS input biases have appreciable curvature on the .005 scale.
        epsilon = .0003
        positive = _replace_leaf(params, path, original.at[index].add(epsilon))
        negative = _replace_leaf(params, path, original.at[index].add(-epsilon))
        numerical = (float(objective(positive)) - float(objective(negative))) / (2 * epsilon)
        np.testing.assert_allclose(analytic, numerical, rtol=.01, atol=3e-5,
                                   err_msg=f'Outer derivative through three support BC updates: {path}{index}')


def test_outer_objective_contains_query_nll_only(setup):
    cfg, batch, params = setup
    state, trace = bc.adapt_support(params, batch['support_tokens'], batch['support_mask'], cfg)
    distribution = bc.autoregressive_distribution(params, state, batch['query_tokens'], cfg)
    expected = []
    for task in range(2):
        expected.append(_independent_nll(
            jax.tree.map(lambda value: value[task], distribution), batch['query_tokens'][task],
            batch['query_point_mask'][task], batch['query_mask'][task], cfg))
    np.testing.assert_allclose(_objective(params, batch, cfg), jnp.mean(jnp.stack(expected)),
                               rtol=2e-6, atol=1e-6)
    assert np.asarray(trace['write_loss']).min() > 0.
    empty_query = {**batch, 'query_mask': jnp.zeros_like(batch['query_mask']),
                   'query_point_mask': jnp.zeros_like(batch['query_point_mask'])}
    value, gradient = jax.jit(jax.value_and_grad(_objective), static_argnums=2)(params, empty_query, cfg)
    assert float(value) == 0.
    assert all(np.isfinite(value).all() and not np.any(value) for value in jax.tree.leaves(gradient))


def test_task_reset_padding_and_empty_support_are_exact_noops(setup):
    cfg, batch, params = setup
    adapt = jax.jit(lambda tokens, mask: bc.adapt_support(params, tokens, mask, cfg))
    baseline, _ = adapt(batch['support_tokens'], batch['support_mask'])
    repeat, _ = adapt(batch['support_tokens'], batch['support_mask'])
    padded, _ = adapt(jnp.where(batch['support_mask'][..., None], batch['support_tokens'], jnp.nan),
                      batch['support_mask'])
    _assert_tree_close(baseline, repeat, exact=True)
    _assert_tree_close(baseline, padded, exact=True)
    for task in range(2):
        single, _ = adapt(batch['support_tokens'][task:task + 1], batch['support_mask'][task:task + 1])
        _assert_tree_close(jax.tree.map(lambda value: value[task:task + 1], baseline), single)
    changed, _ = adapt(batch['support_tokens'].at[0, :, :, :2].multiply(-1.2), batch['support_mask'])
    _assert_tree_close(jax.tree.map(lambda value: value[1], baseline),
                       jax.tree.map(lambda value: value[1], changed), exact=True)
    assert _norm(jax.tree.map(lambda a, b: a[0] - b[0], baseline, changed)) > 1e-4
    drift_cfg = replace(cfg, fast_drift_weight=.5)
    empty, trace = jax.jit(lambda tokens, mask: bc.adapt_support(params, tokens, mask, drift_cfg))(
        jnp.full_like(batch['support_tokens'], jnp.nan), jnp.zeros_like(batch['support_mask']))
    initial = jax.tree.map(lambda value: jnp.broadcast_to(value, (2,) + value.shape),
                          generic.initial_fast_state(params['adapter']))
    _assert_tree_close(empty, initial, exact=True)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(trace))
    assert not np.any(np.asarray(trace['fast_delta_norm']))
    assert not np.any(np.asarray(trace['write_loss']))
