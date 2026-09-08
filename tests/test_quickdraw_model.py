from __future__ import annotations

from dataclasses import replace
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig, TTTAdaptConfig, adapt_encoded_support,
    fast_read_residual, init_fast_adapter_params, initial_fast_state,
    tree_difference_norm, tree_l2_norm,
)
from icil_jax_rlbench.quickdraw.model import (
    MODEL_TYPES, SketchModelConfig, adapt_sketch, encode_context, first_write_diagnostics, generate,
    initial_query_carry, init_sketch_params, meta_objective, query_loss,
    query_step, sample_token, teacher_forced_distribution,
)


def _section(offset=0., demos=1, pad=0):
    tokens = np.asarray([[-.8, .2, 0., 0.], [-.2, .7, 1., 0.],
                         [.6, -.1, 1., 0.], [0., 0., 0., 1.]], np.float32)
    tokens[:3, :2] += offset
    tokens = np.broadcast_to(tokens[None], (demos, 4, 4)).copy()
    return {
        'tokens': jnp.asarray(np.pad(tokens, ((0, 0), (0, pad), (0, 0)))),
        'event_mask': jnp.asarray(np.tile([True] * 4 + [False] * pad, (demos, 1))),
        'point_mask': jnp.asarray(np.tile([True] * 3 + [False] * (pad + 1), (demos, 1))),
        'frame': jnp.tile(jnp.asarray([[.1, -.1, .2, .9]], jnp.float32), (demos, 1)),
        'state': jnp.zeros((demos, 4 + pad, 3), jnp.float32),
    }


def _fixture(model_type='ttt_kvb_full', **kwargs):
    cfg = SketchModelConfig(model_type=model_type, hidden_dim=6, fast_dim=3,
                            fast_hidden_dim=4, mixture_components=2,
                            max_steps=8, segment_size=3, inner_lr_init=.12, **kwargs)
    params = init_sketch_params(jax.random.key(31), cfg)
    batch = {'support': _section(), 'query': _section(.15)}
    batch = jax.tree_util.tree_map(lambda x: x[None], batch)
    return cfg, params, batch


def _task(section):
    return jax.tree_util.tree_map(lambda x: x[0], section)


def _assert_tree_equal(a, b, *, exact=True):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        if exact:
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, atol=1e-6, rtol=2e-5)


def test_full_and_fomaml_share_updates_but_differ_in_write_meta_gradients():
    cfg, params, batch = _fixture()
    first = replace(cfg, model_type='ttt_kvb_first_order')
    support = _task(batch['support'])
    _assert_tree_equal(adapt_sketch(params, support, cfg)[0],
                       adapt_sketch(params, support, first)[0])
    full_grad = jax.grad(lambda p: meta_objective(p, batch, cfg)[0])(params)
    first_grad = jax.grad(lambda p: meta_objective(p, batch, first)[0])(params)
    for name in ('support_encoder', 'key_projection', 'value_projection'):
        assert float(tree_l2_norm(full_grad[name])) > 1e-8, name
        assert float(tree_l2_norm(first_grad[name])) == 0., name
    for name in ('fast_init', 'inner_lr_raw', 'query_projection', 'read_projection'):
        assert float(tree_l2_norm(full_grad[name])) > 1e-8, name
        assert float(tree_l2_norm(first_grad[name])) > 1e-8, name


def test_meta_gradient_directional_finite_differences_all_paths():
    # Float64 outer arithmetic, fixed smooth inputs, no clipping thresholds.
    # Shared normalization intentionally retains float32 for production parity.
    with jax.enable_x64():
        cfg, params, batch = _fixture(fast_grad_clip_norm=0.)
        params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)
        batch = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x, batch)
        objective = jax.jit(lambda p: meta_objective(p, batch, cfg)[0])
        gradient = jax.grad(objective)(params)
        for name in ('support_encoder', 'key_projection', 'value_projection',
                     'fast_init', 'inner_lr_raw', 'query_projection', 'read_projection'):
            group = gradient[name]
            norm = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(group)))
            assert float(norm) > 1e-9, name
            direction = jax.tree_util.tree_map(lambda x: x / norm, group)
            def shifted(amount):
                changed = dict(params)
                changed[name] = jax.tree_util.tree_map(lambda x, d: x + amount * d, params[name], direction)
                return float(objective(changed))
            epsilon = 2e-3
            numerical = (shifted(epsilon) - shifted(-epsilon)) / (2 * epsilon)
            np.testing.assert_allclose(numerical, float(norm), rtol=.025, atol=2e-5, err_msg=name)


@pytest.mark.parametrize('model_type', ['ttt_kvb_full', 'ttt_supervised_write'])
def test_padding_cannot_change_write_grouping_or_drift_updates(model_type):
    cfg, params, _ = _fixture(model_type, fast_drift_weight=.7)
    support = _section(demos=2)
    padded = _section(demos=2, pad=5)
    padded['tokens'] = padded['tokens'].at[:, 4:].set(500.)
    padded['state'] = padded['state'].at[:, 4:].set(-200.)
    normal, _ = adapt_sketch(params, support, cfg)
    extra, trace = adapt_sketch(params, padded, cfg)
    _assert_tree_equal(normal, extra, exact=False)
    # Padding after each drawing creates an inactive third segment.
    np.testing.assert_array_equal(np.asarray(trace['fast_update_norm'])[[2, 5]], 0.)
    empty = {**padded, 'event_mask': jnp.zeros_like(padded['event_mask']),
             'point_mask': jnp.zeros_like(padded['point_mask'])}
    unchanged, _ = adapt_sketch(params, empty, cfg)
    _assert_tree_equal(unchanged, initial_fast_state(params))
    gradient = jax.grad(lambda p: query_loss(p, adapt_sketch(p, empty, cfg)[0], support, cfg)[0])(params)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(gradient))


def test_shared_encoded_writer_empty_segment_is_exact_noop_with_drift():
    cfg = FastWeightTTTConfig(hidden_dim=6, fast_dim=3, fast_hidden_dim=4)
    params = init_fast_adapter_params(jax.random.key(9), cfg)
    adapt_cfg = TTTAdaptConfig(fast_drift_weight=2.)
    registers = jax.random.normal(jax.random.key(2), (2, 3, 6))
    mask = jnp.asarray([[True, True, False], [False, False, False]])
    one, _ = adapt_encoded_support(params, registers[:1], mask[:1], cfg, adapt_cfg)
    two, trace = adapt_encoded_support(params, registers, mask, cfg, adapt_cfg)
    _assert_tree_equal(one, two)
    assert float(trace['fast_update_norm'][-1]) == 0.


def test_causal_teacher_forcing_and_public_generation_boundary():
    cfg, params, batch = _fixture()
    support, query = _task(batch['support']), _task(batch['query'])
    fast, _ = adapt_sketch(params, support, cfg)
    changed = {**query, 'tokens': query['tokens'].at[:, 2:].set(4.),
               'event_mask': jnp.zeros_like(query['event_mask']),
               'point_mask': jnp.zeros_like(query['point_mask'])}
    left = teacher_forced_distribution(params, fast, query, cfg)
    right = teacher_forced_distribution(params, fast, changed, cfg)
    for name in left:
        np.testing.assert_array_equal(left[name][:, :3], right[name][:, :3])
    assert all(name not in inspect.signature(generate).parameters for name in
               ('support', 'target', 'query', 'length', 'point_mask', 'event_mask'))
    key = jax.random.key(27)
    first = generate(params, fast, cfg, key)
    del changed
    _assert_tree_equal(first, generate(params, fast, cfg, key))
    _assert_tree_equal(fast, adapt_sketch(params, support, cfg)[0])
    _assert_tree_equal(initial_fast_state(params), params['fast_init'])


def test_delta_read_initial_noop_and_nonzero_fast_derivative():
    cfg, params, batch = _fixture()
    hidden = jnp.linspace(-.7, .9, cfg.hidden_dim)
    initial = initial_fast_state(params)
    residual = lambda state: fast_read_residual(params, state, hidden, cfg.fast_config(), read_mode='delta')
    np.testing.assert_array_equal(residual(initial), jnp.zeros_like(hidden))
    gradient = jax.grad(lambda state: jnp.sum(residual(state)))(initial)
    assert float(tree_l2_norm(gradient)) > 1e-6
    changed = jax.tree_util.tree_map(lambda w, g: w + .01 * g, initial, gradient)
    assert float(jnp.linalg.norm(residual(changed))) > 1e-6
    supervised = replace(cfg, model_type='ttt_supervised_write')
    state, _ = adapt_sketch(params, _task(batch['support']), supervised)
    assert float(tree_difference_norm(state, initial)) > 1e-8


@pytest.mark.parametrize('model_type', MODEL_TYPES)
def test_all_comparators_train_generate_and_enforce_context_boundary(model_type):
    cfg, params, batch = _fixture(model_type)
    loss, gradient = jax.value_and_grad(lambda p: meta_objective(p, batch, cfg)[0])(params)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(gradient))
    support = _task(batch['support'])
    fast, _ = adapt_sketch(params, support, cfg)
    context = encode_context(params, support, cfg) if model_type == 'explicit_context' else None
    result = generate(params, fast, cfg, jax.random.key(19), context=context)
    assert result['tokens'].shape == (cfg.max_steps, 4)
    assert int(jnp.sum(result['event_mask'])) == int(result['length']) + int(result['stopped'])
    if model_type != 'explicit_context':
        with pytest.raises(ValueError, match='explicit_context'):
            generate(params, fast, cfg, jax.random.key(0), context={'registers': jnp.zeros((1, 6)), 'mask': jnp.ones((1,), bool)})


def test_context_empty_support_is_noop_even_with_nonzero_projection_bias():
    cfg, params, batch = _fixture('explicit_context')
    params['context_projection']['bias'] = jnp.ones((cfg.hidden_dim,))
    support, query = _task(batch['support']), _task(batch['query'])
    support = {**support, 'event_mask': jnp.zeros_like(support['event_mask']),
               'point_mask': jnp.zeros_like(support['point_mask'])}
    context = encode_context(params, support, cfg)
    initial = initial_fast_state(params)
    _assert_tree_equal(teacher_forced_distribution(params, initial, query, cfg),
                       teacher_forced_distribution(params, initial, query, cfg, context))


@pytest.mark.parametrize('model_type', ['ttt_kvb_full', 'explicit_context'])
def test_public_support_knots_are_evidence_but_query_knots_never_enter_read(model_type):
    cfg, params, batch = _fixture(model_type, experiment='b2')
    support, query = _task(batch['support']), _task(batch['query'])
    explicit = {**support, 'knot_mask': support['point_mask']}
    changed = {**explicit, 'knot_mask': explicit['knot_mask'].at[:, 1].set(False)}
    if model_type == 'explicit_context':
        expected = encode_context(params, support, cfg)
        normal = encode_context(params, explicit, cfg)
        different = encode_context(params, changed, cfg)
        assert not np.allclose(normal['registers'], different['registers'])
        fast = initial_fast_state(params)
        context = normal
    else:
        expected = adapt_sketch(params, support, cfg)[0]
        normal = adapt_sketch(params, explicit, cfg)[0]
        different = adapt_sketch(params, changed, cfg)[0]
        assert float(tree_difference_norm(normal, different)) > 1e-8
        fast, context = normal, None
    _assert_tree_equal(expected, normal)
    changed_query = {**query, 'knot_mask': jnp.zeros_like(query['point_mask'])}
    _assert_tree_equal(teacher_forced_distribution(params, fast, query, cfg, context),
                       teacher_forced_distribution(params, fast, changed_query, cfg, context))
    _assert_tree_equal(query_loss(params, fast, query, cfg, context),
                       query_loss(params, fast, changed_query, cfg, context))
    padded = _section(pad=3)
    padded['knot_mask'] = padded['point_mask'].at[:, 4:].set(True)
    clean = {**padded, 'knot_mask': padded['point_mask']}
    if model_type == 'explicit_context':
        _assert_tree_equal(encode_context(params, padded, cfg), encode_context(params, clean, cfg))
    else:
        _assert_tree_equal(adapt_sketch(params, padded, cfg)[0], adapt_sketch(params, clean, cfg)[0])


def test_eager_jit_vmap_and_task_loop_agree():
    cfg, params, batch = _fixture(fast_grad_clip_norm=.002, fast_update_clip_norm=.0001)
    batch = jax.tree_util.tree_map(lambda x: jnp.concatenate([x, x], axis=0), batch)
    batch['support']['tokens'] = batch['support']['tokens'].at[1, :, :3, :2].add(.1)
    eager = meta_objective(params, batch, cfg)
    compiled = jax.jit(lambda p, b: meta_objective(p, b, cfg))(params, batch)
    _assert_tree_equal(eager, compiled, exact=False)
    batched_fast, batched_trace = jax.vmap(
        lambda support: adapt_sketch(params, support, cfg)
    )(batch['support'])
    loop = []
    for index in range(2):
        support = jax.tree_util.tree_map(lambda x: x[index], batch['support'])
        query = jax.tree_util.tree_map(lambda x: x[index], batch['query'])
        fast, trace = adapt_sketch(params, support, cfg)
        selected = jax.tree_util.tree_map(lambda x: x[index], batched_fast)
        for left, right in zip(jax.tree_util.tree_leaves(fast), jax.tree_util.tree_leaves(selected)):
            np.testing.assert_allclose(left, right, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(trace['fast_update_norm'], batched_trace['fast_update_norm'][index],
                                   rtol=2e-5, atol=1e-8)
        assert bool(jnp.all(trace['fast_grad_norm'] > cfg.fast_grad_clip_norm))
        bound = cfg.fast_update_clip_norm * np.sqrt(len(jax.tree_util.tree_leaves(fast)))
        assert bool(jnp.all(trace['fast_update_norm'] <= bound + 1e-8))
        loop.append(query_loss(params, fast, query, cfg)[0])
    batched_losses = jax.vmap(lambda fast, query: query_loss(params, fast, query, cfg)[0])(
        batched_fast, batch['query']
    )
    # GPU batched/unbatched float32 kernels round differently. Direct fast-state
    # and active-clipping checks above isolate the update rule from NLL rounding.
    np.testing.assert_allclose(batched_losses, jnp.stack(loop), rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(eager[0], jnp.mean(jnp.stack(loop)), rtol=2e-5, atol=1e-6)
    fast, _ = adapt_sketch(params, _task(batch['support']), cfg)
    key = jax.random.key(8)
    _assert_tree_equal(generate(params, fast, cfg, key),
                       jax.jit(lambda p, f, k: generate(p, f, cfg, k))(params, fast, key), exact=False)


def test_stochastic_output_consumes_generation_keys():
    cfg, params, batch = _fixture()
    distribution = {'mixture_logits': jnp.zeros((2,)), 'xy_mean': jnp.zeros((2, 2)),
                    'xy_log_scale': jnp.zeros((2, 2)), 'pen_logit': jnp.asarray(0.),
                    'stop_logit': jnp.asarray(-100.)}
    a = sample_token(distribution, cfg, jax.random.key(1))
    b = sample_token(distribution, cfg, jax.random.key(2))
    assert not np.array_equal(a[:2], b[:2])


def test_first_update_diagnostics_use_actual_clipped_update_and_descent_sign():
    cfg, params, batch = _fixture(fast_grad_clip_norm=.002, fast_update_clip_norm=.0001)
    support, query = _task(batch['support']), _task(batch['query'])
    diagnostic = first_write_diagnostics(params, support, query, cfg)
    initial = initial_fast_state(params)
    first = jax.tree_util.tree_map(lambda w, d: w + d, initial, diagnostic['first_update'])
    np.testing.assert_allclose(diagnostic['query_loss_after_first_update'], query_loss(params, first, query, cfg)[0])
    predicted = -sum(jnp.sum(g * d) for g, d in zip(jax.tree_util.tree_leaves(diagnostic['query_gradient']), jax.tree_util.tree_leaves(diagnostic['first_update'])))
    np.testing.assert_allclose(diagnostic['local_improvement_prediction'], predicted)
    assert float(tree_l2_norm(diagnostic['raw_write_gradient'])) > float(tree_l2_norm(diagnostic['clipped_write_gradient']))


def test_b2_step_reads_actual_state_and_bounds_motion():
    cfg, params, batch = _fixture(experiment='b2', motion_bound=.07)
    fast, _ = adapt_sketch(params, _task(batch['support']), cfg)
    carry = initial_query_carry(cfg)
    previous = jnp.zeros((4,))
    frame = jnp.asarray([0., 0., 0., 1.])
    _, left = query_step(params, fast, cfg, carry, previous, 0, frame, jnp.zeros((3,)))
    _, right = query_step(params, fast, cfg, carry, previous, 0, frame, jnp.asarray([.7, -.2, 0.]))
    assert bool(jnp.all(jnp.abs(right['xy']) <= cfg.motion_bound))
    assert not np.allclose(left['xy'], right['xy'])
    with pytest.raises(ValueError, match='environment rollout'):
        generate(params, fast, cfg, jax.random.key(0))
