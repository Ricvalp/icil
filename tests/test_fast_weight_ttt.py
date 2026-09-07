from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from icil_jax_rlbench.data.hidden_goal import (
    HiddenGoalConfig,
    HiddenGoalMetaSampler,
    HiddenGoalTaskBank,
    fit_state_normalizer,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    TTTAdaptConfig,
    adapt_fast_state,
    initial_fast_state,
    init_fast_weight_ttt_params,
    predict_action,
    tree_difference_norm,
    tree_l2_norm,
)
from icil_jax_rlbench.train.ttt_step import (
    TTTStepConfig,
    create_ttt_train_state,
    create_ttt_train_step,
    ttt_meta_objective,
)
from icil_jax_rlbench.train.ttt_runner import _replicate_state


def _setup(*, first_order: bool = False):
    benchmark_cfg = HiddenGoalConfig(
        num_train_tasks=5,
        num_validation_tasks=2,
        num_test_tasks=2,
        horizon=4,
        support_episodes=1,
        query_episodes=1,
    )
    task_bank = HiddenGoalTaskBank(benchmark_cfg)
    normalizer = fit_state_normalizer(
        benchmark_cfg, task_bank, episodes_per_task=1
    )
    batch = HiddenGoalMetaSampler(
        benchmark_cfg, task_bank, normalizer, split='train', seed=3
    ).build_batch(1)
    batch = jax.tree_util.tree_map(jnp.asarray, batch)
    model_cfg = FastWeightTTTConfig(
        hidden_dim=12,
        fast_dim=6,
        fast_hidden_dim=8,
        gate_init=0.1,
        inner_lr_init=0.1,
    )
    adapt_cfg = TTTAdaptConfig(
        write_objective='kvb',
        write_segment_size=2,
        first_order=first_order,
    )
    params = init_fast_weight_ttt_params(jax.random.key(0), model_cfg)
    return batch, model_cfg, adapt_cfg, params


def _one_task(section):
    return jax.tree_util.tree_map(lambda value: value[0], section)


def _gradient(params, batch, model_cfg, adapt_cfg):
    step_cfg = TTTStepConfig()
    return jax.grad(
        lambda value: ttt_meta_objective(
            value, batch, model_cfg, adapt_cfg, step_cfg
        )[0]
    )(params)


def test_prediction_api_structurally_forbids_direct_support_and_query_actions():
    signature = inspect.signature(predict_action)
    assert 'support' not in signature.parameters
    assert 'action' not in signature.parameters
    assert tuple(signature.parameters)[:3] == ('params', 'fast_state', 'observation')


def test_delta_read_is_exact_query_only_noop_at_initial_fast_state():
    batch, model_cfg, _, params = _setup()
    query = _one_task(batch['query'])
    observation = query['observation'][0, 0]
    read_projection = dict(params['read_projection'])
    read_projection['bias'] = jnp.linspace(
        -0.5, 0.5, model_cfg.hidden_dim, dtype=jnp.float32
    )
    params = {**params, 'read_projection': read_projection}
    initial = initial_fast_state(params)

    query_only = predict_action(
        params, initial, observation, model_cfg, read_enabled=False
    )
    delta_read = predict_action(
        params,
        initial,
        observation,
        model_cfg,
        read_mode='delta',
        read_scale=1.0,
    )
    np.testing.assert_array_equal(np.asarray(query_only), np.asarray(delta_read))


def test_delta_read_repairs_zero_gate_support_bc_adaptation():
    batch, model_cfg, _, params = _setup()
    params = {**params, 'read_gate': jnp.zeros_like(params['read_gate'])}
    support = _one_task(batch['support'])
    initial = initial_fast_state(params)
    legacy_cfg = TTTAdaptConfig(
        write_objective='action_bc',
        write_segment_size=2,
        read_mode='absolute_gated',
    )
    repaired_cfg = TTTAdaptConfig(
        write_objective='action_bc',
        write_segment_size=2,
        read_mode='delta',
    )

    legacy_state, _ = adapt_fast_state(params, support, model_cfg, legacy_cfg)
    repaired_state, _ = adapt_fast_state(params, support, model_cfg, repaired_cfg)
    assert float(tree_difference_norm(initial, legacy_state)) == 0.0
    assert float(tree_difference_norm(initial, repaired_state)) > 1e-8


def test_full_second_order_support_bc_delta_read_has_meta_gradient_paths():
    batch, model_cfg, _, params = _setup(first_order=False)
    params = {**params, 'read_gate': jnp.zeros_like(params['read_gate'])}
    adapt_cfg = TTTAdaptConfig(
        write_objective='action_bc',
        write_segment_size=2,
        read_mode='delta',
        first_order=False,
    )
    gradient = _gradient(params, batch, model_cfg, adapt_cfg)
    for name in ('query_projection', 'inner_lr_raw', 'read_projection'):
        assert float(tree_l2_norm(gradient[name])) > 1e-12, name
    assert float(tree_l2_norm(gradient['read_gate'])) == 0.0


def test_fast_state_reset_carry_and_no_direct_support_bypass():
    batch, model_cfg, adapt_cfg, params = _setup()
    support = _one_task(batch['support'])
    query = _one_task(batch['query'])
    initial_a = initial_fast_state(params)
    initial_b = initial_fast_state(params)
    assert float(tree_difference_norm(initial_a, initial_b)) == 0.0

    adapted, _ = adapt_fast_state(params, support, model_cfg, adapt_cfg)
    assert float(tree_difference_norm(initial_a, adapted)) > 0.0
    observation = query['observation'][0, 0]
    before = predict_action(params, initial_a, observation, model_cfg)
    after = predict_action(params, adapted, observation, model_cfg)
    assert not np.allclose(np.asarray(before), np.asarray(after), atol=1e-9)

    changed_support = dict(support)
    changed_support['action'] = -support['action']
    # Frozen fast state means changed support is irrelevant to READ by construction.
    frozen_a = predict_action(params, adapted, observation, model_cfg)
    del changed_support
    frozen_b = predict_action(params, adapted, observation, model_cfg)
    np.testing.assert_array_equal(np.asarray(frozen_a), np.asarray(frozen_b))


def test_support_outer_mask_never_enters_meta_objective():
    batch, model_cfg, adapt_cfg, params = _setup()
    step_cfg = TTTStepConfig()
    loss_a = ttt_meta_objective(
        params, batch, model_cfg, adapt_cfg, step_cfg
    )[0]
    changed = dict(batch)
    changed_support = dict(batch['support'])
    changed_support['outer_loss_mask'] = jnp.ones_like(
        changed_support['outer_loss_mask']
    )
    changed['support'] = changed_support
    loss_b = ttt_meta_objective(
        params, changed, model_cfg, adapt_cfg, step_cfg
    )[0]
    np.testing.assert_array_equal(np.asarray(loss_a), np.asarray(loss_b))


def test_full_second_order_reaches_every_write_parameter_group():
    batch, model_cfg, adapt_cfg, params = _setup(first_order=False)
    gradient = _gradient(params, batch, model_cfg, adapt_cfg)
    for name in (
        'key_projection',
        'value_projection',
        'query_projection',
        'fast_init',
        'inner_lr_raw',
        'read_gate',
    ):
        assert float(tree_l2_norm(gradient[name])) > 1e-12, name


def test_fomaml_removes_projection_paths_that_only_act_through_write():
    batch, model_cfg, full_cfg, params = _setup(first_order=False)
    first_cfg = TTTAdaptConfig(**{**full_cfg.__dict__, 'first_order': True})
    full_gradient = _gradient(params, batch, model_cfg, full_cfg)
    first_gradient = _gradient(params, batch, model_cfg, first_cfg)
    assert float(tree_l2_norm(full_gradient['key_projection'])) > 1e-12
    assert float(tree_l2_norm(full_gradient['value_projection'])) > 1e-12
    assert float(tree_l2_norm(first_gradient['key_projection'])) < 1e-12
    assert float(tree_l2_norm(first_gradient['value_projection'])) < 1e-12
    assert float(tree_l2_norm(first_gradient['query_projection'])) > 1e-12


def test_selected_second_order_gradient_matches_finite_difference():
    batch, model_cfg, adapt_cfg, params = _setup(first_order=False)
    step_cfg = TTTStepConfig()
    gradient = _gradient(params, batch, model_cfg, adapt_cfg)
    index = (0, 0)
    analytic = float(gradient['value_projection']['kernel'][index])
    epsilon = 5e-3

    def shifted(delta):
        changed = dict(params)
        projection = dict(params['value_projection'])
        projection['kernel'] = projection['kernel'].at[index].add(delta)
        changed['value_projection'] = projection
        return float(
            ttt_meta_objective(
                changed, batch, model_cfg, adapt_cfg, step_cfg
            )[0]
        )

    finite_difference = (shifted(epsilon) - shifted(-epsilon)) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, finite_difference, rtol=0.25, atol=2e-5)


def test_jit_and_eager_objectives_match_and_update_is_deterministic():
    batch, model_cfg, adapt_cfg, params = _setup()
    step_cfg = TTTStepConfig()
    eager = ttt_meta_objective(params, batch, model_cfg, adapt_cfg, step_cfg)
    compiled = jax.jit(
        lambda value, data: ttt_meta_objective(
            value, data, model_cfg, adapt_cfg, step_cfg
        )
    )(params, batch)
    np.testing.assert_allclose(
        np.asarray(eager[0]), np.asarray(compiled[0]), rtol=5e-6, atol=1e-6
    )

    optimizer = optax.adam(1e-3)
    state_a = create_ttt_train_state(params, optimizer, jax.random.key(11))
    state_b = create_ttt_train_state(params, optimizer, jax.random.key(11))
    train_step = create_ttt_train_step(
        optimizer, model_cfg, adapt_cfg, step_cfg
    )
    next_a, metrics_a = train_step(state_a, batch)
    next_b, metrics_b = train_step(state_b, batch)
    leaves_a = jax.tree_util.tree_leaves(next_a.params)
    leaves_b = jax.tree_util.tree_leaves(next_b.params)
    for left, right in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    np.testing.assert_array_equal(
        np.asarray(metrics_a['loss']), np.asarray(metrics_b['loss'])
    )


def test_full_second_order_is_finite_with_fully_padded_write_segment():
    batch, model_cfg, adapt_cfg, params = _setup(first_order=False)
    support = dict(batch['support'])
    for name in ('observation', 'action', 'next_observation'):
        support[name] = support[name].at[:, :, 2:].set(0.0)
    support['write_mask'] = support['write_mask'].at[:, :, 2:].set(False)
    batch = {**batch, 'support': support}

    gradient = _gradient(params, batch, model_cfg, adapt_cfg)
    for leaf in jax.tree_util.tree_leaves(gradient):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_single_device_jit_and_pmap_steps_agree():
    if jax.local_device_count() != 1:
        pytest.skip('This comparison intentionally isolates one-device pmap semantics.')
    batch, model_cfg, adapt_cfg, params = _setup()
    step_cfg = TTTStepConfig()
    optimizer = optax.adam(1e-3)
    state = create_ttt_train_state(params, optimizer, jax.random.key(31))
    single_step = create_ttt_train_step(
        optimizer, model_cfg, adapt_cfg, step_cfg, distributed=False
    )
    pmap_step = create_ttt_train_step(
        optimizer, model_cfg, adapt_cfg, step_cfg, distributed=True
    )
    single_state, single_metrics = single_step(state, batch)
    devices = jax.local_devices()
    replicated_state = _replicate_state(state, devices)
    sharded_batch = jax.tree_util.tree_map(
        lambda value: value.reshape(
            (len(devices), value.shape[0] // len(devices)) + value.shape[1:]
        ),
        batch,
    )
    distributed_state, distributed_metrics = pmap_step(
        replicated_state, sharded_batch
    )
    distributed_params = jax.tree_util.tree_map(
        lambda value: value[0], distributed_state.params
    )
    for single, distributed in zip(
        jax.tree_util.tree_leaves(single_state.params),
        jax.tree_util.tree_leaves(distributed_params),
    ):
        np.testing.assert_allclose(
            np.asarray(single), np.asarray(distributed), rtol=2e-5, atol=2e-6
        )
    np.testing.assert_allclose(
        np.asarray(single_metrics['loss']),
        np.asarray(distributed_metrics['loss'][0]),
        rtol=2e-5,
        atol=2e-6,
    )
