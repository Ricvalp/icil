from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from icil_jax_rlbench.models.ttt_supernode import (
    RLBenchTTTFeatureEncoder,
    TTTEventEncoderConfig,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    TTTAdaptConfig,
    adapt_encoded_support,
    init_fast_weight_ttt_params,
    read_fast_memory,
    tree_difference_norm,
    initial_fast_state,
)


def test_spacetime_supernodes_produce_small_sequential_registers():
    cfg = TTTEventEncoderConfig(
        d_model=16,
        n_heads=4,
        use_rgb=False,
        use_mask_id=False,
        supernodes=4,
        supernode_layers=1,
        spacetime_supernodes=4,
        spacetime_layers=1,
        supernode_center_sampling='linspace',
        support_registers=3,
        query_registers=2,
        register_layers=1,
    )
    model = RLBenchTTTFeatureEncoder(cfg, state_dim=8, action_dim=8)
    support = {
        'xyz': jax.random.normal(jax.random.key(0), (2, 3, 2, 8, 3)) * 0.1,
        'time': jnp.broadcast_to(
            jnp.linspace(0.0, 1.0, 2)[None, None, :, None], (2, 3, 2, 8)
        ),
        'state': jax.random.normal(jax.random.key(1), (2, 3, 2, 8)),
        'action': jax.random.normal(jax.random.key(2), (2, 3, 2, 8)),
        'valid': jnp.ones((2, 3, 2, 8), dtype=jnp.bool_),
    }
    query = {
        'xyz': jax.random.normal(jax.random.key(3), (2, 2, 8, 3)) * 0.1,
        'state': jax.random.normal(jax.random.key(4), (2, 2, 8)),
        'valid': jnp.ones((2, 2, 8), dtype=jnp.bool_),
    }
    variables = model.init(jax.random.key(5), support, query)
    support_registers, support_mask, query_registers, query_mask = model.apply(
        variables, support, query
    )
    assert support_registers.shape == (2, 3, 3, 16)
    assert support_mask.shape == (2, 3, 3)
    assert query_registers.shape == (2, 2, 16)
    assert query_mask.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(support_registers)))
    assert 'action' not in query


def test_supernode_occupancy_and_bandwidth_metrics_are_finite():
    cfg = TTTEventEncoderConfig(
        d_model=8,
        n_heads=2,
        use_rgb=False,
        spacetime_supernodes=3,
        spacetime_layers=0,
        supernode_center_sampling='linspace',
        support_registers=2,
        query_registers=1,
    )
    model = RLBenchTTTFeatureEncoder(cfg, state_dim=8, action_dim=8)
    support = {
        'xyz': jnp.zeros((1, 1, 2, 4, 3), dtype=jnp.float32),
        'time': jnp.broadcast_to(
            jnp.asarray([0.0, 1.0])[None, None, :, None], (1, 1, 2, 4)
        ),
        'state': jnp.zeros((1, 1, 2, 8), dtype=jnp.float32),
        'action': jnp.zeros((1, 1, 2, 8), dtype=jnp.float32),
        'valid': jnp.ones((1, 1, 2, 4), dtype=jnp.bool_),
    }
    query = {
        'xyz': jnp.zeros((1, 1, 4, 3), dtype=jnp.float32),
        'state': jnp.zeros((1, 1, 8), dtype=jnp.float32),
        'valid': jnp.ones((1, 1, 4), dtype=jnp.bool_),
    }
    variables = model.init(jax.random.key(6), support, query)
    _, _, stats = model.apply(
        variables, support, method=RLBenchTTTFeatureEncoder.encode_support, return_stats=True
    )
    for value in stats.values():
        assert np.all(np.isfinite(np.asarray(value)))
    assert float(stats['spacetime_bandwidth_xyz']) > 0.0
    assert float(stats['spacetime_bandwidth_time']) > 0.0


def test_visual_registers_use_identical_fast_weight_write_read_interface():
    model_cfg = FastWeightTTTConfig(
        observation_dim=4,
        action_dim=3,
        translation_dim=2,
        hidden_dim=8,
        fast_dim=4,
        fast_hidden_dim=6,
        gate_init=0.1,
    )
    params = init_fast_weight_ttt_params(jax.random.key(20), model_cfg)
    registers = jax.random.normal(jax.random.key(21), (3, 2, 8))
    mask = jnp.ones((3, 2), dtype=jnp.bool_)
    adapted, trace = adapt_encoded_support(
        params,
        registers,
        mask,
        model_cfg,
        TTTAdaptConfig(write_segment_size=2),
    )
    assert float(tree_difference_norm(adapted, initial_fast_state(params))) > 0.0
    assert trace['write_loss'].shape == (3,)
    query_feature = jax.random.normal(jax.random.key(22), (8,))
    before = read_fast_memory(
        params, initial_fast_state(params), query_feature, model_cfg
    )
    after = read_fast_memory(params, adapted, query_feature, model_cfg)
    assert not np.allclose(np.asarray(before), np.asarray(after), atol=1e-9)
