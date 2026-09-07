from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    init_fast_weight_ttt_params,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint, save_checkpoint
from icil_jax_rlbench.train.ttt_step import create_ttt_train_state


def test_checkpoint_contains_w0_but_not_transient_task_fast_state(tmp_path):
    model_cfg = FastWeightTTTConfig(hidden_dim=8, fast_dim=4, fast_hidden_dim=6)
    params = init_fast_weight_ttt_params(jax.random.key(0), model_cfg)
    optimizer = optax.adam(1e-3)
    state = create_ttt_train_state(params, optimizer, jax.random.key(1))
    path = tmp_path / 'ttt.pkl'
    save_checkpoint(
        path,
        state=state,
        step=0,
        config={'mode': 'ttt_adaptation_only'},
        extra={'transient_fast_state_saved': False},
        replicated=False,
    )
    payload = load_checkpoint(path)
    assert 'fast_init' in payload['params']
    assert 'fast_state' not in payload
    assert payload['extra']['transient_fast_state_saved'] is False
    restored = jax.tree_util.tree_map(jnp.asarray, payload['params']['fast_init'])
    for expected, actual in zip(
        jax.tree_util.tree_leaves(params['fast_init']),
        jax.tree_util.tree_leaves(restored),
    ):
        assert jnp.array_equal(expected, actual)
