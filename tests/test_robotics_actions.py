from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from icil_jax_rlbench.models.robotics_actions import (
    RLBenchActionLossConfig,
    decode_rlbench_action_prediction,
    encode_rlbench_action_target,
    matrix_to_quaternion_xyzw,
    matrix_to_rotation_6d,
    quaternion_geodesic_loss,
    quaternion_xyzw_to_matrix,
    rlbench_action_loss,
    rotation_6d_to_matrix,
)


def _random_quaternion(key, count=16):
    quaternion = jax.random.normal(key, (count, 4))
    return quaternion / jnp.linalg.norm(quaternion, axis=-1, keepdims=True)


def test_rotation_6d_round_trip_produces_valid_rotation_matrices():
    quaternion = _random_quaternion(jax.random.key(0))
    matrix = quaternion_xyzw_to_matrix(quaternion)
    reconstructed = rotation_6d_to_matrix(matrix_to_rotation_6d(matrix))
    np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(matrix), atol=1e-5)
    identity = jnp.matmul(jnp.swapaxes(reconstructed, -1, -2), reconstructed)
    np.testing.assert_allclose(
        np.asarray(identity), np.broadcast_to(np.eye(3), identity.shape), atol=1e-5
    )


def test_quaternion_matrix_round_trip_is_sign_invariant():
    quaternion = _random_quaternion(jax.random.key(1))
    reconstructed = matrix_to_quaternion_xyzw(quaternion_xyzw_to_matrix(quaternion))
    loss = quaternion_geodesic_loss(reconstructed, quaternion)
    assert float(jnp.max(loss)) < 5e-3
    sign_loss = quaternion_geodesic_loss(-quaternion, quaternion)
    assert float(jnp.max(sign_loss)) < 5e-3


def test_rlbench_component_loss_and_environment_decode():
    cfg = RLBenchActionLossConfig(translation_scale=0.1)
    current = jnp.asarray([[0.2, -0.1, 0.5]], dtype=jnp.float32)
    quaternion = jnp.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)
    action = jnp.concatenate(
        [current + jnp.asarray([[0.05, -0.02, 0.01]]), quaternion, jnp.ones((1, 1))],
        axis=-1,
    )
    target = encode_rlbench_action_target(action, current, cfg)
    prediction = {
        'translation_delta': target['translation_delta'],
        'rotation_6d': target['rotation_6d'],
        'gripper_logit': jnp.asarray([10.0]),
    }
    loss, metrics = rlbench_action_loss(
        prediction, action, current, jnp.ones((1,), dtype=jnp.bool_), cfg
    )
    assert float(loss) < 5e-3
    assert float(metrics['translation_loss']) < 1e-7
    decoded = decode_rlbench_action_prediction(prediction, current, cfg)
    np.testing.assert_allclose(np.asarray(decoded[..., :3]), np.asarray(action[..., :3]), atol=1e-6)
