from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class RLBenchActionLossConfig:
    translation_weight: float = 1.0
    rotation_weight: float = 1.0
    gripper_weight: float = 0.25
    translation_huber_delta: float = 0.1
    translation_scale: float = 0.05


def rotation_6d_to_matrix(rotation_6d: jax.Array) -> jax.Array:
    rotation_6d = jnp.asarray(rotation_6d, dtype=jnp.float32)
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:6]
    first = first / jnp.maximum(jnp.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - jnp.sum(first * second, axis=-1, keepdims=True) * first
    second = second / jnp.maximum(jnp.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = jnp.cross(first, second)
    return jnp.stack([first, second, third], axis=-1)


def matrix_to_rotation_6d(matrix: jax.Array) -> jax.Array:
    matrix = jnp.asarray(matrix, dtype=jnp.float32)
    return jnp.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def quaternion_xyzw_to_matrix(quaternion: jax.Array) -> jax.Array:
    quaternion = jnp.asarray(quaternion, dtype=jnp.float32)
    quaternion = quaternion / jnp.maximum(
        jnp.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8
    )
    x, y, z, w = [quaternion[..., index] for index in range(4)]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return jnp.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_quaternion_xyzw(matrix: jax.Array) -> jax.Array:
    matrix = jnp.asarray(matrix, dtype=jnp.float32)
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    magnitudes = jnp.sqrt(
        jnp.maximum(
            jnp.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                axis=-1,
            ),
            0.0,
        )
    )
    candidates = jnp.stack(
        [
            jnp.stack([m21 - m12, m02 - m20, m10 - m01, magnitudes[..., 0] ** 2], axis=-1),
            jnp.stack([magnitudes[..., 1] ** 2, m01 + m10, m02 + m20, m21 - m12], axis=-1),
            jnp.stack([m01 + m10, magnitudes[..., 2] ** 2, m12 + m21, m02 - m20], axis=-1),
            jnp.stack([m02 + m20, m12 + m21, magnitudes[..., 3] ** 2, m10 - m01], axis=-1),
        ],
        axis=-2,
    )
    candidates = candidates / jnp.maximum(2.0 * magnitudes[..., :, None], 1e-8)
    choice = jnp.argmax(magnitudes, axis=-1)
    quaternion = jnp.take_along_axis(
        candidates, choice[..., None, None], axis=-2
    )[..., 0, :]
    return quaternion / jnp.maximum(
        jnp.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8
    )


def rotation_geodesic_loss(
    prediction_matrix: jax.Array, target_matrix: jax.Array
) -> jax.Array:
    relative = jnp.matmul(
        jnp.swapaxes(prediction_matrix, -1, -2), target_matrix
    )
    cosine = (jnp.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5
    return jnp.arccos(jnp.clip(cosine, -1.0 + 1e-6, 1.0 - 1e-6))


def quaternion_geodesic_loss(
    prediction_xyzw: jax.Array, target_xyzw: jax.Array
) -> jax.Array:
    prediction = prediction_xyzw / jnp.maximum(
        jnp.linalg.norm(prediction_xyzw, axis=-1, keepdims=True), 1e-8
    )
    target = target_xyzw / jnp.maximum(
        jnp.linalg.norm(target_xyzw, axis=-1, keepdims=True), 1e-8
    )
    cosine = jnp.abs(jnp.sum(prediction * target, axis=-1))
    return 2.0 * jnp.arccos(jnp.clip(cosine, 0.0, 1.0 - 1e-6))


def encode_rlbench_action_target(
    action: jax.Array,
    current_eef_position: jax.Array,
    cfg: RLBenchActionLossConfig,
) -> Dict[str, jax.Array]:
    action = jnp.asarray(action, dtype=jnp.float32)
    current_eef_position = jnp.asarray(current_eef_position, dtype=jnp.float32)
    translation_delta = (action[..., :3] - current_eef_position) / float(
        cfg.translation_scale
    )
    rotation_matrix = quaternion_xyzw_to_matrix(action[..., 3:7])
    return {
        'translation_delta': translation_delta,
        'rotation_6d': matrix_to_rotation_6d(rotation_matrix),
        'rotation_matrix': rotation_matrix,
        'gripper': action[..., 7],
    }


def decode_rlbench_action_prediction(
    prediction: Dict[str, jax.Array],
    current_eef_position: jax.Array,
    cfg: RLBenchActionLossConfig,
) -> jax.Array:
    translation = current_eef_position + jnp.clip(
        prediction['translation_delta'], -1.0, 1.0
    ) * float(cfg.translation_scale)
    rotation_matrix = rotation_6d_to_matrix(prediction['rotation_6d'])
    quaternion = matrix_to_quaternion_xyzw(rotation_matrix)
    gripper = jax.nn.sigmoid(prediction['gripper_logit'])
    return jnp.concatenate(
        [translation, quaternion, gripper[..., None]], axis=-1
    )


def _masked_mean(value: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(jnp.float32)
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    weighted = value.astype(jnp.float32) * mask
    denominator = jnp.sum(jnp.ones_like(value, dtype=jnp.float32) * mask)
    return jnp.sum(weighted) / jnp.maximum(denominator, 1.0)


def rlbench_action_loss(
    prediction: Dict[str, jax.Array],
    target_action: jax.Array,
    current_eef_position: jax.Array,
    mask: jax.Array,
    cfg: RLBenchActionLossConfig,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    target = encode_rlbench_action_target(target_action, current_eef_position, cfg)
    translation_error = prediction['translation_delta'] - target['translation_delta']
    absolute = jnp.abs(translation_error)
    delta = float(cfg.translation_huber_delta)
    translation_element = jnp.where(
        absolute <= delta,
        0.5 * jnp.square(translation_error) / max(delta, 1e-6),
        absolute - 0.5 * delta,
    )
    translation_loss = _masked_mean(translation_element, mask)
    prediction_rotation = rotation_6d_to_matrix(prediction['rotation_6d'])
    rotation_loss = _masked_mean(
        rotation_geodesic_loss(prediction_rotation, target['rotation_matrix']), mask
    )
    gripper_logit = prediction['gripper_logit']
    gripper_target = target['gripper']
    gripper_element = jnp.maximum(gripper_logit, 0.0) - (
        gripper_logit * gripper_target
    ) + jnp.log1p(jnp.exp(-jnp.abs(gripper_logit)))
    gripper_loss = _masked_mean(gripper_element, mask)
    total = (
        float(cfg.translation_weight) * translation_loss
        + float(cfg.rotation_weight) * rotation_loss
        + float(cfg.gripper_weight) * gripper_loss
    )
    return total, {
        'translation_loss': translation_loss,
        'rotation_loss': rotation_loss,
        'gripper_loss': gripper_loss,
        'translation_l1': _masked_mean(jnp.abs(translation_error), mask),
    }
