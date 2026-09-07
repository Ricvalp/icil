from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .attention import TransformerConfig
from .perceiver import LatentPerceiver, PerceiverConfig, SelfAttentionStack


@dataclass(frozen=True)
class TTTEventEncoderConfig:
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.0
    mlp_mult: int = 4
    supernodes: int = 64
    supernode_temperature: float = 0.005
    supernode_center_sampling: str = 'linspace'
    supernode_layers: int = 2
    spacetime_supernodes: int = 256
    spacetime_temperature_xyz: float = 0.005
    spacetime_temperature_t: float = 0.04
    spacetime_layers: int = 2
    mask_id_vocab: int = 256
    use_rgb: bool = True
    use_mask_id: bool = False
    support_registers: int = 16
    query_registers: int = 8
    register_layers: int = 1
    min_bandwidth: float = 1e-4
    occupancy_threshold: float = 1e-3
    learnable_bandwidths: bool = True

    def transformer(self) -> TransformerConfig:
        return TransformerConfig(
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            mlp_mult=int(self.mlp_mult),
            dropout=float(self.dropout),
        )

    def perceiver(self, *, num_latents: int) -> PerceiverConfig:
        return PerceiverConfig(
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_layers=int(self.register_layers),
            num_latents=int(num_latents),
            mlp_mult=int(self.mlp_mult),
            dropout=float(self.dropout),
        )


class _PointFeatureEmbed(nn.Module):
    cfg: TTTEventEncoderConfig

    @nn.compact
    def __call__(
        self,
        xyz: jax.Array,
        *,
        rgb: Optional[jax.Array] = None,
        mask_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        pieces = [xyz.astype(jnp.float32)]
        if bool(self.cfg.use_rgb) and rgb is not None:
            pieces.append(rgb.astype(jnp.float32))
        features = nn.Dense(int(self.cfg.d_model), name='xyz_rgb_projection')(
            jnp.concatenate(pieces, axis=-1)
        )
        if bool(self.cfg.use_mask_id) and mask_id is not None:
            vocab = int(self.cfg.mask_id_vocab)
            indices = jnp.clip(mask_id.astype(jnp.int32), 0, vocab - 1)
            features = features + nn.Embed(vocab, int(self.cfg.d_model), name='mask_embedding')(
                indices
            )
        return features


def _linspace_center_indices(
    batch_size: int, num_points: int, num_centers: int
) -> jax.Array:
    indices = jnp.linspace(
        0, max(int(num_points) - 1, 0), int(num_centers)
    ).round().astype(jnp.int32)
    return jnp.broadcast_to(indices[None, :], (int(batch_size), int(num_centers)))


def _mask_balanced_center_indices(
    valid: jax.Array, mask_id: jax.Array, num_centers: int
) -> jax.Array:
    valid = valid.astype(jnp.bool_)
    mask_id = mask_id.astype(jnp.int32)
    batch, points = map(int, valid.shape)
    centers = int(num_centers)

    def one_row(row_valid: jax.Array, row_mask: jax.Array) -> jax.Array:
        invalid_mask = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
        safe_mask = jnp.where(row_valid, row_mask, invalid_mask)
        order = jnp.argsort(safe_mask, stable=True)
        sorted_valid = row_valid[order]
        sorted_mask = safe_mask[order]

        previous_mask = jnp.concatenate(
            [jnp.asarray([-1], dtype=jnp.int32), sorted_mask[:-1]], axis=0
        )
        group_start = (sorted_mask != previous_mask) & sorted_valid
        group_id = jnp.cumsum(group_start.astype(jnp.int32), axis=0) - 1
        safe_group_id = jnp.maximum(group_id, 0)

        sorted_position = jnp.arange(points, dtype=jnp.int32)
        start_position = jnp.where(group_start, sorted_position, 0)
        group_start_position = jax.lax.associative_scan(
            jnp.maximum, start_position, axis=0
        )
        rank_in_group = sorted_position - group_start_position
        group_counts = jnp.bincount(
            safe_group_id,
            weights=sorted_valid.astype(jnp.float32),
            length=points,
        )
        count = jnp.maximum(group_counts[safe_group_id], 1.0)
        rank_fraction = rank_in_group.astype(jnp.float32) / count
        group_tie = safe_group_id.astype(jnp.float32) / float(max(points, 1))
        balanced_key = rank_fraction + 1e-3 * group_tie
        invalid_key = 2.0 + sorted_position.astype(jnp.float32) / float(
            max(points, 1)
        )
        balanced_key = jnp.where(sorted_valid, balanced_key, invalid_key)
        balanced_order = jnp.argsort(balanced_key, stable=True)
        take_position = jnp.arange(centers, dtype=jnp.int32) % max(points, 1)
        return order[balanced_order[take_position]]

    return jax.vmap(one_row)(valid, mask_id).reshape(batch, centers)


def _supernode_center_indices(
    *,
    valid: jax.Array,
    mask_id: Optional[jax.Array],
    num_centers: int,
    center_sampling: str,
) -> jax.Array:
    batch, points = map(int, valid.shape)
    if str(center_sampling) == 'mask_balanced' and mask_id is not None:
        return _mask_balanced_center_indices(valid, mask_id, int(num_centers))
    if str(center_sampling) != 'linspace':
        raise ValueError(
            "supernode_center_sampling must be 'linspace' or 'mask_balanced'."
        )
    return _linspace_center_indices(batch, points, int(num_centers))


class _SupernodeFrameTokenizer(nn.Module):
    cfg: TTTEventEncoderConfig

    @nn.compact
    def __call__(
        self,
        xyz: jax.Array,
        state: jax.Array,
        valid: jax.Array,
        *,
        rgb: Optional[jax.Array] = None,
        mask_id: Optional[jax.Array] = None,
        train: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        center_indices = _supernode_center_indices(
            valid=valid,
            mask_id=mask_id,
            num_centers=int(self.cfg.supernodes),
            center_sampling=str(self.cfg.supernode_center_sampling),
        )
        point_tokens = _PointFeatureEmbed(self.cfg, name='point_embed')(
            xyz, rgb=rgb, mask_id=mask_id
        )
        centers = jnp.take_along_axis(
            xyz.astype(jnp.float32), center_indices[:, :, None], axis=1
        )
        distance = jnp.sum(
            jnp.square(centers[:, :, None, :] - xyz[:, None, :, :]), axis=-1
        )
        logits = -distance / max(float(self.cfg.supernode_temperature), 1e-6)
        logits = jnp.where(
            valid[:, None, :].astype(jnp.bool_), logits, jnp.asarray(-1e9)
        )
        weights = jax.nn.softmax(logits, axis=-1)
        supernodes = jnp.einsum(
            'bmn,bnd->bmd', weights.astype(point_tokens.dtype), point_tokens
        )
        state_token = nn.Dense(int(self.cfg.d_model), name='state_projection')(
            state.astype(jnp.float32)
        )[:, None, :]
        tokens = jnp.concatenate([supernodes, state_token], axis=1)
        token_mask = jnp.ones(tokens.shape[:-1], dtype=jnp.bool_)
        if int(self.cfg.supernode_layers) > 0:
            tokens = SelfAttentionStack(
                self.cfg.transformer(),
                int(self.cfg.supernode_layers),
                name='supernode_refinement',
            )(tokens, mask=token_mask, train=train)
        return tokens, token_mask


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return float(jnp.log(jnp.expm1(value)))


class SpacetimeEventEncoder(nn.Module):
    """Compress local point-cloud segments into a small register set.

    Robot state and demonstrated actions are separate token types. They are not
    repeated across points before supernode pooling.
    """

    cfg: TTTEventEncoderConfig
    state_dim: int
    action_dim: int

    @nn.compact
    def __call__(
        self,
        xyz: jax.Array,
        time: jax.Array,
        state: jax.Array,
        action: jax.Array,
        valid: jax.Array,
        *,
        rgb: Optional[jax.Array] = None,
        mask_id: Optional[jax.Array] = None,
        train: bool = False,
        return_stats: bool = False,
    ):
        # xyz=[B,S,F,N,3], state/action=[B,S,F,D].
        if xyz.ndim != 5:
            raise ValueError(f'Expected xyz=[B,S,F,N,3], got {xyz.shape}.')
        batch, segments, frames, points = map(int, xyz.shape[:4])
        flat_batch = batch * segments
        point_count = frames * points
        encoder_cfg = self.cfg
        d_model = int(encoder_cfg.d_model)
        supernode_count = int(encoder_cfg.spacetime_supernodes)
        if supernode_count <= 0:
            raise ValueError('encoder.spacetime_supernodes must be positive.')

        flat_xyz = xyz.reshape(flat_batch, point_count, 3).astype(jnp.float32)
        flat_time = time.reshape(flat_batch, point_count).astype(jnp.float32)
        flat_valid = valid.reshape(flat_batch, point_count).astype(jnp.bool_)
        flat_rgb = (
            None
            if rgb is None
            else rgb.reshape(flat_batch, point_count, rgb.shape[-1])
        )
        flat_mask_id = (
            None
            if mask_id is None
            else mask_id.reshape(flat_batch, point_count)
        )
        point_tokens = _PointFeatureEmbed(encoder_cfg, name='point_embed')(
            flat_xyz, rgb=flat_rgb, mask_id=flat_mask_id
        )
        center_indices = _supernode_center_indices(
            valid=flat_valid,
            mask_id=flat_mask_id,
            num_centers=supernode_count,
            center_sampling=str(encoder_cfg.supernode_center_sampling),
        )
        centers_xyz = jnp.take_along_axis(
            flat_xyz, center_indices[:, :, None], axis=1
        )
        centers_time = jnp.take_along_axis(flat_time, center_indices, axis=1)
        xyz_distance = jnp.sum(
            jnp.square(centers_xyz[:, :, None, :] - flat_xyz[:, None, :, :]),
            axis=-1,
        )
        time_distance = jnp.square(
            centers_time[:, :, None] - flat_time[:, None, :]
        )
        if bool(self.cfg.learnable_bandwidths):
            raw_xyz = self.param(
                'raw_bandwidth_xyz',
                nn.initializers.constant(
                    _inverse_softplus(
                        float(encoder_cfg.spacetime_temperature_xyz)
                        - float(self.cfg.min_bandwidth)
                    )
                ),
                (),
            )
            raw_time = self.param(
                'raw_bandwidth_time',
                nn.initializers.constant(
                    _inverse_softplus(
                        float(encoder_cfg.spacetime_temperature_t)
                        - float(self.cfg.min_bandwidth)
                    )
                ),
                (),
            )
            bandwidth_xyz = jax.nn.softplus(raw_xyz) + float(self.cfg.min_bandwidth)
            bandwidth_time = jax.nn.softplus(raw_time) + float(self.cfg.min_bandwidth)
        else:
            bandwidth_xyz = jnp.asarray(
                encoder_cfg.spacetime_temperature_xyz, dtype=jnp.float32
            )
            bandwidth_time = jnp.asarray(
                encoder_cfg.spacetime_temperature_t, dtype=jnp.float32
            )
        logits = -(
            xyz_distance / jnp.maximum(bandwidth_xyz, 1e-8)
            + time_distance / jnp.maximum(bandwidth_time, 1e-8)
        )
        logits = jnp.where(
            flat_valid[:, None, :], logits, jnp.asarray(-1e9, dtype=logits.dtype)
        )
        pooling_weights = jax.nn.softmax(logits, axis=-1)
        supernodes = jnp.einsum(
            'bmp,bpd->bmd', pooling_weights.astype(point_tokens.dtype), point_tokens
        )

        point_assignments = jax.nn.softmax(logits, axis=1)
        point_assignments = point_assignments * flat_valid[:, None, :]
        valid_count = jnp.maximum(
            jnp.sum(flat_valid.astype(jnp.float32), axis=-1, keepdims=True), 1.0
        )
        occupancy = jnp.sum(point_assignments, axis=-1) / valid_count
        occupied_mask = occupancy > float(self.cfg.occupancy_threshold)
        center_features = jnp.concatenate(
            [
                centers_xyz,
                centers_time[..., None],
                jnp.log(occupancy[..., None] + 1e-6),
            ],
            axis=-1,
        )
        supernodes = supernodes + nn.Dense(d_model, name='center_feature_projection')(
            center_features
        )

        flat_state = state.reshape(flat_batch, frames, int(self.state_dim)).astype(
            jnp.float32
        )
        flat_action = action.reshape(flat_batch, frames, int(self.action_dim)).astype(
            jnp.float32
        )
        frame_valid = jnp.any(valid.astype(jnp.bool_), axis=-1).reshape(
            flat_batch, frames
        )
        frame_time = jnp.mean(time.astype(jnp.float32), axis=-1).reshape(
            flat_batch, frames, 1
        )
        state_tokens = nn.Dense(d_model, name='state_projection')(flat_state)
        action_tokens = nn.Dense(d_model, name='action_projection')(flat_action)
        temporal_tokens = nn.Dense(d_model, name='frame_time_projection')(frame_time)
        state_tokens = state_tokens + temporal_tokens
        action_tokens = action_tokens + temporal_tokens
        token_types = self.param(
            'token_types', nn.initializers.normal(stddev=0.02), (3, d_model)
        )
        supernodes = supernodes + token_types[0]
        state_tokens = state_tokens + token_types[1]
        action_tokens = action_tokens + token_types[2]
        tokens = jnp.concatenate([supernodes, state_tokens, action_tokens], axis=1)
        token_mask = jnp.concatenate(
            [occupied_mask, frame_valid, frame_valid], axis=1
        )
        if int(encoder_cfg.spacetime_layers) > 0:
            tokens = SelfAttentionStack(
                encoder_cfg.transformer(),
                int(encoder_cfg.spacetime_layers),
                name='event_refinement',
            )(tokens, mask=token_mask, train=train)
        registers = LatentPerceiver(
            encoder_cfg.perceiver(num_latents=int(self.cfg.support_registers)),
            name='event_registers',
        )(tokens, token_mask=token_mask, train=train)
        registers = registers.reshape(
            batch, segments, int(self.cfg.support_registers), d_model
        )
        register_mask = jnp.ones(registers.shape[:-1], dtype=jnp.bool_)
        if not return_stats:
            return registers, register_mask

        assignment_probability = point_assignments / jnp.maximum(
            jnp.sum(point_assignments, axis=1, keepdims=True), 1e-8
        )
        assignment_entropy = -jnp.sum(
            jnp.where(
                assignment_probability > 0.0,
                assignment_probability
                * jnp.log(assignment_probability + 1e-8),
                0.0,
            ),
            axis=1,
        ) / jnp.log(jnp.asarray(max(2, supernode_count), dtype=jnp.float32))
        assignment_entropy = jnp.sum(
            assignment_entropy * flat_valid.astype(jnp.float32)
        ) / jnp.maximum(jnp.sum(flat_valid), 1.0)
        occupancy_distribution = occupancy / jnp.maximum(
            jnp.sum(occupancy, axis=-1, keepdims=True), 1e-8
        )
        occupancy_entropy = -jnp.sum(
            occupancy_distribution * jnp.log(occupancy_distribution + 1e-8),
            axis=-1,
        )
        effective_supernodes = jnp.exp(occupancy_entropy)
        stats = {
            'supernode_assignment_entropy': assignment_entropy,
            'supernode_occupied_count': jnp.mean(jnp.sum(occupied_mask, axis=-1)),
            'supernode_effective_count': jnp.mean(effective_supernodes),
            'supernode_occupancy_min': jnp.mean(jnp.min(occupancy, axis=-1)),
            'supernode_occupancy_max': jnp.mean(jnp.max(occupancy, axis=-1)),
            'spacetime_bandwidth_xyz': bandwidth_xyz,
            'spacetime_bandwidth_time': bandwidth_time,
        }
        return registers, register_mask, stats


class RLBenchTTTFeatureEncoder(nn.Module):
    """Visual bridge for the gated RLBench phase.

    Support produces sequential event registers. Query encoding accepts no
    demonstrated action and therefore cannot leak the unknown query target.
    """

    cfg: TTTEventEncoderConfig
    state_dim: int
    action_dim: int

    def setup(self):
        self.support_events = SpacetimeEventEncoder(
            self.cfg, self.state_dim, self.action_dim, name='support_events'
        )
        self.query_frames = _SupernodeFrameTokenizer(self.cfg, name='query_frames')
        self.query_registers = LatentPerceiver(
            self.cfg.perceiver(num_latents=int(self.cfg.query_registers)),
            name='query_registers',
        )

    def encode_support(
        self,
        support: Dict[str, jax.Array],
        *,
        train: bool = False,
        return_stats: bool = False,
    ):
        return self.support_events(
            support['xyz'],
            support['time'],
            support['state'],
            support['action'],
            support['valid'],
            rgb=support.get('rgb'),
            mask_id=support.get('mask_id'),
            train=train,
            return_stats=return_stats,
        )

    def encode_query(
        self, query: Dict[str, jax.Array], *, train: bool = False
    ) -> Tuple[jax.Array, jax.Array]:
        # query xyz=[B,T,N,3]; no query action is accepted.
        xyz = query['xyz']
        batch, frames = int(xyz.shape[0]), int(xyz.shape[1])
        flat_xyz = xyz.reshape(batch * frames, xyz.shape[-2], 3)
        flat_state = query['state'].reshape(batch * frames, int(self.state_dim))
        flat_valid = query['valid'].reshape(batch * frames, query['valid'].shape[-1])
        rgb = query.get('rgb')
        mask_id = query.get('mask_id')
        tokens, token_mask = self.query_frames(
            flat_xyz,
            flat_state,
            flat_valid,
            rgb=None if rgb is None else rgb.reshape(batch * frames, rgb.shape[-2], rgb.shape[-1]),
            mask_id=None
            if mask_id is None
            else mask_id.reshape(batch * frames, mask_id.shape[-1]),
            train=train,
        )
        tokens = tokens.reshape(batch, frames * tokens.shape[1], tokens.shape[2])
        token_mask = token_mask.reshape(batch, frames * token_mask.shape[1])
        registers = self.query_registers(
            tokens, token_mask=token_mask, train=train
        )
        return registers, jnp.ones(registers.shape[:-1], dtype=jnp.bool_)

    def __call__(
        self,
        support: Dict[str, jax.Array],
        query: Dict[str, jax.Array],
        *,
        train: bool = False,
    ):
        support_registers, support_mask = self.encode_support(support, train=train)
        query_registers, query_mask = self.encode_query(query, train=train)
        return support_registers, support_mask, query_registers, query_mask
