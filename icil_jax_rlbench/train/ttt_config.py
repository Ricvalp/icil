from __future__ import annotations

from ml_collections import ConfigDict

from icil_jax_rlbench.models.fast_weight_ttt import TTTAdaptConfig
from icil_jax_rlbench.train.ttt_step import TTTStepConfig


def adaptation_config_from(cfg: ConfigDict) -> TTTAdaptConfig:
    return TTTAdaptConfig(
        write_objective=str(cfg.adaptation.write_objective),
        write_segment_size=int(cfg.adaptation.write_segment_size),
        write_steps_per_segment=int(cfg.adaptation.write_steps_per_segment),
        first_order=bool(cfg.adaptation.first_order),
        fast_grad_clip_norm=float(cfg.adaptation.fast_grad_clip_norm),
        fast_update_clip_norm=float(cfg.adaptation.fast_update_clip_norm),
        fast_drift_weight=float(cfg.adaptation.fast_drift_weight),
        write_enabled=bool(cfg.conditioning.fast_weight_write),
        read_enabled=bool(cfg.conditioning.fast_weight_read),
        read_mode=str(cfg.adaptation.get('read_mode', 'absolute_gated')),
        read_scale=float(cfg.adaptation.get('read_scale', 1.0)),
    )


def step_config_from(cfg: ConfigDict) -> TTTStepConfig:
    return TTTStepConfig(
        slow_grad_clip_norm=float(cfg.train.slow_grad_clip_norm),
        outer_fast_drift_weight=float(cfg.adaptation.outer_fast_drift_weight),
    )


__all__ = ['adaptation_config_from', 'step_config_from']
