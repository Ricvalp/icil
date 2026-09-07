from __future__ import annotations

import os

from ml_collections import ConfigDict

from icil_jax_rlbench.configs.metaworld_ml1_reach_query_only import (
    get_config as _query_only_config,
)


def get_config() -> ConfigDict:
    cfg = _query_only_config()
    cfg.mode = 'metaworld_ml1_reach_ttt'

    cfg.conditioning = ConfigDict()
    cfg.conditioning.support_token_cross_attention = False
    cfg.conditioning.support_summary_film = False
    cfg.conditioning.support_trajectory_tokens = False
    cfg.conditioning.support_memory_initialization = False
    cfg.conditioning.fast_weight_write = True
    cfg.conditioning.fast_weight_read = True
    cfg.conditioning.query_history = False

    cfg.adaptation = ConfigDict()
    cfg.adaptation.write_objective = 'kvb'
    cfg.adaptation.read_objective = 'robotics_action_imitation'
    cfg.adaptation.read_mode = 'absolute_gated'
    cfg.adaptation.read_scale = 1.0
    cfg.adaptation.write_segment_size = 16
    cfg.adaptation.write_steps_per_segment = 1
    cfg.adaptation.first_order = False
    cfg.adaptation.fast_grad_clip_norm = 1.0
    cfg.adaptation.fast_update_clip_norm = 0.0
    cfg.adaptation.fast_drift_weight = 0.0
    cfg.adaptation.outer_fast_drift_weight = 0.0
    cfg.adaptation.reset_policy = 'reset_to_meta_learned_w0_per_task'
    cfg.adaptation.query_carry_policy = (
        'freeze_after_support_and_reuse_across_query_episodes'
    )

    cfg.train.initial_checkpoint_path = os.environ.get(
        'ICIL_ML1_REACH_QUERY_CHECKPOINT', ''
    )
    cfg.train.support_episodes_per_task = 2
    cfg.train.query_episodes_per_task = 2
    cfg.train.batch_size = 4
    cfg.train.eager_debug = False
    cfg.train.debug_max_time_steps = 0
    cfg.train.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_RUN_ROOT', 'outputs'),
        'metaworld_ml1_reach_ttt',
    )

    cfg.evaluation = ConfigDict()
    cfg.evaluation.support_counts = (1, 2, 4)
    cfg.evaluation.conditions = (
        'no_update',
        'correct_support',
        'wrong_task_support',
        'shuffled_actions',
        'shuffled_time',
        'observations_only',
        'actions_only',
        'duplicated_support',
        'random_update_matched_norm',
    )
    return cfg


__all__ = ['get_config']
