from __future__ import annotations

import os

from icil_jax_rlbench.configs.metaworld_ml1_reach_ttt_base import (
    get_config as _base,
)


def get_config():
    cfg = _base()
    cfg.mode = 'metaworld_ml10_ttt'
    cfg.dataset.integration = 'metaworld_ml10'
    cfg.dataset.cache_root = os.environ.get('PHI_MUJOCO_ML10_CACHE', '')
    cfg.dataset.protocol = 'development'
    cfg.dataset.train_split = 'train'
    cfg.dataset.validation_split = 'latent_validation'
    cfg.dataset.horizon_buckets = (64, 128, 256, 512)
    cfg.train.initial_checkpoint_path = os.environ.get(
        'ICIL_ML10_QUERY_CHECKPOINT', ''
    )
    cfg.train.batch_size = 8
    cfg.train.support_episodes_per_task = 2
    cfg.train.query_episodes_per_task = 2
    cfg.train.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_RUN_ROOT', 'outputs'),
        'metaworld_ml10_ttt',
    )
    cfg.evaluation.conditions = (
        'no_update',
        'correct_support',
        'same_family_wrong_instance',
        'different_family_support',
        'shuffled_actions',
        'shuffled_time',
        'observations_only',
        'actions_only',
        'duplicated_support',
        'random_update_matched_norm',
    )
    return cfg


__all__ = ['get_config']
