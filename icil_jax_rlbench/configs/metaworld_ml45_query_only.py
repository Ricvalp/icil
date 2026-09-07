from __future__ import annotations

import os

from icil_jax_rlbench.configs.metaworld_ml1_reach_query_only import (
    get_config as _base,
)


def get_config():
    cfg = _base()
    cfg.mode = 'metaworld_ml45_query_only'
    cfg.dataset.integration = 'metaworld_ml45'
    cfg.dataset.cache_root = os.environ.get('PHI_MUJOCO_ML45_CACHE', '')
    cfg.dataset.protocol = 'development'
    cfg.dataset.train_split = 'train'
    cfg.dataset.validation_split = 'latent_validation'
    cfg.dataset.horizon_buckets = (64, 128, 256, 512)
    cfg.train.num_steps = 100_000
    cfg.train.batch_size = 32
    cfg.train.query_episodes_per_task = 2
    cfg.train.eval_every = 1_000
    cfg.train.ckpt_every = 5_000
    cfg.train.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_RUN_ROOT', 'outputs'),
        'metaworld_ml45_query_only',
    )
    return cfg


__all__ = ['get_config']

