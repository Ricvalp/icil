from __future__ import annotations

import os

from icil_jax_rlbench.configs.metaworld_ml1_reach_query_only import (
    get_config as _reach_config,
)


def get_config():
    cfg = _reach_config()
    cfg.mode = 'metaworld_ml1_push_query_only'
    cfg.dataset.integration = 'metaworld_ml1_push'
    cfg.dataset.cache_root = os.environ.get('PHI_MUJOCO_ML1_PUSH_CACHE', '')
    cfg.train.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_RUN_ROOT', 'outputs'),
        'metaworld_ml1_push_query_only',
    )
    return cfg


__all__ = ['get_config']
