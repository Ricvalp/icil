from __future__ import annotations

import os

from icil_jax_rlbench.configs.metaworld_ml10_query_only import get_config as _base


def get_config():
    cfg = _base()
    cfg.dataset.protocol = 'final'
    cfg.dataset.validation_split = 'train'
    cfg.train.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_RUN_ROOT', 'outputs'),
        'metaworld_ml10_query_only_final',
    )
    return cfg


__all__ = ['get_config']
