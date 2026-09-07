from __future__ import annotations

import os

from ml_collections import ConfigDict

from icil_jax_rlbench.configs.metaworld_ml45_query_only import get_config as _base


def get_config() -> ConfigDict:
    cfg = _base()
    cfg.mode = 'metaworld_ml45_family_conditioned_query_only'
    cfg.conditioning = ConfigDict()
    cfg.conditioning.mode = 'family'
    cfg.conditioning.normalization_eps = 1e-4
    cfg.train.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_RUN_ROOT', 'outputs'),
        'metaworld_ml45_family_conditioned_query_only',
    )
    return cfg


__all__ = ['get_config']
