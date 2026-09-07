from __future__ import annotations

import os

from icil_jax_rlbench.configs.eval_metaworld_ml1_reach_ttt import (
    get_config as _reach_config,
)


def get_config():
    cfg = _reach_config()
    cfg.integration = 'metaworld_ml1_push'
    cfg.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_OUTPUT_DIR', 'eval_outputs'),
        'metaworld_ml1_push_ttt',
    )
    return cfg


__all__ = ['get_config']
