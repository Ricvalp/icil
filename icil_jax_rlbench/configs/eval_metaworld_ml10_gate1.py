from __future__ import annotations

import os

from icil_jax_rlbench.configs.eval_metaworld_ml1_reach_gate1 import (
    get_config as _base,
)


def get_config():
    cfg = _base()
    cfg.integration = 'metaworld_ml10'
    cfg.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_OUTPUT_DIR', 'eval_outputs'),
        'metaworld_ml10_gate1',
    )
    cfg.split = 'family_validation'
    cfg.max_tasks = 10
    cfg.support_episodes = 2
    cfg.offline_query_episodes = 2
    cfg.closed_loop_episodes = 5
    return cfg


__all__ = ['get_config']
