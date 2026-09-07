from __future__ import annotations

import os

from icil_jax_rlbench.configs.eval_metaworld_ml1_reach_ttt import (
    get_config as _base,
)


def get_config():
    cfg = _base()
    cfg.integration = 'metaworld_ml45'
    cfg.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_OUTPUT_DIR', 'eval_outputs'),
        'metaworld_ml45_ttt',
    )
    cfg.split = 'family_validation'
    cfg.max_tasks = 10
    cfg.support_counts = (2,)
    cfg.offline_query_episodes = 2
    cfg.conditions = (
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
    cfg.closed_loop_episodes = 3
    return cfg


__all__ = ['get_config']
