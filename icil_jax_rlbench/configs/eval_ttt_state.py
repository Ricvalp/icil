from __future__ import annotations

import os

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.checkpoint_path = ''
    cfg.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_OUTPUT_DIR', 'eval_outputs'),
        'ttt_state',
    )
    cfg.seed = 0
    cfg.split = 'test'
    cfg.query_episodes_per_task = 20
    cfg.support_counts = (1, 2, 4)
    cfg.conditions = (
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
    cfg.max_tasks = 0
    return cfg
