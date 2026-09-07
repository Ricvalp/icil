from __future__ import annotations

import os

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.integration = 'metaworld_ml1_reach'
    cfg.checkpoint_path = ''
    cfg.cache_root = ''
    cfg.cache_prepared_episodes = True
    cfg.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_OUTPUT_DIR', 'eval_outputs'),
        'metaworld_ml1_reach_ttt',
    )
    cfg.seed = 0
    cfg.split = 'validation'
    cfg.max_tasks = 0

    cfg.support_counts = (1, 2, 4)
    cfg.offline_query_episodes = 4
    cfg.write_steps_per_segment_override = 0
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

    cfg.closed_loop_episodes = 20
    cfg.closed_loop_base_seed = 3_000_000
    cfg.closed_loop_max_steps = 0
    cfg.save_rollout_artifacts = False
    cfg.record_video = False
    cfg.render_width = 320
    cfg.render_height = 240
    return cfg


__all__ = ['get_config']
