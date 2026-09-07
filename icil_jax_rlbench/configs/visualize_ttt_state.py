from __future__ import annotations

import os

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.checkpoint_path = ''
    cfg.evaluation_summary_path = ''
    cfg.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_OUTPUT_DIR', 'eval_outputs'),
        'ttt_state_visualizations',
    )
    cfg.seed = 0
    cfg.split = 'validation'
    cfg.task_ids = (0, 1, 2)
    cfg.support_count = 2
    cfg.query_episodes = 3
    cfg.conditions = (
        'no_update',
        'correct_support',
        'wrong_task_support',
        'shuffled_actions',
        'shuffled_time',
        'random_update_matched_norm',
    )
    cfg.vector_field_grid_size = 13
    cfg.vector_field_phase = 0.0
    cfg.figure_dpi = 160
    cfg.write_video = True
    cfg.video_fps = 4.0
    return cfg
