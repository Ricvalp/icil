from __future__ import annotations

import os

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.checkpoint_path = ''
    cfg.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_OUTPUT_DIR', 'eval_outputs'),
        'ttt_state_gate1',
    )
    cfg.seed = 0
    cfg.split = 'validation'
    cfg.support_episodes = 2
    cfg.query_episodes_per_task = 20
    cfg.inner_steps = 100
    cfg.inner_lr = 1e-2
    cfg.inner_grad_clip_norm = 1.0
    cfg.adapt_subset = 'action_heads'  # action_heads | query_policy | all
    cfg.conditions = ('no_update', 'correct_support', 'wrong_task_support', 'shuffled_actions')
    cfg.max_tasks = 0
    return cfg
