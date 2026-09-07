from __future__ import annotations

import os

from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()
    cfg.integration = 'metaworld_ml10'
    cfg.checkpoint_path = ''
    cfg.cache_root = os.environ.get('PHI_MUJOCO_ML10_CACHE', '')
    cfg.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_OUTPUT_DIR', 'eval_outputs'),
        'metaworld_ml10_update_information',
    )
    cfg.seed = 0
    cfg.support_counts = (1, 2)
    cfg.query_episodes = 2
    cfg.samples_per_task = 2
    cfg.probe_observations = 64
    cfg.max_train_tasks = 0
    cfg.max_latent_tasks = 0
    cfg.max_family_tasks = 40
    cfg.conditions = (
        'correct_support',
        'same_family_wrong_instance',
        'different_family_support',
        'shuffled_actions',
        'shuffled_time',
        'observations_only',
        'actions_only',
    )
    cfg.ridge = 1e-2
    cfg.cache_prepared_episodes = True
    cfg.progress_every = 20
    cfg.visualization = ConfigDict()
    cfg.visualization.enabled = False
    cfg.visualization.condition = 'correct_support'
    cfg.visualization.support_count = 2
    cfg.visualization.representations = (
        'first_write_gradient',
        'final_fast_delta',
        'read_action_delta',
    )
    cfg.visualization.splits = (
        'train',
        'latent_validation',
        'family_validation',
    )
    cfg.visualization.pca_components = 50
    cfg.visualization.perplexities = (30.0, 80.0)
    cfg.visualization.max_iter = 1_000
    cfg.visualization.seed = 0
    return cfg


__all__ = ['get_config']
