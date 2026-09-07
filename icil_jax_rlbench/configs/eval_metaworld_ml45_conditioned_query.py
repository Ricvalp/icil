from __future__ import annotations

import os

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.integration = 'metaworld_ml45'
    cfg.checkpoint_path = ''
    cfg.cache_root = ''
    cfg.cache_prepared_episodes = True
    cfg.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_OUTPUT_DIR', 'eval_outputs'),
        'metaworld_ml45_conditioned_query',
    )
    cfg.seed = 0
    cfg.split = 'latent_validation'
    cfg.max_tasks = 40
    cfg.allow_unseen_families = False
    cfg.offline_query_episodes = 2
    cfg.closed_loop_episodes = 3
    cfg.closed_loop_base_seed = 6_000_000
    cfg.closed_loop_max_steps = 0
    cfg.save_rollout_artifacts = False
    cfg.record_video = False
    cfg.render_width = 320
    cfg.render_height = 240
    return cfg


__all__ = ['get_config']
