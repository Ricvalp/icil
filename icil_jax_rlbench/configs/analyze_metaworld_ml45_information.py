from __future__ import annotations

import os

from icil_jax_rlbench.configs.analyze_metaworld_ml10_information import (
    get_config as _base,
)


def get_config():
    cfg = _base()
    cfg.integration = 'metaworld_ml45'
    cfg.cache_root = os.environ.get('PHI_MUJOCO_ML45_CACHE', '')
    cfg.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_OUTPUT_DIR', 'eval_outputs'),
        'metaworld_ml45_update_information',
    )

    # Ten instances per family preserve broad family coverage while keeping the
    # extraction comparable in size to the existing ML10 diagnostic.
    cfg.support_counts = (2,)
    cfg.max_train_tasks = 400
    cfg.max_latent_tasks = 400
    cfg.max_family_tasks = 50

    # Two independent support/query assignments give each task instance two
    # points, so same-instance update stability can be measured and visualized.
    cfg.samples_per_task = 2
    cfg.visualization.enabled = True
    cfg.visualization.support_count = 2
    return cfg


__all__ = ['get_config']
