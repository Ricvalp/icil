from __future__ import annotations

import os

from icil_jax_rlbench.configs.metaworld_ml10_kvb_delta_read import (
    get_config as _base,
)


def get_config():
    cfg = _base()
    cfg.dataset.cache_prepared_episodes = False
    cfg.train.num_steps = 2
    cfg.train.batch_size = 1
    cfg.train.support_episodes_per_task = 1
    cfg.train.query_episodes_per_task = 1
    cfg.train.eager_debug = True
    cfg.train.debug_max_time_steps = 32
    cfg.train.log_every = 1
    cfg.train.eval_every = 2
    cfg.train.eval_batches = 1
    cfg.train.ckpt_every = 2
    cfg.train.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_RUN_ROOT', 'outputs'),
        'debug',
        'metaworld_ml10_kvb',
    )
    cfg.wandb.enable = False
    return cfg


__all__ = ['get_config']
