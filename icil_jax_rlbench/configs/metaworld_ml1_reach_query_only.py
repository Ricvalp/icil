from __future__ import annotations

import os

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.mode = 'metaworld_ml1_reach_query_only'

    cfg.dataset = ConfigDict()
    cfg.dataset.integration = 'metaworld_ml1_reach'
    cfg.dataset.cache_root = os.environ.get('PHI_MUJOCO_ML1_REACH_CACHE', '')
    cfg.dataset.normalization_eps = 1e-4
    cfg.dataset.cache_prepared_episodes = True

    cfg.model = ConfigDict()
    cfg.model.hidden_dim = 128
    cfg.model.fast_dim = 32
    cfg.model.fast_hidden_dim = 64
    cfg.model.fast_model = 'mlp'
    cfg.model.gate_init = 1e-3
    cfg.model.inner_lr_init = 3e-2
    cfg.model.inner_lr_min = 1e-5

    cfg.action = ConfigDict()
    cfg.action.representation = 'standardized_cartesian_delta_plus_continuous_gripper'
    cfg.action.translation_loss = 'huber'
    cfg.action.translation_loss_weight = 1.0
    cfg.action.translation_huber_delta = 1.0
    cfg.action.gripper_loss = 'huber'
    cfg.action.gripper_loss_weight = 0.1
    cfg.action.gripper_huber_delta = 1.0

    cfg.train = ConfigDict()
    cfg.train.seed = 0
    cfg.train.num_steps = 20_000
    cfg.train.batch_size = 16
    cfg.train.query_episodes_per_task = 2
    cfg.train.lr = 3e-4
    cfg.train.weight_decay = 1e-5
    cfg.train.slow_grad_clip_norm = 1.0
    cfg.train.log_every = 50
    cfg.train.eval_every = 500
    cfg.train.eval_batches = 8
    cfg.train.ckpt_every = 2_000
    cfg.train.resume_path = ''
    cfg.train.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_RUN_ROOT', 'outputs'),
        'metaworld_ml1_reach_query_only',
    )

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get(
        'ICIL_JAX_RLBENCH_WANDB_PROJECT', 'icil-jax-rlbench'
    )
    cfg.wandb.entity = os.environ.get('WANDB_ENTITY', '')
    cfg.wandb.mode = os.environ.get('WANDB_MODE', 'online')
    cfg.wandb.name = ''
    return cfg
