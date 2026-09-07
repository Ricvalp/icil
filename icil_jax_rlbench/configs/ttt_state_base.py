from __future__ import annotations

import os

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.mode = 'ttt_adaptation_only'

    cfg.conditioning = ConfigDict()
    cfg.conditioning.support_token_cross_attention = False
    cfg.conditioning.support_summary_film = False
    cfg.conditioning.support_trajectory_tokens = False
    cfg.conditioning.support_memory_initialization = False
    cfg.conditioning.fast_weight_write = True
    cfg.conditioning.fast_weight_read = True
    cfg.conditioning.query_history = False

    cfg.benchmark = ConfigDict()
    cfg.benchmark.seed = 0
    cfg.benchmark.num_train_tasks = 48
    cfg.benchmark.num_validation_tasks = 12
    cfg.benchmark.num_test_tasks = 12
    cfg.benchmark.horizon = 16
    cfg.benchmark.support_episodes = 2
    cfg.benchmark.query_episodes = 2
    cfg.benchmark.world_limit = 1.0
    cfg.benchmark.goal_limit = 0.65
    cfg.benchmark.initial_limit = 0.85
    cfg.benchmark.max_translation = 0.16
    cfg.benchmark.expert_gain = 0.8
    cfg.benchmark.close_radius = 0.10
    cfg.benchmark.success_radius = 0.11
    cfg.benchmark.expert_action_noise = 0.0
    cfg.benchmark.min_goal_separation = 0.10
    cfg.benchmark.normalizer_episodes_per_task = 4

    cfg.model = ConfigDict()
    cfg.model.hidden_dim = 64
    cfg.model.fast_dim = 32
    cfg.model.fast_hidden_dim = 64
    cfg.model.fast_model = 'mlp'
    cfg.model.gate_init = 1e-3
    cfg.model.inner_lr_init = 3e-2
    cfg.model.inner_lr_min = 1e-5

    cfg.action = ConfigDict()
    cfg.action.representation = 'normalized_planar_delta_plus_gripper'
    cfg.action.translation_loss = 'huber'
    cfg.action.translation_loss_weight = 1.0
    cfg.action.translation_huber_delta = 0.1
    cfg.action.gripper_loss = 'binary_cross_entropy'
    cfg.action.gripper_loss_weight = 0.25

    cfg.adaptation = ConfigDict()
    cfg.adaptation.write_objective = 'kvb'
    cfg.adaptation.read_objective = 'robotics_action_imitation'
    cfg.adaptation.read_mode = 'absolute_gated'
    cfg.adaptation.read_scale = 1.0
    cfg.adaptation.write_segment_size = 4
    cfg.adaptation.write_steps_per_segment = 1
    cfg.adaptation.first_order = False
    cfg.adaptation.fast_grad_clip_norm = 1.0
    cfg.adaptation.fast_update_clip_norm = 0.0
    cfg.adaptation.fast_drift_weight = 0.0
    cfg.adaptation.outer_fast_drift_weight = 0.0
    cfg.adaptation.reset_policy = 'reset_to_meta_learned_w0_per_task'
    cfg.adaptation.query_carry_policy = 'freeze_after_support_and_reuse_across_query_episodes'

    cfg.train = ConfigDict()
    cfg.train.seed = 0
    cfg.train.num_steps = 20_000
    cfg.train.batch_size = 32
    cfg.train.lr = 3e-4
    cfg.train.weight_decay = 1e-5
    cfg.train.slow_grad_clip_norm = 1.0
    cfg.train.distributed = False
    cfg.train.log_every = 50
    cfg.train.eval_every = 500
    cfg.train.eval_batches = 8
    cfg.train.ckpt_every = 2_000
    cfg.train.resume_path = ''
    cfg.train.output_dir = os.path.join(
        os.environ.get('ICIL_JAX_RLBENCH_RUN_ROOT', 'outputs'), 'ttt_state'
    )

    cfg.evaluation = ConfigDict()
    cfg.evaluation.split = 'test'
    cfg.evaluation.query_episodes_per_task = 20
    cfg.evaluation.support_counts = (1, 2, 4)
    cfg.evaluation.conditions = (
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

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get(
        'ICIL_JAX_RLBENCH_WANDB_PROJECT', 'icil-jax-rlbench'
    )
    cfg.wandb.entity = os.environ.get('WANDB_ENTITY', '')
    cfg.wandb.mode = os.environ.get('WANDB_MODE', 'online')
    cfg.wandb.name = ''
    return cfg
