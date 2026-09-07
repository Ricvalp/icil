from icil_jax_rlbench.configs.ttt_state_base import get_config as _base


def get_config():
    cfg = _base()
    cfg.benchmark.num_train_tasks = 4
    cfg.benchmark.num_validation_tasks = 2
    cfg.benchmark.num_test_tasks = 2
    cfg.benchmark.horizon = 8
    cfg.benchmark.support_episodes = 1
    cfg.benchmark.query_episodes = 1
    cfg.benchmark.normalizer_episodes_per_task = 2
    cfg.model.hidden_dim = 32
    cfg.model.fast_dim = 16
    cfg.model.fast_hidden_dim = 24
    cfg.adaptation.write_segment_size = 4
    cfg.train.batch_size = 4
    cfg.train.lr = 1e-3

    cfg.diagnostic = type(cfg)()
    cfg.diagnostic.steps = 500
    cfg.diagnostic.seed = 17
    cfg.diagnostic.finite_difference_epsilon = 5e-3
    cfg.diagnostic.expected_relative_loss_reduction = 0.80
    cfg.diagnostic.output_dir = 'outputs/ttt_state_gate2'
    return cfg
