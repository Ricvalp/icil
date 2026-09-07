from __future__ import annotations

import numpy as np

from icil_jax_rlbench.data.hidden_goal import (
    HiddenGoalConfig,
    HiddenGoalMetaSampler,
    HiddenGoalTaskBank,
    benchmark_integrity_report,
    fit_state_normalizer,
)


def _benchmark():
    cfg = HiddenGoalConfig(
        num_train_tasks=8,
        num_validation_tasks=3,
        num_test_tasks=3,
        horizon=16,
        support_episodes=2,
        query_episodes=2,
    )
    task_bank = HiddenGoalTaskBank(cfg)
    normalizer = fit_state_normalizer(cfg, task_bank, episodes_per_task=1)
    return cfg, task_bank, normalizer


def test_task_splits_and_episode_axes_are_explicit_and_disjoint():
    cfg, task_bank, normalizer = _benchmark()
    train = task_bank.goals('train')
    validation = task_bank.goals('validation')
    test = task_bank.goals('test')
    assert not np.any(np.all(train[:, None] == validation[None], axis=-1))
    assert not np.any(np.all(train[:, None] == test[None], axis=-1))

    batch = HiddenGoalMetaSampler(
        cfg, task_bank, normalizer, split='train', seed=13
    ).build_batch(4)
    assert batch['support']['observation'].shape == (4, 2, 16, 4)
    assert batch['query']['observation'].shape == (4, 2, 16, 4)
    assert batch['meta']['task_latent'].shape == (4, 2)
    assert not np.any(batch['support']['outer_loss_mask'])
    assert np.all(batch['query']['outer_loss_mask'])
    for task_index in range(4):
        assert not np.intersect1d(
            batch['support']['episode_id'][task_index],
            batch['query']['episode_id'][task_index],
        ).size


def test_benchmark_integrity_has_oracle_and_no_obvious_query_leakage():
    cfg, task_bank, normalizer = _benchmark()
    report = benchmark_integrity_report(
        cfg, task_bank, normalizer, samples=512, seed=19
    )
    assert report['oracle_expert_success_rate'] == 1.0
    assert report['support_endpoint_goal_mse'] < 1e-6
    assert report['query_only_initial_state_linear_r2'] < 0.05
    assert not report['support_query_episode_overlap']
    assert not report['support_query_identical_initial_state']
    assert not report['support_outer_loss_mask_nonzero']


def test_normalizer_is_reproducible_and_fit_only_from_train_split():
    cfg, task_bank, normalizer_a = _benchmark()
    normalizer_b = fit_state_normalizer(cfg, task_bank, episodes_per_task=1)
    assert normalizer_a.identifier == normalizer_b.identifier
    np.testing.assert_array_equal(
        normalizer_a.observation_mean, normalizer_b.observation_mean
    )
    np.testing.assert_array_equal(
        normalizer_a.observation_std, normalizer_b.observation_std
    )
    assert normalizer_a.identifier.startswith('hidden_goal_train_')
