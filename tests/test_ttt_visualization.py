from __future__ import annotations

import json

import jax
import numpy as np
import optax

from icil_jax_rlbench.configs.ttt_state_base import get_config as train_config
from icil_jax_rlbench.configs.visualize_ttt_state import (
    get_config as visualization_config,
)
from icil_jax_rlbench.data.hidden_goal import (
    HiddenGoalTaskBank,
    fit_state_normalizer,
)
from icil_jax_rlbench.eval.ttt_state_common import (
    bootstrap_mean_confidence_interval,
    confidence_interval,
)
from icil_jax_rlbench.models.fast_weight_ttt import init_fast_weight_ttt_params
from icil_jax_rlbench.train.checkpoints import save_checkpoint
from icil_jax_rlbench.train.ttt_runner import (
    fast_weight_config_from,
    hidden_goal_config_from,
)
from icil_jax_rlbench.train.ttt_step import create_ttt_train_state
from icil_jax_rlbench.visualization.ttt_state import visualize_ttt_state


def _tiny_checkpoint(path):
    cfg = train_config()
    cfg.benchmark.num_train_tasks = 4
    cfg.benchmark.num_validation_tasks = 2
    cfg.benchmark.num_test_tasks = 2
    cfg.benchmark.horizon = 4
    cfg.benchmark.support_episodes = 1
    cfg.benchmark.query_episodes = 1
    cfg.benchmark.normalizer_episodes_per_task = 1
    cfg.model.hidden_dim = 12
    cfg.model.fast_dim = 6
    cfg.model.fast_hidden_dim = 8
    cfg.model.gate_init = 0.1
    cfg.adaptation.write_segment_size = 2

    benchmark_cfg = hidden_goal_config_from(cfg)
    model_cfg = fast_weight_config_from(cfg, benchmark_cfg)
    task_bank = HiddenGoalTaskBank(benchmark_cfg)
    normalizer = fit_state_normalizer(
        benchmark_cfg,
        task_bank,
        episodes_per_task=1,
    )
    params = init_fast_weight_ttt_params(jax.random.key(0), model_cfg)
    optimizer = optax.adam(1e-3)
    state = create_ttt_train_state(params, optimizer, jax.random.key(1))
    save_checkpoint(
        path,
        state=state,
        step=0,
        config=cfg,
        extra={
            'checkpoint_type': 'fast_weight_ttt_state',
            'normalizer': normalizer.to_dict(),
            'transient_fast_state_saved': False,
        },
        replicated=False,
    )


def test_bootstrap_interval_is_deterministic_and_contains_mean():
    values = np.asarray([-0.1, 0.2, 0.4, 0.5], dtype=np.float32)
    first = bootstrap_mean_confidence_interval(
        values, seed=7, bootstrap_samples=1_000
    )
    second = bootstrap_mean_confidence_interval(
        values, seed=7, bootstrap_samples=1_000
    )
    assert first == second
    mean, low, high = first
    assert low <= mean <= high


def test_pooled_success_interval_handles_boundary_counts():
    zero_low, zero_high = confidence_interval(0, 20)
    full_low, full_high = confidence_interval(20, 20)
    assert zero_low == 0.0
    assert zero_high > 0.0
    assert full_low < 1.0
    assert full_high == 1.0


def test_visualizer_writes_matched_trajectory_artifacts(tmp_path):
    checkpoint_path = tmp_path / 'last.pkl'
    _tiny_checkpoint(checkpoint_path)
    cfg = visualization_config()
    cfg.checkpoint_path = str(checkpoint_path)
    cfg.output_dir = str(tmp_path / 'visualizations')
    cfg.task_ids = (0,)
    cfg.support_count = 1
    cfg.query_episodes = 2
    cfg.conditions = (
        'no_update',
        'correct_support',
        'wrong_task_support',
    )
    cfg.vector_field_grid_size = 5
    cfg.figure_dpi = 40
    cfg.write_video = False

    run_dir = visualize_ttt_state(cfg)
    task_dir = run_dir / 'task-000'
    for filename in (
        'support_trajectories.png',
        'matched_trajectories.png',
        'action_changes.png',
        'write_diagnostics.png',
        'fast_tensor_deltas.png',
        'vector_fields.png',
        'trajectory_data.npz',
        'manifest.json',
    ):
        assert (task_dir / filename).stat().st_size > 0

    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['matched_query_episode_ids_across_conditions'] is True
    assert manifest['privileged_goal_usage'] == 'visualization_only'
    condition_summaries = manifest['tasks'][0]['conditions']
    baseline_ids = condition_summaries['no_update']['query_episode_ids']
    assert condition_summaries['correct_support']['query_episode_ids'] == baseline_ids
    assert (
        condition_summaries['wrong_task_support']['query_episode_ids']
        == baseline_ids
    )

    with np.load(task_dir / 'trajectory_data.npz') as arrays:
        np.testing.assert_array_equal(
            arrays['no_update_episode_ids'],
            arrays['correct_support_episode_ids'],
        )
        assert arrays['correct_support_reference_actions'].shape == (2, 4, 3)
