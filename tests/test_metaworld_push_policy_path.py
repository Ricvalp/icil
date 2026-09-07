from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ml_collections import ConfigDict
from phi_mujoco.dataset import EpisodeData, write_processed_bundle
from phi_mujoco.integrations.metaworld_ml1_push import (
    ML1PushEpisodeRecord,
    ML1PushTask,
    ML1PushTaskCatalog,
    ML1PushTaskIndex,
)

from icil_jax_rlbench.configs.eval_metaworld_ml1_reach_gate1 import (
    get_config as get_reach_gate1_config,
)
from icil_jax_rlbench.configs.metaworld_ml1_push_kvb import (
    get_config as get_ttt_config,
)
from icil_jax_rlbench.configs.metaworld_ml1_push_query_only import (
    get_config as get_query_config,
)
from icil_jax_rlbench.data.metaworld_ml1_push import (
    ML1PushTaskDataset,
    ML1PushTaskSampler,
)
from icil_jax_rlbench.eval.metaworld_hidden_goal_gate1 import (
    evaluate_metaworld_gate1,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.metaworld_query_runner import (
    train_metaworld_query_only,
)
from icil_jax_rlbench.train.metaworld_ttt_runner import train_metaworld_ttt


def _tasks() -> tuple[ML1PushTask, ...]:
    tasks = []
    for index in range(100):
        if index < 40:
            split = 'train'
            task_id = f'train-{index:03d}'
            source_partition = 'train'
            source_index = index
        elif index < 50:
            split = 'validation'
            task_id = f'validation-{index - 40:03d}'
            source_partition = 'train'
            source_index = index
        else:
            split = 'test'
            task_id = f'test-{index - 50:03d}'
            source_partition = 'test'
            source_index = index - 50
        goal = (float(index), float(index + 1), 0.02)
        tasks.append(
            ML1PushTask(
                task_id=task_id,
                split=split,
                source_partition=source_partition,
                source_index=source_index,
                source_object_position=(0.0, 0.6, 0.02),
                source_goal_position=goal,
                goal=goal,
                native_task_sha256=f'{index + 1:064x}',
            )
        )
    return tuple(tasks)


def _state_sequence(episode_in_task: int, actions: np.ndarray) -> np.ndarray:
    state = np.zeros((3, 39), dtype=np.float32)
    state[0, :3] = (episode_in_task * 0.01, 0.5, 0.2)
    state[0, 4:7] = (-0.05 + episode_in_task * 0.01, 0.62, 0.02)
    state[1:] = state[0]
    state[1:, :3] += np.cumsum(actions[:, :3], axis=0) * np.float32(0.01)
    state[0, 18:36] = state[0, :18]
    state[1:, 18:36] = state[:-1, :18]
    return state


@pytest.fixture(scope='module')
def processed_push_cache(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp('ml1_push') / 'processed'
    catalog = ML1PushTaskCatalog(0, _tasks())
    records = []
    episodes = []
    episode_index = 0
    for task_position, task in enumerate(catalog.tasks):
        for episode_in_task in range(2):
            seed = episode_index + 100
            action_value = np.float32((task_position - 50) / 100.0)
            actions = np.asarray(
                [
                    [action_value, 0.1, -0.1, 1.0],
                    [action_value, 0.05, -0.05, 1.0],
                ],
                dtype=np.float32,
            )
            records.append(
                ML1PushEpisodeRecord(
                    episode_index=episode_index,
                    task_id=task.task_id,
                    task_episode_index=episode_in_task,
                    seed=seed,
                    hand_start=(0.0, 0.5, 0.2 + episode_in_task * 0.01),
                    object_start=(
                        -0.05 + episode_in_task * 0.01,
                        0.62,
                        0.02,
                    ),
                )
            )
            episodes.append(
                EpisodeData(
                    episode_index=episode_index,
                    seed=seed,
                    observations={
                        'state': _state_sequence(episode_in_task, actions)
                    },
                    actions=actions,
                    action_history_padding=np.zeros((4,), dtype=np.float32),
                    rewards=np.asarray([0.0, 1.0], dtype=np.float64),
                    terminated=np.asarray([False, True], dtype=np.bool_),
                    truncated=np.asarray([False, False], dtype=np.bool_),
                    success=np.asarray([False, True], dtype=np.bool_),
                    termination_reason='success',
                    source_episode_id=(
                        f'metaworld-ml1-push:{task.task_id}:'
                        f'{episode_in_task:06d}'
                    ),
                )
            )
            episode_index += 1
    task_index = ML1PushTaskIndex(catalog, tuple(records))
    write_processed_bundle(
        root,
        integration='metaworld_ml1_push',
        episodes=episodes,
        provenance={'metaworld_ml1_push': task_index.to_dict()},
    )
    return root


def test_push_loader_preserves_task_demo_and_time_axes(processed_push_cache):
    dataset = ML1PushTaskDataset(processed_push_cache)
    assert tuple(
        len(dataset.task_ids(split)) for split in ('train', 'validation', 'test')
    ) == (40, 10, 50)
    assert dataset.normalization.source_episode_indices == tuple(range(80))
    assert dataset.normalization_id.startswith('ml1_push_train_')

    batch = ML1PushTaskSampler(dataset, split='train', seed=3).build_batch(
        1,
        support_episodes=1,
        query_episodes=1,
        task_ids=['train-000'],
    )
    assert batch['support']['observation'].shape == (1, 1, 2, 39)
    assert batch['query']['action'].shape == (1, 1, 2, 4)
    assert not np.intersect1d(
        batch['support']['episode_id'], batch['query']['episode_id']
    ).size


def test_push_query_checkpoint_initializes_kvb_training(
    processed_push_cache,
    tmp_path,
):
    query_cfg = get_query_config()
    query_cfg.dataset.cache_root = str(processed_push_cache)
    query_cfg.model.hidden_dim = 16
    query_cfg.model.fast_dim = 8
    query_cfg.model.fast_hidden_dim = 8
    query_cfg.train.num_steps = 1
    query_cfg.train.batch_size = 1
    query_cfg.train.query_episodes_per_task = 1
    query_cfg.train.log_every = 1
    query_cfg.train.eval_every = 100
    query_cfg.train.ckpt_every = 1
    query_cfg.train.output_dir = str(tmp_path / 'query')
    query_checkpoint = train_metaworld_query_only(ConfigDict(query_cfg))
    query_payload = load_checkpoint(query_checkpoint)
    assert query_payload['extra']['checkpoint_type'] == (
        'metaworld_ml1_push_query_only'
    )
    assert query_payload['extra']['normalizer_id'].startswith('ml1_push_train_')
    wrong_eval_cfg = get_reach_gate1_config()
    wrong_eval_cfg.checkpoint_path = str(query_checkpoint)
    with pytest.raises(ValueError, match='checkpoint belongs to'):
        evaluate_metaworld_gate1(ConfigDict(wrong_eval_cfg))

    ttt_cfg = get_ttt_config()
    ttt_cfg.dataset.cache_root = str(processed_push_cache)
    ttt_cfg.model.hidden_dim = 16
    ttt_cfg.model.fast_dim = 8
    ttt_cfg.model.fast_hidden_dim = 8
    ttt_cfg.adaptation.write_segment_size = 2
    ttt_cfg.train.initial_checkpoint_path = str(query_checkpoint)
    ttt_cfg.train.num_steps = 1
    ttt_cfg.train.batch_size = 1
    ttt_cfg.train.support_episodes_per_task = 1
    ttt_cfg.train.query_episodes_per_task = 1
    ttt_cfg.train.log_every = 1
    ttt_cfg.train.eval_every = 100
    ttt_cfg.train.ckpt_every = 1
    ttt_cfg.train.output_dir = str(tmp_path / 'ttt')
    ttt_checkpoint = train_metaworld_ttt(ConfigDict(ttt_cfg))
    ttt_payload = load_checkpoint(ttt_checkpoint)
    assert ttt_payload['extra']['checkpoint_type'] == 'metaworld_ml1_push_ttt'
    assert ttt_payload['extra']['initial_query_checkpoint'] == str(
        query_checkpoint.resolve()
    )
    assert ttt_payload['extra']['transient_fast_state_saved'] is False
    assert ttt_payload['config']['dataset']['integration'] == (
        'metaworld_ml1_push'
    )
