from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from phi_mujoco.dataset import EpisodeData, write_processed_bundle
from phi_mujoco.integrations import (
    ML10_TEST_FAMILIES,
    ML10_TRAIN_FAMILIES,
    family_contract,
)
from phi_mujoco.integrations.metaworld_ml10 import (
    ML10EpisodeRecord,
    ML10Task,
    ML10TaskCatalog,
    ML10TaskIndex,
)

from icil_jax_rlbench.configs.metaworld_ml10_action_bc_debug import (
    get_config as get_action_bc_debug_config,
)
from icil_jax_rlbench.configs.metaworld_ml10_kvb import get_config
from icil_jax_rlbench.configs.metaworld_ml10_kvb_debug import (
    get_config as get_kvb_debug_config,
)
from icil_jax_rlbench.data.metaworld_hidden_goal import (
    MetaWorldTaskDataset,
    MetaWorldTaskSampler,
)
from icil_jax_rlbench.models.fast_weight_ttt import init_fast_weight_ttt_params
from icil_jax_rlbench.train.metaworld_query_runner import (
    metaworld_model_config_from,
)
from icil_jax_rlbench.train.ttt_config import adaptation_config_from, step_config_from
from icil_jax_rlbench.train.ttt_step import ttt_meta_objective


def _catalog() -> ML10TaskCatalog:
    tasks = []
    for partition, families in (
        ('train', ML10_TRAIN_FAMILIES),
        ('test', ML10_TEST_FAMILIES),
    ):
        for family in families:
            contract = family_contract(family)
            assert contract.reset_vector_dim is not None
            for instance in range(50):
                tasks.append(
                    ML10Task(
                        task_id=f'{partition}-{family}-{instance:03d}',
                        benchmark_partition=partition,  # type: ignore[arg-type]
                        family=family,
                        instance_index=instance,
                        native_rand_vec=tuple(
                            float(instance + axis / 100.0)
                            for axis in range(contract.reset_vector_dim)
                        ),
                        native_task_sha256=hashlib.sha256(
                            f'{partition}:{family}:{instance}'.encode()
                        ).hexdigest(),
                    )
                )
    return ML10TaskCatalog(0, tuple(tasks))


def _states(task_position: int, episode_in_task: int, actions: np.ndarray) -> np.ndarray:
    current = np.zeros((3, 18), dtype=np.float32)
    current[0, :3] = (
        np.float32(episode_in_task * 0.01),
        np.float32(0.4 + (task_position % 10) * 0.01),
        np.float32(0.2),
    )
    current[1:] = current[0]
    current[1:, :3] += np.cumsum(actions[:, :3], axis=0) * np.float32(0.01)
    state = np.zeros((3, 39), dtype=np.float32)
    state[:, :18] = current
    state[0, 18:36] = current[0]
    state[1:, 18:36] = current[:-1]
    return state


@pytest.fixture(scope='module')
def processed_ml10_cache(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp('ml10') / 'processed'
    catalog = _catalog()
    records = []
    episodes = []
    episode_index = 0
    for task_position, task in enumerate(catalog.tasks):
        for episode_in_task in range(2):
            seed = episode_index + 10
            action_value = np.float32(((task_position % 50) - 25) / 50.0)
            actions = np.asarray(
                [
                    [action_value, 0.1, -0.1, 1.0],
                    [action_value, 0.05, -0.05, 1.0],
                ],
                dtype=np.float32,
            )
            records.append(
                ML10EpisodeRecord(
                    episode_index=episode_index,
                    task_id=task.task_id,
                    task_episode_index=episode_in_task,
                    seed=seed,
                    hand_start=(0.0, 0.6, 0.2 + episode_in_task * 0.01),
                    episode_rand_vec=task.native_rand_vec,
                )
            )
            episodes.append(
                EpisodeData(
                    episode_index=episode_index,
                    seed=seed,
                    observations={
                        'state': _states(task_position, episode_in_task, actions)
                    },
                    actions=actions,
                    action_history_padding=np.zeros((4,), dtype=np.float32),
                    rewards=np.asarray([0.0, 1.0], dtype=np.float64),
                    terminated=np.asarray([False, True], dtype=np.bool_),
                    truncated=np.asarray([False, False], dtype=np.bool_),
                    success=np.asarray([False, True], dtype=np.bool_),
                    termination_reason='success',
                    source_episode_id=(
                        f'metaworld-ml10:{task.task_id}:{episode_in_task:06d}'
                    ),
                )
            )
            episode_index += 1
    task_index = ML10TaskIndex(catalog, tuple(records))
    write_processed_bundle(
        root,
        integration='metaworld_ml10',
        episodes=episodes,
        provenance={'metaworld_ml10': task_index.to_dict()},
    )
    return root


@pytest.fixture(scope='module')
def ml10_dataset(processed_ml10_cache) -> MetaWorldTaskDataset:
    return MetaWorldTaskDataset(
        processed_ml10_cache,
        integration_name='metaworld_ml10',
        protocol='development',
        horizon_buckets=(4, 8),
    )


def test_ml10_loader_preserves_hierarchy_and_uses_train_only_stats(
    ml10_dataset,
) -> None:
    dataset = ml10_dataset
    assert len(dataset.task_ids('train')) == 320
    assert len(dataset.task_ids('latent_validation')) == 80
    assert len(dataset.task_ids('family_validation')) == 100
    assert len(dataset.task_ids('test')) == 250
    assert len(dataset.normalization.source_episode_indices) == 640
    assert dataset.integrity_report()['normalizer_uses_exact_training_task_episodes']

    sampler = MetaWorldTaskSampler(dataset, split='train', seed=3)
    batch = sampler.build_batch(
        2,
        support_episodes=1,
        query_episodes=1,
        task_ids=['train-reach-v3-000', 'train-push-v3-001'],
    )
    assert batch['support']['observation'].shape == (2, 1, 4, 39)
    assert batch['query']['action'].shape == (2, 1, 4, 4)
    assert not np.intersect1d(
        batch['support']['episode_id'], batch['query']['episode_id']
    ).size
    assert dataset.same_family_wrong_task(
        'train-reach-v3-040', 'latent_validation'
    ).startswith('train-reach-v3-')
    assert dataset.task_family(
        dataset.different_family_task(
            'train-reach-v3-040', 'latent_validation'
        )
    ) != 'reach-v3'
    family_screen = dataset.balanced_task_ids('family_validation', 10)
    assert {
        family: sum(dataset.task_family(task_id) == family for task_id in family_screen)
        for family in ('pick-place-v3', 'door-open-v3')
    } == {'pick-place-v3': 5, 'door-open-v3': 5}
    test_screen = dataset.balanced_task_ids('test', 10)
    assert {
        family: sum(dataset.task_family(task_id) == family for task_id in test_screen)
        for family in ML10_TEST_FAMILIES
    } == {family: 2 for family in ML10_TEST_FAMILIES}


def test_ml10_one_step_second_order_objective_is_finite(ml10_dataset) -> None:
    cfg = get_config()
    cfg.model.hidden_dim = 16
    cfg.model.fast_dim = 8
    cfg.model.fast_hidden_dim = 8
    cfg.adaptation.write_segment_size = 4
    dataset = ml10_dataset
    batch = MetaWorldTaskSampler(dataset, split='train', seed=9).build_batch(
        1,
        support_episodes=1,
        query_episodes=1,
        task_ids=['train-reach-v3-000'],
    )
    meta_batch = {
        'support': {
            name: jnp.asarray(batch['support'][name])
            for name in ('observation', 'action', 'next_observation', 'write_mask')
        },
        'query': {
            name: jnp.asarray(batch['query'][name])
            for name in ('observation', 'action', 'outer_loss_mask')
        },
    }
    model_cfg = metaworld_model_config_from(cfg, dataset)
    params = init_fast_weight_ttt_params(jax.random.key(0), model_cfg)
    loss, metrics = ttt_meta_objective(
        params,
        meta_batch,
        model_cfg,
        adaptation_config_from(cfg),
        step_config_from(cfg),
    )
    assert np.isfinite(float(loss))
    assert np.isfinite(float(metrics['fast_delta_norm']))


@pytest.mark.parametrize(
    ('factory', 'write_objective'),
    (
        (get_kvb_debug_config, 'kvb'),
        (get_action_bc_debug_config, 'action_bc'),
    ),
)
def test_ml10_debug_configs_are_small_eager_full_second_order_runs(
    factory, write_objective
) -> None:
    cfg = factory()
    assert cfg.adaptation.write_objective == write_objective
    assert cfg.adaptation.read_mode == 'delta'
    assert not cfg.adaptation.first_order
    assert cfg.train.eager_debug
    assert cfg.train.debug_max_time_steps == 32
    assert cfg.train.batch_size == 1
    assert cfg.train.support_episodes_per_task == 1
    assert cfg.train.query_episodes_per_task == 1
    assert cfg.train.num_steps == 2
    assert not cfg.wandb.enable
