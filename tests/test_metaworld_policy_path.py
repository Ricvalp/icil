from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict
from phi_mujoco.dataset import EpisodeData, write_processed_bundle
from phi_mujoco.evaluation import PolicyInput
from phi_mujoco.integrations.metaworld_ml1_reach import (
    SPEC,
    ML1ReachEpisodeRecord,
    ML1ReachTask,
    ML1ReachTaskCatalog,
    ML1ReachTaskIndex,
)

from icil_jax_rlbench.configs.metaworld_ml1_reach_kvb import (
    get_config as get_ttt_config,
)
from icil_jax_rlbench.configs.metaworld_ml1_reach_query_only import get_config
from icil_jax_rlbench.data.metaworld_ml1_reach import (
    ML1ReachTaskDataset,
    ML1ReachTaskSampler,
)
from icil_jax_rlbench.eval.metaworld_ml1_reach_gate1 import (
    _ordinary_adapt,
    _parameter_mask,
)
from icil_jax_rlbench.eval.metaworld_policy import (
    ML1ReachJaxPolicy,
    MetaWorldConditionedJaxPolicy,
)
from icil_jax_rlbench.eval.support_controls import condition_support
from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    TTTAdaptConfig,
    adapt_fast_state,
    init_fast_weight_ttt_params,
    initial_fast_state,
    predict_action,
    robotics_action_loss,
    tree_difference_norm,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.metaworld_query_runner import (
    CHECKPOINT_TYPE as QUERY_CHECKPOINT_TYPE,
    metaworld_model_config_from,
    train_metaworld_query_only,
)
from icil_jax_rlbench.train.metaworld_ttt_runner import (
    CHECKPOINT_TYPE as TTT_CHECKPOINT_TYPE,
    train_metaworld_ttt,
)


def _tasks() -> tuple[ML1ReachTask, ...]:
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
        tasks.append(
            ML1ReachTask(
                task_id=task_id,
                split=split,
                source_partition=source_partition,
                source_index=source_index,
                source_object_position=(0.0, 0.6, 0.02),
                goal=(float(index), float(index + 1), float(index + 2)),
                native_task_sha256=f'{index + 1:064x}',
            )
        )
    return tuple(tasks)


def _state_sequence(episode_in_task: int, action: np.ndarray) -> np.ndarray:
    current = np.zeros((3, 18), dtype=np.float32)
    current[0, 0] = np.float32(episode_in_task * 0.1)
    current[0, 1] = np.float32(-0.2)
    current[1] = current[0]
    current[2] = current[1]
    current[1:, :3] += np.cumsum(action[:, :3], axis=0) * np.float32(0.01)
    state = np.zeros((3, 39), dtype=np.float32)
    state[:, :18] = current
    state[0, 18:36] = current[0]
    state[1:, 18:36] = current[:-1]
    return state


@pytest.fixture(scope='module')
def processed_cache(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp('ml1_reach') / 'processed'
    catalog = ML1ReachTaskCatalog(0, _tasks())
    records = []
    episodes = []
    episode_index = 0
    for task_index, task in enumerate(catalog.tasks):
        for episode_in_task in range(2):
            seed = episode_index + 10
            action_value = np.float32((task_index - 50) / 100.0)
            actions = np.asarray(
                [
                    [action_value, 0.1, -0.1, -0.2],
                    [action_value, 0.05, -0.05, -0.2],
                ],
                dtype=np.float32,
            )
            state = _state_sequence(episode_in_task, actions)
            records.append(
                ML1ReachEpisodeRecord(
                    episode_index=episode_index,
                    task_id=task.task_id,
                    task_episode_index=episode_in_task,
                    seed=seed,
                    hand_start=(0.0, 0.5, 0.1 + episode_in_task * 0.01),
                )
            )
            episodes.append(
                EpisodeData(
                    episode_index=episode_index,
                    seed=seed,
                    observations={'state': state},
                    actions=actions,
                    action_history_padding=np.zeros((4,), dtype=np.float32),
                    rewards=np.asarray([0.0, 1.0], dtype=np.float64),
                    terminated=np.asarray([False, True], dtype=np.bool_),
                    truncated=np.asarray([False, False], dtype=np.bool_),
                    success=np.asarray([False, True], dtype=np.bool_),
                    termination_reason='success',
                    source_episode_id=(
                        f'metaworld-ml1-reach:{task.task_id}:{episode_in_task:06d}'
                    ),
                )
            )
            episode_index += 1
    task_index = ML1ReachTaskIndex(catalog, tuple(records))
    write_processed_bundle(
        root,
        integration='metaworld_ml1_reach',
        episodes=episodes,
        provenance={'metaworld_ml1_reach': task_index.to_dict()},
    )
    return root


def test_task_loader_uses_declared_splits_and_separate_episode_axes(processed_cache):
    dataset = ML1ReachTaskDataset(processed_cache)
    assert tuple(len(dataset.task_ids(split)) for split in ('train', 'validation', 'test')) == (
        40,
        10,
        50,
    )
    assert dataset.normalization.source_episode_indices == tuple(range(80))
    assert dataset.integrity_report()['normalizer_uses_exact_training_task_episodes']

    batch = ML1ReachTaskSampler(dataset, split='train', seed=3).build_batch(
        1,
        support_episodes=1,
        query_episodes=1,
        task_ids=['train-000'],
    )
    assert batch['support']['observation'].shape == (1, 1, 2, 39)
    assert batch['query']['action'].shape == (1, 1, 2, 4)
    assert batch['support']['write_mask'].shape == (1, 1, 2)
    assert not np.any(batch['support']['outer_loss_mask'])
    assert np.all(batch['query']['outer_loss_mask'])
    assert not np.intersect1d(
        batch['support']['episode_id'], batch['query']['episode_id']
    ).size
    assert set(batch) == {'support', 'query'}


def test_continuous_action_objective_and_ordinary_adaptation(processed_cache):
    dataset = ML1ReachTaskDataset(processed_cache)
    cfg = get_config()
    model_cfg = replace(
        metaworld_model_config_from(cfg, dataset),
        hidden_dim=16,
        fast_dim=8,
        fast_hidden_dim=8,
    )
    params = init_fast_weight_ttt_params(jax.random.key(4), model_cfg)
    batch = ML1ReachTaskSampler(dataset, split='train', seed=7).build_batch(
        1,
        support_episodes=1,
        query_episodes=1,
        task_ids=['train-003'],
    )
    support = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value[0]), batch['support']
    )
    initial_prediction = predict_action(
        params,
        initial_fast_state(params),
        support['observation'],
        model_cfg,
        read_enabled=False,
    )
    initial_loss, metrics = robotics_action_loss(
        initial_prediction, support['action'], support['write_mask'], model_cfg
    )
    assert np.isfinite(float(initial_loss))
    assert np.isfinite(float(metrics['gripper_l1']))

    adapted, trace = _ordinary_adapt(
        params,
        support,
        model_cfg=model_cfg,
        parameter_mask=_parameter_mask(params, 'action_heads'),
        steps=20,
        learning_rate=1e-2,
        clip_norm=1.0,
    )
    final_prediction = predict_action(
        adapted,
        initial_fast_state(adapted),
        support['observation'],
        model_cfg,
        read_enabled=False,
    )
    final_loss, _ = robotics_action_loss(
        final_prediction, support['action'], support['write_mask'], model_cfg
    )
    assert float(final_loss) < float(initial_loss)
    assert float(trace['support_loss'][-1]) < float(trace['support_loss'][0])


class _ProjectingIntegration:
    spec = SPEC

    @staticmethod
    def project_action(action, observations=None):
        del observations
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def test_policy_adapter_denormalizes_and_projects_actions(processed_cache):
    dataset = ML1ReachTaskDataset(processed_cache)
    model_cfg = FastWeightTTTConfig(
        observation_dim=39,
        action_dim=4,
        translation_dim=3,
        hidden_dim=8,
        fast_dim=4,
        fast_hidden_dim=4,
        translation_output='linear',
        gripper_loss='huber',
    )
    params = init_fast_weight_ttt_params(jax.random.key(9), model_cfg)
    params = jax.tree_util.tree_map(jnp.zeros_like, params)
    integration = _ProjectingIntegration()
    policy = ML1ReachJaxPolicy(
        integration=integration,
        params=params,
        model_cfg=model_cfg,
        normalization=dataset.normalization,
    )
    action = policy.predict(
        PolicyInput(
            observations={'state': np.zeros((39,), dtype=np.float32)},
            episode_index=0,
            step_index=0,
            seed=123,
            integration=SPEC,
        )
    )
    np.testing.assert_allclose(
        action,
        np.asarray(dataset.normalization.action_mean, dtype=np.float32),
        atol=1e-6,
    )
    assert action.dtype == np.float32


def test_conditioned_policy_appends_context_to_raw_simulator_state(processed_cache):
    dataset = ML1ReachTaskDataset(processed_cache)
    context = np.asarray([1.0, -0.5], dtype=np.float32)
    model_cfg = FastWeightTTTConfig(
        observation_dim=39 + context.size,
        action_dim=4,
        translation_dim=3,
        hidden_dim=8,
        fast_dim=4,
        fast_hidden_dim=4,
        translation_output='linear',
        gripper_loss='huber',
    )
    params = jax.tree_util.tree_map(
        jnp.zeros_like,
        init_fast_weight_ttt_params(jax.random.key(10), model_cfg),
    )
    policy = MetaWorldConditionedJaxPolicy(
        integration=_ProjectingIntegration(),
        params=params,
        model_cfg=model_cfg,
        normalization=dataset.normalization,
        context=context,
    )
    action = policy.predict(
        PolicyInput(
            observations={'state': np.zeros((39,), dtype=np.float32)},
            episode_index=0,
            step_index=0,
            seed=123,
            integration=SPEC,
        )
    )
    np.testing.assert_allclose(
        action,
        np.asarray(dataset.normalization.action_mean, dtype=np.float32),
        atol=1e-6,
    )


def test_policy_adapter_uses_one_frozen_adapted_fast_state(processed_cache):
    dataset = ML1ReachTaskDataset(processed_cache)
    model_cfg = FastWeightTTTConfig(
        observation_dim=39,
        action_dim=4,
        translation_dim=3,
        hidden_dim=12,
        fast_dim=6,
        fast_hidden_dim=8,
        gate_init=0.3,
        inner_lr_init=0.1,
        translation_output='linear',
        gripper_loss='huber',
    )
    params = init_fast_weight_ttt_params(jax.random.key(13), model_cfg)
    batch = ML1ReachTaskSampler(dataset, split='train', seed=17).build_batch(
        1,
        support_episodes=1,
        query_episodes=1,
        task_ids=['train-004'],
    )
    support = {
        name: jnp.asarray(batch['support'][name][0])
        for name in ('observation', 'action', 'next_observation', 'write_mask')
    }
    adapted, _ = adapt_fast_state(
        params,
        support,
        model_cfg,
        TTTAdaptConfig(write_segment_size=2),
    )
    assert float(tree_difference_norm(adapted, initial_fast_state(params))) > 0.0

    integration = _ProjectingIntegration()
    base = ML1ReachJaxPolicy(
        integration=integration,
        params=params,
        model_cfg=model_cfg,
        normalization=dataset.normalization,
        read_enabled=True,
    )
    adapted_policy = ML1ReachJaxPolicy(
        integration=integration,
        params=params,
        fast_state=adapted,
        model_cfg=model_cfg,
        normalization=dataset.normalization,
        read_enabled=True,
    )
    inputs = PolicyInput(
        observations={'state': np.zeros((39,), dtype=np.float32)},
        episode_index=0,
        step_index=0,
        seed=123,
        integration=SPEC,
    )
    assert not np.allclose(base.predict(inputs), adapted_policy.predict(inputs))
    before_reset = adapted_policy.predict(inputs)
    adapted_policy.reset(integration=SPEC, seed=999)
    np.testing.assert_array_equal(before_reset, adapted_policy.predict(inputs))


def test_support_shuffles_leave_padding_fixed():
    support = {
        'observation': np.arange(8, dtype=np.float32).reshape(1, 4, 2),
        'action': np.arange(12, dtype=np.float32).reshape(1, 4, 3),
        'next_observation': np.arange(8, dtype=np.float32).reshape(1, 4, 2) + 1,
        'write_mask': np.asarray([[True, True, False, False]]),
    }
    shuffled = condition_support(
        'shuffled_actions', support, support, np.random.default_rng(5)
    )
    np.testing.assert_array_equal(shuffled['action'][0, 2:], support['action'][0, 2:])
    np.testing.assert_array_equal(
        np.sort(shuffled['action'][0, :2], axis=0),
        np.sort(support['action'][0, :2], axis=0),
    )


def test_one_step_query_only_training_writes_self_describing_checkpoint(
    processed_cache,
    tmp_path,
):
    cfg = get_config()
    cfg.dataset.cache_root = str(processed_cache)
    cfg.model.hidden_dim = 16
    cfg.model.fast_dim = 8
    cfg.model.fast_hidden_dim = 8
    cfg.train.num_steps = 1
    cfg.train.batch_size = 2
    cfg.train.query_episodes_per_task = 1
    cfg.train.log_every = 1
    cfg.train.eval_every = 100
    cfg.train.ckpt_every = 1
    cfg.train.output_dir = str(tmp_path / 'runs')
    checkpoint_path = train_metaworld_query_only(ConfigDict(cfg))
    payload = load_checkpoint(checkpoint_path)
    assert payload['step'] == 1
    assert payload['extra']['checkpoint_type'] == QUERY_CHECKPOINT_TYPE
    assert payload['extra']['normalizer_id'].startswith('ml1_reach_train_')
    assert payload['extra']['transient_fast_state_saved'] is False


def test_metaworld_kvb_training_and_resume_restore_scientific_config(
    processed_cache,
    tmp_path,
):
    query_cfg = get_config()
    query_cfg.dataset.cache_root = str(processed_cache)
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

    cfg = get_ttt_config()
    cfg.dataset.cache_root = str(processed_cache)
    cfg.model.hidden_dim = 16
    cfg.model.fast_dim = 8
    cfg.model.fast_hidden_dim = 8
    cfg.adaptation.write_segment_size = 2
    cfg.train.initial_checkpoint_path = str(query_checkpoint)
    cfg.train.num_steps = 1
    cfg.train.batch_size = 1
    cfg.train.support_episodes_per_task = 1
    cfg.train.query_episodes_per_task = 1
    cfg.train.log_every = 1
    cfg.train.eval_every = 100
    cfg.train.ckpt_every = 1
    cfg.train.output_dir = str(tmp_path / 'ttt')
    checkpoint = train_metaworld_ttt(ConfigDict(cfg))
    payload = load_checkpoint(checkpoint)
    assert payload['step'] == 1
    assert payload['extra']['checkpoint_type'] == TTT_CHECKPOINT_TYPE
    assert payload['extra']['initial_query_checkpoint'] == str(query_checkpoint.resolve())
    assert payload['extra']['adaptation_config']['first_order'] is False
    assert payload['extra']['training_contract']['meta_batch'] == {
        'task_batch_size': 1,
        'support_episodes_per_task': 1,
        'query_episodes_per_task': 1,
    }
    assert 'train_sampler_rng_state' in payload['extra']
    assert 'fast_state' not in payload
    assert payload['extra']['transient_fast_state_saved'] is False

    resume_cfg = get_ttt_config()
    resume_cfg.dataset.cache_root = str(processed_cache)
    resume_cfg.train.resume_path = str(checkpoint)
    resume_cfg.train.num_steps = 2
    resume_cfg.train.output_dir = str(tmp_path / 'resumed')
    resume_cfg.train.log_every = 1
    resume_cfg.train.eval_every = 100
    resume_cfg.train.ckpt_every = 1
    resume_cfg.model.hidden_dim = 999
    resume_cfg.adaptation.first_order = True
    resume_cfg.train.batch_size = 2
    resume_cfg.train.lr = 9e-2
    resumed = train_metaworld_ttt(ConfigDict(resume_cfg))
    resumed_payload = load_checkpoint(resumed)
    assert resumed_payload['step'] == 2
    assert resumed_payload['config']['model']['hidden_dim'] == 16
    assert resumed_payload['config']['adaptation']['first_order'] is False
    assert resumed_payload['config']['train']['batch_size'] == 1
    assert resumed_payload['config']['train']['lr'] == pytest.approx(3e-4)
