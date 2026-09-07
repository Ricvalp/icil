from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from phi_mujoco.integrations import family_contract

from icil_jax_rlbench.configs.eval_metaworld_ml45_conditioned_query import (
    get_config as get_conditioned_eval_config,
)
from icil_jax_rlbench.configs.eval_metaworld_ml45_ttt import (
    get_config as get_eval_config,
)
from icil_jax_rlbench.configs.metaworld_ml45_action_bc import (
    get_config as get_action_bc_config,
)
from icil_jax_rlbench.configs.metaworld_ml45_fomaml import (
    get_config as get_fomaml_config,
)
from icil_jax_rlbench.configs.metaworld_ml45_kvb import (
    get_config as get_kvb_config,
)
from icil_jax_rlbench.configs.metaworld_ml45_family_conditioned_query_only import (
    get_config as get_family_conditioned_config,
)
from icil_jax_rlbench.configs.metaworld_ml45_oracle_conditioned_query_only import (
    get_config as get_oracle_conditioned_config,
)
from icil_jax_rlbench.configs.metaworld_ml45_query_only import (
    get_config as get_query_config,
)
from icil_jax_rlbench.data.metaworld_conditioning import MetaWorldConditioning
from icil_jax_rlbench.data.metaworld_hidden_goal import (
    benchmark_for_integration,
    benchmark_from_config,
)
from icil_jax_rlbench.eval.metaworld_hidden_goal_ttt import _grouped_aggregate
from icil_jax_rlbench.train.metaworld_query_runner import query_checkpoint_type


def _reset_vector(
    family: str,
    task_latent: tuple[float, ...],
    *,
    nuisance: float,
) -> tuple[float, ...]:
    contract = family_contract(family)
    assert contract.reset_vector_dim is not None
    values = np.full((contract.reset_vector_dim,), nuisance, dtype=np.float32)
    cursor = 0
    for field in contract.reset_fields:
        if field.role != 'task_latent':
            continue
        width = field.stop - field.start
        values[field.start : field.stop] = task_latent[cursor : cursor + width]
        cursor += width
    assert cursor == len(task_latent)
    return tuple(float(value) for value in values)


class _ConditioningDataset:
    integration_name = 'metaworld_ml45'
    observation_dim = 39

    def __init__(self) -> None:
        self.descriptors = {
            'reach-a': {
                'family': 'reach-v3',
                'native_rand_vec': _reset_vector(
                    'reach-v3', (0.1, 0.7, 0.2), nuisance=-0.4
                ),
            },
            'reach-b': {
                'family': 'reach-v3',
                'native_rand_vec': _reset_vector(
                    'reach-v3', (0.1, 0.7, 0.2), nuisance=0.4
                ),
            },
            'reach-c': {
                'family': 'reach-v3',
                'native_rand_vec': _reset_vector(
                    'reach-v3', (0.2, 0.8, 0.3), nuisance=0.0
                ),
            },
            'push-back-a': {
                'family': 'push-back-v3',
                'native_rand_vec': _reset_vector(
                    'push-back-v3', (0.3, 0.6), nuisance=0.1
                ),
            },
            'hammer-a': {
                'family': 'hammer-v3',
                'native_rand_vec': _reset_vector(
                    'hammer-v3', (), nuisance=0.2
                ),
            },
        }
        tasks = tuple(
            SimpleNamespace(family=value['family'])
            for value in self.descriptors.values()
        )
        self.task_index = SimpleNamespace(
            catalog=SimpleNamespace(tasks=tasks)
        )

    def task_ids(self, split: str) -> tuple[str, ...]:
        assert split == 'train'
        return tuple(self.descriptors)

    def task_family(self, task_id: str) -> str:
        return str(self.descriptors[task_id]['family'])

    def task_descriptor(self, task_id: str) -> dict[str, object]:
        return dict(self.descriptors[task_id])


def test_ml45_benchmark_uses_the_shared_hidden_goal_policy_contract() -> None:
    benchmark = benchmark_for_integration('metaworld_ml45')
    assert benchmark.label == 'ML45'
    assert benchmark.query_mode == 'metaworld_ml45_query_only'
    assert benchmark.ttt_mode == 'metaworld_ml45_ttt'
    assert benchmark.protocols == ('development', 'final')
    assert benchmark.family_aware
    assert benchmark.split_names == (
        'train',
        'latent_validation',
        'family_validation',
        'validation',
        'test',
    )


def test_ml45_query_and_main_kvb_configs_select_delta_read() -> None:
    query = get_query_config()
    kvb = get_kvb_config()
    assert benchmark_from_config(query).integration_name == 'metaworld_ml45'
    assert query.mode == 'metaworld_ml45_query_only'
    assert query.dataset.protocol == 'development'
    assert query.train.num_steps == 100_000
    assert kvb.mode == 'metaworld_ml45_ttt'
    assert kvb.dataset.integration == 'metaworld_ml45'
    assert kvb.adaptation.write_objective == 'kvb'
    assert kvb.adaptation.read_mode == 'delta'
    assert not kvb.adaptation.first_order
    assert kvb.train.num_steps == 100_000


def test_ml45_conditioned_baseline_configs_are_separate_and_explicit() -> None:
    family = get_family_conditioned_config()
    oracle = get_oracle_conditioned_config()
    evaluation = get_conditioned_eval_config()
    assert family.conditioning.mode == 'family'
    assert oracle.conditioning.mode == 'family_task_latent'
    assert query_checkpoint_type(family) == family.mode
    assert query_checkpoint_type(oracle) == oracle.mode
    assert evaluation.split == 'latent_validation'
    assert not evaluation.allow_unseen_families


def test_ml45_oracle_conditioning_uses_only_declared_task_latents() -> None:
    dataset = _ConditioningDataset()
    family = MetaWorldConditioning.fit(dataset, mode='family')
    oracle = MetaWorldConditioning.fit(dataset, mode='family_task_latent')

    assert family.family_names == ('reach-v3', 'push-back-v3', 'hammer-v3')
    assert family.context_dim == 3
    assert oracle.latent_dim == 3
    assert oracle.context_dim == 9
    np.testing.assert_array_equal(
        oracle.context(dataset, 'reach-a'),
        oracle.context(dataset, 'reach-b'),
    )
    np.testing.assert_array_equal(
        oracle.context(dataset, 'push-back-a')[-3:],
        np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        oracle.context(dataset, 'hammer-a')[-3:],
        np.zeros((3,), dtype=np.float32),
    )
    restored = MetaWorldConditioning.from_dict(oracle.to_dict())
    restored.validate_dataset(dataset)
    assert not oracle.to_dict()['contains_task_id']
    assert not oracle.to_dict()['contains_episode_nuisance']


def test_ml45_conditioning_broadcasts_over_demo_and_time_axes() -> None:
    dataset = _ConditioningDataset()
    conditioning = MetaWorldConditioning.fit(
        dataset, mode='family_task_latent'
    )
    observations = np.zeros((2, 2, 4, 39), dtype=np.float32)
    augmented = conditioning.augment_observations(
        dataset, observations, ('reach-a', 'push-back-a')
    )
    assert augmented.shape == (2, 2, 4, 48)
    np.testing.assert_array_equal(
        augmented[0, 0, 0, 39:], augmented[0, 1, 3, 39:]
    )
    assert not np.array_equal(augmented[0, 0, 0, 39:], augmented[1, 0, 0, 39:])


def test_ml45_ablation_and_evaluation_configs_remain_explicit() -> None:
    fomaml = get_fomaml_config()
    action_bc = get_action_bc_config()
    evaluation = get_eval_config()
    assert fomaml.adaptation.first_order
    assert fomaml.adaptation.write_objective == 'kvb'
    assert action_bc.adaptation.write_objective == 'action_bc'
    assert not action_bc.adaptation.first_order
    assert action_bc.adaptation.read_mode == 'delta'
    assert evaluation.integration == 'metaworld_ml45'
    assert evaluation.split == 'family_validation'
    assert 'same_family_wrong_instance' in evaluation.conditions
    assert 'different_family_support' in evaluation.conditions


def test_ml45_evaluation_can_aggregate_families_and_shared_motion_phases() -> None:
    def record(task_id: str, family: str, phases: list[str], loss: float, success: int):
        return {
            'task_id': task_id,
            'task_family': family,
            'task_motion_phases': phases,
            'offline_query_loss': loss,
            'fast_delta_norm': 0.2,
            'closed_loop': {
                'success_rate': float(success),
                'successful_episodes': success,
                'attempted_episodes': 1,
            },
        }

    records = {
        '2': {
            'no_update': [
                record('a', 'pick-place-wall-v3', ['reach', 'grasp'], 2.0, 0),
                record('b', 'coffee-pull-v3', ['reach', 'grasp'], 1.0, 0),
            ],
            'correct_support': [
                record('a', 'pick-place-wall-v3', ['reach', 'grasp'], 1.0, 1),
                record('b', 'coffee-pull-v3', ['reach', 'grasp'], 0.5, 1),
            ],
        }
    }
    by_family = _grouped_aggregate(
        records, metadata_key='task_family', seed=0
    )
    by_phase = _grouped_aggregate(
        records, metadata_key='task_motion_phases', seed=0
    )
    assert set(by_family) == {'coffee-pull-v3', 'pick-place-wall-v3'}
    assert set(by_phase) == {'grasp', 'reach'}
    assert by_phase['reach']['2']['correct_support_gain']['success_rate'] == 1.0
