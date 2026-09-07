from __future__ import annotations

import numpy as np

from icil_jax_rlbench.analysis.metaworld_update_information import (
    raw_support_statistics,
    ridge_predict,
)
from icil_jax_rlbench.configs.analyze_metaworld_ml10_information import get_config
from icil_jax_rlbench.configs.analyze_metaworld_ml45_information import (
    get_config as get_ml45_config,
)
from icil_jax_rlbench.visualization.metaworld_update_information import (
    update_clustering_summary,
    write_update_tsne_artifacts,
)


def test_raw_support_statistics_preserve_demo_time_contract_and_ignore_padding() -> None:
    observation = np.zeros((2, 4, 3), dtype=np.float32)
    action = np.zeros((2, 4, 2), dtype=np.float32)
    next_observation = np.zeros_like(observation)
    mask = np.asarray(
        [[True, True, False, False], [True, True, True, False]], dtype=np.bool_
    )
    observation[mask] = np.arange(15, dtype=np.float32).reshape(5, 3)
    next_observation[mask] = observation[mask] + 1.0
    action[mask] = 0.5
    support = {
        'observation': observation,
        'action': action,
        'next_observation': next_observation,
        'write_mask': mask,
    }
    first = raw_support_statistics(support)
    support['observation'][~mask] = 1e6
    support['action'][~mask] = -1e6
    second = raw_support_statistics(support)

    evidence_width = observation.shape[-1] * 2 + action.shape[-1]
    assert first.shape == ((2 + 4) * evidence_width + 2,)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


def test_ridge_probe_recovers_linear_targets_in_high_dimensional_dual_case() -> None:
    rng = np.random.default_rng(4)
    train_latent = rng.normal(size=(20, 4))
    test_latent = rng.normal(size=(5, 4))
    feature_map = rng.normal(size=(4, 40))
    target_map = rng.normal(size=(4, 3))
    train = train_latent @ feature_map
    test = test_latent @ feature_map
    target = train_latent @ target_map
    expected = test_latent @ target_map

    prediction = ridge_predict(train, target, test, ridge=1e-8)

    np.testing.assert_allclose(prediction, expected, atol=2e-4, rtol=2e-4)


def test_ml10_information_config_requests_matched_controls_and_full_vectors() -> None:
    cfg = get_config()
    assert cfg.support_counts == (1, 2)
    assert cfg.samples_per_task == 2
    assert 'correct_support' in cfg.conditions
    assert 'same_family_wrong_instance' in cfg.conditions
    assert 'different_family_support' in cfg.conditions
    assert 'shuffled_actions' in cfg.conditions
    assert 'shuffled_time' in cfg.conditions
    assert not cfg.visualization.enabled


def _synthetic_update_rows_and_features():
    rows = []
    vectors = []
    split_families = {
        'train': ('reach', 'push'),
        'latent_validation': ('reach', 'push'),
        'family_validation': ('pick', 'door'),
    }
    family_basis = {
        family: index for index, family in enumerate(('reach', 'push', 'pick', 'door'))
    }
    for split, families in split_families.items():
        for family in families:
            for task_index in range(2):
                task_id = f'{split}-{family}-{task_index:03d}'
                for sample_index in range(2):
                    vector = np.zeros((13,), dtype=np.float32)
                    vector[family_basis[family]] = 4.0
                    vector[4 + 2 * family_basis[family] + task_index] = 1.0
                    vector[-1] = sample_index * 0.01
                    vectors.append(vector)
                    rows.append(
                        {
                            'condition': 'correct_support',
                            'support_count': 2,
                            'target_split': split,
                            'target_task_id': task_id,
                            'target_family': family,
                            'target_instance_index': task_index,
                            'target_motion_phases': ['reach', family],
                            'target_motion_signature': f'reach > {family}',
                            'target_terminal_phase': family,
                            'sample_index': sample_index,
                            'query_gain': 0.1,
                            'final_fast_delta_norm': 1.0,
                        }
                    )
    return rows, {'final_fast_delta': np.asarray(vectors)}


def test_ml45_information_config_balances_all_development_families() -> None:
    cfg = get_ml45_config()
    assert cfg.integration == 'metaworld_ml45'
    assert cfg.support_counts == (2,)
    assert cfg.max_train_tasks == 400
    assert cfg.max_latent_tasks == 400
    assert cfg.max_family_tasks == 50
    assert cfg.samples_per_task == 2
    assert cfg.visualization.enabled
    assert 'final_fast_delta' in cfg.visualization.representations


def test_update_clustering_summary_measures_heldout_instances() -> None:
    rows, features = _synthetic_update_rows_and_features()

    summary = update_clustering_summary(
        rows,
        features,
        condition='correct_support',
        support_count=2,
        representations=('final_fast_delta',),
    )['final_fast_delta']

    heldout = summary['by_split']['family_validation']
    assert heldout['rows'] == 8
    assert heldout['families'] == 2
    assert heldout['same_instance_1nn_accuracy'] == 1.0
    assert heldout['family_1nn_excluding_same_instance_accuracy'] == 1.0
    assert set(summary['heldout_nearest_familiar_families']) == {'door', 'pick'}


def test_update_tsne_writes_heldout_family_views(tmp_path) -> None:
    rows, features = _synthetic_update_rows_and_features()

    manifest = write_update_tsne_artifacts(
        rows,
        features,
        tmp_path / 'tsne',
        benchmark_label='synthetic',
        condition='correct_support',
        support_count=2,
        representations=('final_fast_delta',),
        pca_components=6,
        perplexities=(3.0,),
        max_iter=250,
        seed=3,
    )

    assert manifest['rows'] == len(rows)
    assert not manifest['random_projection_used']
    embedding = manifest['embeddings'][0]
    assert (tmp_path / 'tsne' / embedding['family_plot']).is_file()
    assert (tmp_path / 'tsne' / embedding['heldout_focus_plot']).is_file()
    assert (tmp_path / 'tsne' / embedding['coordinates_csv']).is_file()
