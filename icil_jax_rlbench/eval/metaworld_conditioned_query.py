from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from phi_mujoco.evaluation import EvaluationConfig, EvaluationRunner

from icil_jax_rlbench.data.metaworld_conditioning import MetaWorldConditioning
from icil_jax_rlbench.data.metaworld_hidden_goal import (
    MetaWorldTaskDataset,
    MetaWorldTaskSampler,
    benchmark_from_config,
)
from icil_jax_rlbench.eval.metaworld_policy import MetaWorldConditionedJaxPolicy
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from icil_jax_rlbench.train.metaworld_query_runner import (
    CONDITIONED_CHECKPOINT_TYPES,
    metaworld_model_config_from,
    validate_metaworld_query_config,
)
from icil_jax_rlbench.train.provenance import config_to_dict
from icil_jax_rlbench.train.query_only_step import query_only_objective

_LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _confidence_interval(successes: int, attempts: int) -> tuple[float, float]:
    if attempts < 1:
        return 0.0, 0.0
    probability = successes / attempts
    standard_error = np.sqrt(
        max(probability * (1.0 - probability), 1e-12) / attempts
    )
    return (
        float(max(0.0, probability - 1.96 * standard_error)),
        float(min(1.0, probability + 1.96 * standard_error)),
    )


def _fresh_evaluation_seed(
    requested_seed: int,
    episodes: int,
    dataset: MetaWorldTaskDataset,
) -> int:
    used = {episode.seed for episode in dataset.bundle.episodes}
    candidate = int(requested_seed)
    while any(candidate + offset in used for offset in range(int(episodes))):
        candidate += int(episodes)
    if candidate + int(episodes) - 1 > 2**32 - 1:
        raise ValueError('Could not choose fresh closed-loop evaluation seeds.')
    return candidate


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(
        int(record['closed_loop']['successful_episodes']) for record in records
    )
    attempts = sum(
        int(record['closed_loop']['attempted_episodes']) for record in records
    )
    return {
        'task_count': len(records),
        'offline_query_loss': float(
            np.mean([record['offline_query_loss'] for record in records])
        ),
        'translation_loss': float(
            np.mean([record['translation_loss'] for record in records])
        ),
        'gripper_loss': float(
            np.mean([record['gripper_loss'] for record in records])
        ),
        'success_rate': successes / attempts,
        'success_rate_95pct_ci': list(
            _confidence_interval(successes, attempts)
        ),
        'successful_episodes': successes,
        'attempted_episodes': attempts,
    }


def evaluate_metaworld_conditioned_query(cfg: ConfigDict) -> Path:
    if not str(cfg.checkpoint_path):
        raise ValueError('checkpoint_path is required.')
    if int(cfg.offline_query_episodes) < 1:
        raise ValueError('offline_query_episodes must be positive.')
    if int(cfg.closed_loop_episodes) < 1:
        raise ValueError('closed_loop_episodes must be positive.')

    payload = load_checkpoint(str(cfg.checkpoint_path))
    extra = payload.get('extra', {})
    train_cfg = ConfigDict(payload['config'])
    validate_metaworld_query_config(train_cfg)
    benchmark = benchmark_from_config(train_cfg)
    if benchmark.integration_name != 'metaworld_ml45':
        raise ValueError('Conditioned query evaluation requires an ML45 checkpoint.')
    if str(cfg.integration) != benchmark.integration_name:
        raise ValueError(
            f'Evaluation requests {cfg.integration!r}, but the checkpoint belongs '
            f'to {benchmark.integration_name!r}.'
        )

    conditioning_metadata = extra.get('conditioning')
    if not isinstance(conditioning_metadata, Mapping):
        raise ValueError('Checkpoint does not contain task-conditioning metadata.')
    conditioning = MetaWorldConditioning.from_dict(conditioning_metadata)
    expected_checkpoint_type = CONDITIONED_CHECKPOINT_TYPES[conditioning.mode]
    if extra.get('checkpoint_type') != expected_checkpoint_type:
        raise ValueError('Checkpoint type and conditioning mode disagree.')

    checkpoint_normalization = extra.get('normalization')
    if not isinstance(checkpoint_normalization, Mapping):
        raise ValueError('Checkpoint does not contain normalization statistics.')
    cache_root = str(cfg.cache_root) or str(train_cfg.dataset.cache_root)
    dataset = MetaWorldTaskDataset(
        cache_root,
        integration_name=benchmark.integration_name,
        protocol=str(train_cfg.dataset.protocol),
        horizon_buckets=tuple(train_cfg.dataset.horizon_buckets),
        normalization=checkpoint_normalization,
        cache_prepared_episodes=bool(cfg.cache_prepared_episodes),
    )
    if dataset.bundle.data_sha256 != extra.get('cache_data_sha256'):
        raise ValueError('Evaluation cache differs from the checkpoint training cache.')
    if dataset.normalization_id != extra.get('normalizer_id'):
        raise ValueError('Evaluation normalization differs from the checkpoint.')
    conditioning.validate_dataset(dataset)

    model_cfg = metaworld_model_config_from(
        train_cfg,
        dataset,
        observation_dim=dataset.observation_dim + conditioning.context_dim,
    )
    if asdict(model_cfg) != extra.get('model_config'):
        raise ValueError('Checkpoint model metadata differs from its resolved config.')
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])

    split = str(cfg.split)
    task_ids = dataset.balanced_task_ids(split, int(cfg.max_tasks))
    if not task_ids:
        raise ValueError(f'No tasks selected from split {split!r}.')
    evaluation_families = tuple(
        dict.fromkeys(dataset.task_family(task_id) for task_id in task_ids)
    )
    unseen_families = tuple(
        family
        for family in evaluation_families
        if family not in conditioning.training_families
    )
    if unseen_families and not bool(cfg.allow_unseen_families):
        raise ValueError(
            'Evaluation contains family one-hot entries never optimized during '
            f'training: {unseen_families}. Set allow_unseen_families=True only '
            'when this limitation is intentional.'
        )

    sampler = MetaWorldTaskSampler(
        dataset, split=split, seed=int(cfg.seed) + 100
    )
    closed_loop_episodes = int(cfg.closed_loop_episodes)
    evaluation_seed = _fresh_evaluation_seed(
        int(cfg.closed_loop_base_seed), closed_loop_episodes, dataset
    )
    output_dir = (
        Path(cfg.output_dir).expanduser().resolve()
        / (
            f'{Path(cfg.checkpoint_path).stem}_'
            f'{datetime.now(UTC).strftime("%Y%m%d-%H%M%S")}'
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    records = []
    base_integration = benchmark.create_integration(
        catalog_seed=dataset.task_index.catalog.catalog_seed
    )
    for task_position, task_id in enumerate(task_ids):
        batch = sampler.build_query_batch(
            1,
            query_episodes=int(cfg.offline_query_episodes),
            task_ids=[task_id],
        )
        query_section = batch['query']
        query = {
            'observation': jnp.asarray(
                conditioning.augment_observations(
                    dataset, query_section['observation'], batch['task_ids']
                )
            ),
            'action': jnp.asarray(query_section['action']),
            'outer_loss_mask': jnp.asarray(query_section['outer_loss_mask']),
        }
        offline_loss, offline_metrics = query_only_objective(
            params, query, model_cfg
        )

        integration = base_integration.for_task(task_id)
        policy = MetaWorldConditionedJaxPolicy(
            integration=integration,
            params=params,
            model_cfg=model_cfg,
            normalization=dataset.normalization,
            context=conditioning.context(dataset, task_id),
        )
        save_rollouts = bool(cfg.save_rollout_artifacts) or bool(cfg.record_video)
        rollout_directory = (
            output_dir / 'rollouts' / task_id if save_rollouts else None
        )
        closed_loop = EvaluationRunner(
            integration,
            policy,
            EvaluationConfig(
                episodes=closed_loop_episodes,
                base_seed=evaluation_seed,
                max_steps=(
                    None
                    if int(cfg.closed_loop_max_steps) <= 0
                    else int(cfg.closed_loop_max_steps)
                ),
                run_directory=rollout_directory,
                record_video=bool(cfg.record_video),
                render_width=int(cfg.render_width),
                render_height=int(cfg.render_height),
            ),
        ).run()
        host_metrics = jax.device_get(offline_metrics)
        records.append(
            {
                'task_id': task_id,
                'task_family': dataset.task_family(task_id),
                'offline_query_loss': float(offline_loss),
                'translation_loss': float(host_metrics['translation_loss']),
                'gripper_loss': float(host_metrics['gripper_loss']),
                'closed_loop': closed_loop.as_dict(),
            }
        )
        _LOGGER.info(
            'ML45 conditioned query task %d/%d complete: %s',
            task_position + 1,
            len(task_ids),
            task_id,
        )

    records_by_family = {
        family: [
            record for record in records if record['task_family'] == family
        ]
        for family in evaluation_families
    }
    summary = {
        'evaluation': 'metaworld_ml45_conditioned_query_only',
        'checkpoint': str(Path(cfg.checkpoint_path).expanduser().resolve()),
        'checkpoint_step': int(payload['step']),
        'checkpoint_type': expected_checkpoint_type,
        'integration': benchmark.integration_name,
        'dataset': dataset.provenance(),
        'split': split,
        'task_ids': list(task_ids),
        'conditioning': conditioning.to_dict(),
        'conditioning_is_direct_oracle': conditioning.mode == 'family_task_latent',
        'support_used': False,
        'fast_weight_updates': 0,
        'unseen_families': list(unseen_families),
        'offline_query_episodes': int(cfg.offline_query_episodes),
        'closed_loop_episodes': closed_loop_episodes,
        'closed_loop_base_seed': evaluation_seed,
        'aggregate': _aggregate(records),
        'aggregate_by_family': {
            family: _aggregate(family_records)
            for family, family_records in records_by_family.items()
        },
        'per_task': records,
    }
    _write_json(output_dir / 'summary.json', summary)
    _write_json(output_dir / 'resolved_eval_config.json', config_to_dict(cfg))
    _LOGGER.info('ML45 conditioned query evaluation complete: %s', output_dir)
    return output_dir


__all__ = ['evaluate_metaworld_conditioned_query']
