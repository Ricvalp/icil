from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict

from icil_jax_rlbench.data.metaworld_conditioning import (
    CONDITIONING_MODES,
    MetaWorldConditioning,
)
from icil_jax_rlbench.data.metaworld_hidden_goal import (
    MetaWorldTaskDataset,
    MetaWorldTaskSampler,
    benchmark_from_config,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    FastWeightTTTConfig,
    init_fast_weight_ttt_params,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint, save_checkpoint
from icil_jax_rlbench.train.provenance import (
    collect_experiment_provenance,
    config_to_dict,
    write_experiment_ledger,
)
from icil_jax_rlbench.train.query_only_step import (
    create_query_only_train_step,
    query_only_objective,
)
from icil_jax_rlbench.train.ttt_step import create_ttt_train_state

CHECKPOINT_TYPE = 'metaworld_ml1_reach_query_only'
CONDITIONED_CHECKPOINT_TYPES = {
    'family': 'metaworld_ml45_family_conditioned_query_only',
    'family_task_latent': 'metaworld_ml45_oracle_conditioned_query_only',
}
_LOGGER = logging.getLogger(__name__)


def _conditioning_mode(cfg: ConfigDict) -> str:
    conditioning = cfg.get('conditioning')
    if conditioning is None:
        return 'none'
    return str(conditioning.get('mode', 'none'))


def query_checkpoint_type(cfg: ConfigDict) -> str:
    benchmark = benchmark_from_config(cfg)
    mode = _conditioning_mode(cfg)
    if mode == 'none':
        return benchmark.query_mode
    try:
        return CONDITIONED_CHECKPOINT_TYPES[mode]
    except KeyError as exc:
        choices = ('none', *CONDITIONING_MODES)
        raise ValueError(f'conditioning.mode must be one of {choices}.') from exc


def _conditioning_from_config(
    cfg: ConfigDict,
    dataset: MetaWorldTaskDataset,
) -> MetaWorldConditioning | None:
    mode = _conditioning_mode(cfg)
    if mode == 'none':
        return None
    conditioning = cfg.conditioning
    return MetaWorldConditioning.fit(
        dataset,
        mode=mode,
        train_split=str(cfg.dataset.get('train_split', 'train')),
        normalization_eps=float(conditioning.get('normalization_eps', 1e-4)),
    )


def validate_metaworld_query_config(cfg: ConfigDict) -> None:
    benchmark = benchmark_from_config(cfg)
    checkpoint_type = query_checkpoint_type(cfg)
    if str(cfg.mode) != checkpoint_type:
        raise ValueError(f'Expected mode={checkpoint_type!r}.')
    if (
        _conditioning_mode(cfg) != 'none'
        and benchmark.integration_name != 'metaworld_ml45'
    ):
        raise ValueError('Explicit task conditioning is currently defined for ML45.')
    if not str(cfg.dataset.cache_root):
        raise ValueError(
            'dataset.cache_root is required. Point it at a processed phi-mujoco cache.'
        )
    if str(cfg.action.translation_loss) != 'huber':
        raise ValueError(f'{benchmark.label} uses Huber loss for Cartesian actions.')
    if str(cfg.action.gripper_loss) != 'huber':
        raise ValueError(
            f'{benchmark.label} uses Huber loss for its continuous gripper action.'
        )
    if int(cfg.train.batch_size) < 1:
        raise ValueError('train.batch_size must be positive.')
    if int(cfg.train.query_episodes_per_task) < 1:
        raise ValueError('train.query_episodes_per_task must be positive.')
    train_split = str(cfg.dataset.get('train_split', 'train'))
    validation_split = str(cfg.dataset.get('validation_split', 'validation'))
    if train_split == validation_split:
        _LOGGER.warning(
            'Training and monitoring use the same task split %r; this is intended '
            'only for the frozen-hyperparameter final protocol.',
            train_split,
        )


def metaworld_model_config_from(
    cfg: ConfigDict,
    dataset: MetaWorldTaskDataset,
    *,
    observation_dim: int | None = None,
) -> FastWeightTTTConfig:
    return FastWeightTTTConfig(
        observation_dim=(
            int(dataset.observation_dim)
            if observation_dim is None
            else int(observation_dim)
        ),
        action_dim=int(dataset.action_dim),
        translation_dim=3,
        hidden_dim=int(cfg.model.hidden_dim),
        fast_dim=int(cfg.model.fast_dim),
        fast_hidden_dim=int(cfg.model.fast_hidden_dim),
        fast_model=str(cfg.model.fast_model),
        gate_init=float(cfg.model.gate_init),
        inner_lr_init=float(cfg.model.inner_lr_init),
        inner_lr_min=float(cfg.model.inner_lr_min),
        translation_output='linear',
        translation_loss_weight=float(cfg.action.translation_loss_weight),
        gripper_loss_weight=float(cfg.action.gripper_loss_weight),
        translation_huber_delta=float(cfg.action.translation_huber_delta),
        gripper_loss='huber',
        gripper_huber_delta=float(cfg.action.gripper_huber_delta),
    )


def _weight_decay_mask(params: Mapping[str, Any]) -> Mapping[str, Any]:
    trained_groups = {'query_encoder', 'translation_head', 'gripper_head'}

    def make(path, value):
        top_level = str(getattr(path[0], 'key', path[0])) if path else ''
        return bool(top_level in trained_groups and getattr(value, 'ndim', 0) >= 2)

    return jax.tree_util.tree_map_with_path(make, params)


def _optimizer(cfg: ConfigDict, params: Mapping[str, Any]):
    return optax.adamw(
        learning_rate=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        mask=_weight_decay_mask(params),
    )


def _maybe_wandb(cfg: ConfigDict):
    if not bool(cfg.wandb.enable):
        return None
    import wandb

    kwargs = {
        'project': str(cfg.wandb.project),
        'config': config_to_dict(cfg),
        'mode': str(cfg.wandb.mode),
    }
    if str(cfg.wandb.entity):
        kwargs['entity'] = str(cfg.wandb.entity)
    if str(cfg.wandb.name):
        kwargs['name'] = str(cfg.wandb.name)
    wandb.init(**kwargs)
    return wandb


def _run_id(
    wandb_mod,
    dataset: MetaWorldTaskDataset,
    conditioning: MetaWorldConditioning | None,
) -> str:
    if wandb_mod is not None and wandb_mod.run is not None:
        return str(wandb_mod.run.id)
    label = (
        'query_only'
        if conditioning is None
        else f'{conditioning.mode}_query_only'
    )
    return (
        f'{dataset.benchmark.slug}_{label}_'
        f'{datetime.now(UTC).strftime("%Y%m%d-%H%M%S")}'
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _host_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(np.asarray(value))
        for name, value in jax.device_get(metrics).items()
    }


def _query_to_jax(
    query: Mapping[str, np.ndarray],
    *,
    dataset: MetaWorldTaskDataset,
    task_ids: tuple[str, ...],
    conditioning: MetaWorldConditioning | None,
) -> dict[str, jax.Array]:
    observation = np.asarray(query['observation'])
    if conditioning is not None:
        observation = conditioning.augment_observations(
            dataset, observation, task_ids
        )
    return {
        'observation': jnp.asarray(observation),
        'action': jnp.asarray(query['action']),
        'outer_loss_mask': jnp.asarray(query['outer_loss_mask']),
    }


def _log_metrics(
    prefix: str,
    step: int,
    metrics: Mapping[str, float],
    wandb_mod,
) -> None:
    selected = {
        name: value
        for name, value in metrics.items()
        if name
        in {
            'loss',
            'translation_loss',
            'gripper_loss',
            'translation_l1',
            'gripper_l1',
            'slow_grad_norm',
            'step_s',
        }
    }
    _LOGGER.info(
        '%s step %d | %s',
        prefix,
        int(step),
        ' | '.join(
            f'{name} {value:.6f}' for name, value in sorted(selected.items())
        ),
    )
    if wandb_mod is not None:
        wandb_mod.log(
            {f'{prefix}/{name}': value for name, value in metrics.items()},
            step=int(step),
        )


def _restore_state(
    checkpoint_path: str,
    optimizer,
    dataset: MetaWorldTaskDataset,
    model_cfg: FastWeightTTTConfig,
    *,
    checkpoint_type: str,
    conditioning: MetaWorldConditioning | None,
):
    payload = load_checkpoint(checkpoint_path)
    extra = payload.get('extra', {})
    if extra.get('checkpoint_type') != checkpoint_type:
        raise ValueError(
            f'Resume checkpoint type differs from requested type {checkpoint_type!r}.'
        )
    if extra.get('cache_data_sha256') != dataset.bundle.data_sha256:
        raise ValueError(
            'Resume checkpoint was trained from a different processed cache.'
        )
    if extra.get('normalizer_id') != dataset.normalization_id:
        raise ValueError(
            'Resume checkpoint normalization differs from the current cache.'
        )
    if extra.get('dataset_protocol', dataset.protocol) != dataset.protocol:
        raise ValueError('Resume checkpoint uses another task-split protocol.')
    if extra.get('model_config') != asdict(model_cfg):
        raise ValueError(
            'Resume checkpoint model differs from the requested model config.'
        )
    expected_conditioning = None if conditioning is None else conditioning.to_dict()
    if extra.get('conditioning') != expected_conditioning:
        raise ValueError('Resume checkpoint conditioning metadata differs.')
    return create_ttt_train_state(
        jax.tree_util.tree_map(jnp.asarray, payload['params']),
        optimizer,
        jnp.asarray(payload['rng']),
        step=int(payload['step']),
        opt_state=jax.tree_util.tree_map(jnp.asarray, payload['opt_state']),
    )


def _save_state(
    path: Path,
    state,
    *,
    step: int,
    cfg: ConfigDict,
    dataset: MetaWorldTaskDataset,
    model_cfg: FastWeightTTTConfig,
    provenance: Mapping[str, Any],
    checkpoint_type: str,
    conditioning: MetaWorldConditioning | None,
) -> None:
    save_checkpoint(
        path,
        state=state,
        step=int(step),
        config=cfg,
        extra={
            'checkpoint_type': checkpoint_type,
            'normalization': dataset.normalization.to_dict(),
            'normalizer_id': dataset.normalization_id,
            'cache_data_sha256': dataset.bundle.data_sha256,
            'dataset_protocol': dataset.protocol,
            'horizon_buckets': list(dataset.horizon_buckets),
            'train_task_split': str(cfg.dataset.get('train_split', 'train')),
            'validation_task_split': str(
                cfg.dataset.get('validation_split', 'validation')
            ),
            'model_config': asdict(model_cfg),
            'conditioning': (
                None if conditioning is None else conditioning.to_dict()
            ),
            'experiment_id': provenance['experiment_id'],
            'transient_fast_state_saved': False,
        },
        replicated=False,
    )


def train_metaworld_query_only(cfg: ConfigDict) -> Path:
    validate_metaworld_query_config(cfg)
    benchmark = benchmark_from_config(cfg)
    dataset = MetaWorldTaskDataset(
        str(cfg.dataset.cache_root),
        integration_name=benchmark.integration_name,
        protocol=str(cfg.dataset.get('protocol', 'default')),
        horizon_buckets=tuple(cfg.dataset.get('horizon_buckets', ())),
        normalization_eps=float(cfg.dataset.normalization_eps),
        cache_prepared_episodes=bool(cfg.dataset.cache_prepared_episodes),
    )
    integrity = dataset.integrity_report()
    if not bool(integrity['normalizer_uses_exact_training_task_episodes']):
        raise RuntimeError('Train-task-only normalization check failed.')
    if not bool(integrity['unique_episode_seeds']):
        raise RuntimeError('Cache contains duplicate episode seeds.')
    if float(integrity['expert_success_rate']) != 1.0:
        raise RuntimeError('Cache contains unsuccessful expert episodes.')

    conditioning = _conditioning_from_config(cfg, dataset)
    checkpoint_type = query_checkpoint_type(cfg)
    model_cfg = metaworld_model_config_from(
        cfg,
        dataset,
        observation_dim=(
            None
            if conditioning is None
            else dataset.observation_dim + conditioning.context_dim
        ),
    )
    params = init_fast_weight_ttt_params(
        jax.random.key(int(cfg.train.seed)), model_cfg
    )
    optimizer = _optimizer(cfg, params)
    state = create_ttt_train_state(
        params, optimizer, jax.random.key(int(cfg.train.seed) + 1)
    )
    if str(cfg.train.resume_path):
        checkpoint = load_checkpoint(str(cfg.train.resume_path))
        checkpoint_normalization = checkpoint.get('extra', {}).get('normalization')
        if checkpoint_normalization is None:
            raise ValueError(
                'Resume checkpoint does not contain normalization statistics.'
            )
        dataset = MetaWorldTaskDataset(
            str(cfg.dataset.cache_root),
            integration_name=benchmark.integration_name,
            protocol=str(cfg.dataset.get('protocol', 'default')),
            horizon_buckets=tuple(cfg.dataset.get('horizon_buckets', ())),
            normalization=checkpoint_normalization,
            cache_prepared_episodes=bool(cfg.dataset.cache_prepared_episodes),
        )
        conditioning = _conditioning_from_config(cfg, dataset)
        state = _restore_state(
            str(cfg.train.resume_path),
            optimizer,
            dataset,
            model_cfg,
            checkpoint_type=checkpoint_type,
            conditioning=conditioning,
        )

    train_step = create_query_only_train_step(
        optimizer,
        model_cfg,
        slow_grad_clip_norm=float(cfg.train.slow_grad_clip_norm),
    )
    train_sampler = MetaWorldTaskSampler(
        dataset,
        split=str(cfg.dataset.get('train_split', 'train')),
        seed=int(cfg.train.seed) + 1001,
    )
    validation_sampler = MetaWorldTaskSampler(
        dataset,
        split=str(cfg.dataset.get('validation_split', 'validation')),
        seed=int(cfg.train.seed) + 2001,
    )

    wandb_mod = _maybe_wandb(cfg)
    run_id = _run_id(wandb_mod, dataset, conditioning)
    run_dir = Path(cfg.train.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset_provenance = dataset.provenance()
    dataset_provenance['policy_conditioning'] = (
        None if conditioning is None else conditioning.to_dict()
    )
    provenance = collect_experiment_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        config=cfg,
        experiment_id=run_id,
        dataset=dataset_provenance,
        parent_checkpoint=str(cfg.train.resume_path),
        adaptation_mode=(
            'query_only_no_support'
            if conditioning is None
            else f'query_only_direct_{conditioning.mode}_conditioning'
        ),
        reset_policy='not_applicable',
    )
    write_experiment_ledger(run_dir, config=cfg, provenance=provenance)
    _write_json(run_dir / 'normalization.json', dataset.normalization.to_dict())
    if conditioning is not None:
        _write_json(run_dir / 'conditioning.json', conditioning.to_dict())
    _write_json(run_dir / 'dataset_integrity.json', integrity)
    _write_json(
        run_dir / 'task_splits.json',
        {
            split: list(dataset.task_ids(split))
            for split in dataset.benchmark.split_names
        },
    )
    _LOGGER.info('%s query-only run directory: %s', benchmark.label, run_dir)

    start_step = int(jax.device_get(state.step))
    target_step = int(cfg.train.num_steps)
    if start_step >= target_step:
        raise ValueError(
            f'Checkpoint step {start_step} is already >= train.num_steps {target_step}.'
        )
    last_log_time = time.time()
    last_log_step = start_step
    try:
        for step in range(start_step + 1, target_step + 1):
            batch = train_sampler.build_query_batch(
                int(cfg.train.batch_size),
                query_episodes=int(cfg.train.query_episodes_per_task),
            )
            query = _query_to_jax(
                batch['query'],
                dataset=dataset,
                task_ids=batch['task_ids'],
                conditioning=conditioning,
            )
            state, metrics = train_step(state, query)
            if step == start_step + 1:
                jax.block_until_ready(metrics['loss'])
            if step == start_step + 1 or step % int(cfg.train.log_every) == 0:
                now = time.time()
                host_metrics = _host_metrics(metrics)
                host_metrics['step_s'] = (now - last_log_time) / max(
                    1, step - last_log_step
                )
                _log_metrics('train', step, host_metrics, wandb_mod)
                last_log_time = now
                last_log_step = step

            if step % int(cfg.train.eval_every) == 0:
                values = []
                for _ in range(int(cfg.train.eval_batches)):
                    validation_batch = validation_sampler.build_query_batch(
                        int(cfg.train.batch_size),
                        query_episodes=int(cfg.train.query_episodes_per_task),
                    )
                    _, validation_metrics = query_only_objective(
                        state.params,
                        _query_to_jax(
                            validation_batch['query'],
                            dataset=dataset,
                            task_ids=validation_batch['task_ids'],
                            conditioning=conditioning,
                        ),
                        model_cfg,
                    )
                    values.append(_host_metrics(validation_metrics))
                averaged = {
                    name: float(np.mean([value[name] for value in values]))
                    for name in values[0]
                }
                _log_metrics('validation', step, averaged, wandb_mod)

            if step % int(cfg.train.ckpt_every) == 0:
                _save_state(
                    run_dir / f'step_{step:07d}.pkl',
                    state,
                    step=step,
                    cfg=cfg,
                    dataset=dataset,
                    model_cfg=model_cfg,
                    provenance=provenance,
                    checkpoint_type=checkpoint_type,
                    conditioning=conditioning,
                )
        _save_state(
            run_dir / 'last.pkl',
            state,
            step=target_step,
            cfg=cfg,
            dataset=dataset,
            model_cfg=model_cfg,
            provenance=provenance,
            checkpoint_type=checkpoint_type,
            conditioning=conditioning,
        )
        _LOGGER.info(
            '%s query-only training complete: %s',
            benchmark.label,
            run_dir / 'last.pkl',
        )
        return run_dir / 'last.pkl'
    finally:
        if wandb_mod is not None:
            wandb_mod.finish()


__all__ = [
    'CHECKPOINT_TYPE',
    'CONDITIONED_CHECKPOINT_TYPES',
    'metaworld_model_config_from',
    'query_checkpoint_type',
    'train_metaworld_query_only',
    'validate_metaworld_query_config',
]
