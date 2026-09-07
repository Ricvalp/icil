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

from icil_jax_rlbench.data.metaworld_hidden_goal import (
    MetaWorldTaskDataset,
    MetaWorldTaskSampler,
    benchmark_from_config,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    READ_MODES,
    adapt_fast_state,
    fast_state_effective_rank,
    named_fast_tensor_metrics,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint, save_checkpoint
from icil_jax_rlbench.train.metaworld_query_runner import metaworld_model_config_from
from icil_jax_rlbench.train.provenance import (
    collect_experiment_provenance,
    config_to_dict,
    write_experiment_ledger,
)
from icil_jax_rlbench.train.ttt_config import adaptation_config_from, step_config_from
from icil_jax_rlbench.train.ttt_step import (
    create_ttt_train_state,
    create_ttt_train_step,
    ttt_meta_objective,
    write_query_gradient_alignment,
)

CHECKPOINT_TYPE = 'metaworld_ml1_reach_ttt'
_LOGGER = logging.getLogger(__name__)

_SUPPORT_FIELDS = ('observation', 'action', 'next_observation', 'write_mask')
_QUERY_FIELDS = ('observation', 'action', 'outer_loss_mask')


def validate_metaworld_ttt_config(cfg: ConfigDict) -> None:
    benchmark = benchmark_from_config(cfg)
    if str(cfg.mode) != benchmark.ttt_mode:
        raise ValueError(f'Expected mode={benchmark.ttt_mode!r}.')
    if not str(cfg.dataset.cache_root):
        raise ValueError('dataset.cache_root must point at a processed phi cache.')
    forbidden = {
        'support_token_cross_attention': cfg.conditioning.support_token_cross_attention,
        'support_summary_film': cfg.conditioning.support_summary_film,
        'support_trajectory_tokens': cfg.conditioning.support_trajectory_tokens,
        'support_memory_initialization': cfg.conditioning.support_memory_initialization,
    }
    enabled = [name for name, value in forbidden.items() if bool(value)]
    if enabled:
        raise ValueError(
            'MetaWorld TTT forbids direct support paths: ' + ', '.join(enabled)
        )
    if bool(cfg.conditioning.query_history):
        raise ValueError('MetaWorld TTT does not expose query history to READ.')
    if not bool(cfg.conditioning.fast_weight_write):
        raise ValueError('MetaWorld TTT requires fast-weight WRITE.')
    if not bool(cfg.conditioning.fast_weight_read):
        raise ValueError('MetaWorld TTT requires fast-weight READ.')
    if str(cfg.adaptation.write_objective) not in ('kvb', 'action_bc'):
        raise ValueError("adaptation.write_objective must be 'kvb' or 'action_bc'.")
    if str(cfg.adaptation.read_objective) != 'robotics_action_imitation':
        raise ValueError('The outer READ objective must be action imitation.')
    read_mode = str(cfg.adaptation.get('read_mode', 'absolute_gated'))
    if read_mode not in READ_MODES:
        raise ValueError(f'adaptation.read_mode must be one of {READ_MODES}.')
    read_scale = float(cfg.adaptation.get('read_scale', 1.0))
    if not np.isfinite(read_scale) or read_scale <= 0.0:
        raise ValueError('adaptation.read_scale must be finite and positive.')
    if int(cfg.adaptation.write_segment_size) < 1:
        raise ValueError('adaptation.write_segment_size must be positive.')
    if int(cfg.adaptation.write_steps_per_segment) < 1:
        raise ValueError('adaptation.write_steps_per_segment must be positive.')
    if str(cfg.adaptation.reset_policy) != 'reset_to_meta_learned_w0_per_task':
        raise ValueError('Fast state must reset to W0 at every task boundary.')
    if str(cfg.adaptation.query_carry_policy) != (
        'freeze_after_support_and_reuse_across_query_episodes'
    ):
        raise ValueError('Adapted fast state must be frozen across query episodes.')
    if str(cfg.action.translation_loss) != 'huber':
        raise ValueError(f'{benchmark.label} Cartesian actions require Huber loss.')
    if str(cfg.action.gripper_loss) != 'huber':
        raise ValueError(f'{benchmark.label} has a continuous gripper action.')
    if int(cfg.train.batch_size) < 1:
        raise ValueError('train.batch_size must be positive.')
    if int(cfg.train.support_episodes_per_task) < 1:
        raise ValueError('train.support_episodes_per_task must be positive.')
    if int(cfg.train.query_episodes_per_task) < 1:
        raise ValueError('train.query_episodes_per_task must be positive.')
    if int(cfg.train.get('debug_max_time_steps', 0)) < 0:
        raise ValueError('train.debug_max_time_steps cannot be negative.')
    if not str(cfg.train.resume_path) and not str(cfg.train.initial_checkpoint_path):
        raise ValueError(
            'Set train.initial_checkpoint_path to a competent query-only checkpoint.'
        )


def _resume_config(
    requested: ConfigDict,
    payload: Mapping[str, Any],
) -> ConfigDict:
    """Restore scientific settings while retaining runtime-only overrides."""

    checkpoint_cfg = ConfigDict(payload['config'])
    requested_benchmark = benchmark_from_config(requested)
    checkpoint_benchmark = benchmark_from_config(checkpoint_cfg)
    if (
        checkpoint_benchmark != requested_benchmark
        or str(checkpoint_cfg.mode) != requested_benchmark.ttt_mode
    ):
        raise ValueError('Resume checkpoint belongs to another MetaWorld benchmark.')

    if str(requested.dataset.cache_root):
        checkpoint_cfg.dataset.cache_root = str(requested.dataset.cache_root)
    checkpoint_cfg.dataset.cache_prepared_episodes = bool(
        requested.dataset.cache_prepared_episodes
    )
    for name in (
        'num_steps',
        'log_every',
        'eval_every',
        'eval_batches',
        'ckpt_every',
        'output_dir',
        'resume_path',
    ):
        setattr(checkpoint_cfg.train, name, getattr(requested.train, name))
    checkpoint_cfg.wandb = ConfigDict(config_to_dict(requested.wandb))
    _LOGGER.info(
        'Restored model, adaptation, optimizer, and meta-batch settings from %s.',
        requested.train.resume_path,
    )
    return checkpoint_cfg


def _weight_decay_mask(params: Mapping[str, Any]) -> Mapping[str, Any]:
    excluded = {'fast_init', 'inner_lr_raw', 'read_gate'}

    def make(path, value):
        top_level = str(getattr(path[0], 'key', path[0])) if path else ''
        return bool(top_level not in excluded and getattr(value, 'ndim', 0) >= 2)

    return jax.tree_util.tree_map_with_path(make, params)


def _optimizer(cfg: ConfigDict, params: Mapping[str, Any]):
    return optax.adamw(
        learning_rate=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        mask=_weight_decay_mask(params),
    )


def _training_contract(cfg, dataset, model_cfg, adapt_cfg, step_cfg):
    return {
        'mode': str(cfg.mode),
        'cache_data_sha256': dataset.bundle.data_sha256,
        'normalizer_id': dataset.normalization_id,
        'dataset_protocol': dataset.protocol,
        'horizon_buckets': list(dataset.horizon_buckets),
        'train_task_split': str(cfg.dataset.get('train_split', 'train')),
        'validation_task_split': str(
            cfg.dataset.get('validation_split', 'validation')
        ),
        'model_config': asdict(model_cfg),
        'adaptation_config': asdict(adapt_cfg),
        'step_config': asdict(step_cfg),
        'meta_batch': {
            'task_batch_size': int(cfg.train.batch_size),
            'support_episodes_per_task': int(cfg.train.support_episodes_per_task),
            'query_episodes_per_task': int(cfg.train.query_episodes_per_task),
        },
        'optimizer': {
            'name': 'adamw',
            'learning_rate': float(cfg.train.lr),
            'weight_decay': float(cfg.train.weight_decay),
        },
        'reset_policy': str(cfg.adaptation.reset_policy),
        'query_carry_policy': str(cfg.adaptation.query_carry_policy),
    }


def _normalized_training_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Fill defaults added after the first MetaWorld TTT checkpoints."""

    normalized = dict(contract)
    adaptation = dict(normalized.get('adaptation_config', {}))
    adaptation.setdefault('read_mode', 'absolute_gated')
    adaptation.setdefault('read_scale', 1.0)
    normalized['adaptation_config'] = adaptation
    return normalized


def _validate_checkpoint_dataset(
    payload: Mapping[str, Any],
    dataset: MetaWorldTaskDataset,
    model_config: Mapping[str, Any],
) -> None:
    extra = payload.get('extra', {})
    if extra.get('cache_data_sha256') != dataset.bundle.data_sha256:
        raise ValueError('Checkpoint and requested processed cache differ.')
    if extra.get('normalizer_id') != dataset.normalization_id:
        raise ValueError('Checkpoint and requested normalization differ.')
    if extra.get('model_config') != dict(model_config):
        raise ValueError('Checkpoint and requested model architecture differ.')
    if extra.get('dataset_protocol', dataset.protocol) != dataset.protocol:
        raise ValueError('Checkpoint and requested task-split protocol differ.')


def _maybe_wandb(cfg: ConfigDict, resume_payload: Mapping[str, Any] | None):
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
    resumed_id = (
        None
        if resume_payload is None
        else resume_payload.get('extra', {}).get('wandb_run_id')
    )
    if resumed_id:
        kwargs['id'] = str(resumed_id)
        kwargs['resume'] = 'must'
    wandb.init(**kwargs)
    return wandb


def _run_id(cfg: ConfigDict, wandb_mod, *, resumed: bool) -> str:
    benchmark = benchmark_from_config(cfg)
    objective = str(cfg.adaptation.write_objective)
    read_mode = str(cfg.adaptation.get('read_mode', 'absolute_gated'))
    read_suffix = '' if read_mode == 'absolute_gated' else f'_{read_mode}_read'
    order = 'fomaml' if bool(cfg.adaptation.first_order) else 'full'
    stamp = datetime.now(UTC).strftime('%Y%m%d-%H%M%S')
    wandb_id = (
        str(wandb_mod.run.id)
        if wandb_mod is not None and wandb_mod.run is not None
        else ''
    )
    suffix = wandb_id or stamp
    if resumed:
        suffix = f'{suffix}_resume_{stamp}'
    return f'{benchmark.slug}_{objective}{read_suffix}_{order}_{suffix}'


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _meta_batch_to_jax(
    batch: Mapping[str, Mapping[str, np.ndarray]],
    *,
    max_time_steps: int = 0,
) -> dict[str, dict[str, jax.Array]]:
    def convert(value: np.ndarray) -> jax.Array:
        result = jnp.asarray(value)
        if int(max_time_steps) > 0:
            result = result[:, :, : int(max_time_steps)]
        return result

    return {
        'support': {
            name: convert(batch['support'][name]) for name in _SUPPORT_FIELDS
        },
        'query': {
            name: convert(batch['query'][name]) for name in _QUERY_FIELDS
        },
    }


def _host_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(np.asarray(value))
        for name, value in jax.device_get(metrics).items()
    }


def _log_metrics(
    prefix: str,
    step: int,
    metrics: Mapping[str, float],
    wandb_mod,
) -> None:
    visible = {
        'loss',
        'query_loss_before',
        'query_loss_after',
        'improvement',
        'improvement_ratio',
        'write_loss',
        'fast_delta_norm',
        'read_delta_rms',
        'prediction_delta_rms',
        'slow_grad_norm',
        'step_s',
    }
    _LOGGER.info(
        '%s step %d | %s',
        prefix,
        int(step),
        ' | '.join(
            f'{name} {value:.6f}'
            for name, value in sorted(metrics.items())
            if name in visible
        ),
    )
    if wandb_mod is not None:
        wandb_mod.log(
            {f'{prefix}/{name}': value for name, value in metrics.items()},
            step=int(step),
        )


def _save_state(
    path: Path,
    state,
    *,
    step: int,
    cfg: ConfigDict,
    dataset: MetaWorldTaskDataset,
    model_cfg,
    adapt_cfg,
    step_cfg,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    train_sampler: MetaWorldTaskSampler,
    validation_sampler: MetaWorldTaskSampler,
    query_checkpoint_path: str,
    wandb_mod,
) -> None:
    wandb_id = (
        str(wandb_mod.run.id)
        if wandb_mod is not None and wandb_mod.run is not None
        else ''
    )
    save_checkpoint(
        path,
        state=state,
        step=int(step),
        config=cfg,
        extra={
            'checkpoint_type': dataset.benchmark.ttt_mode,
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
            'adaptation_config': asdict(adapt_cfg),
            'step_config': asdict(step_cfg),
            'training_contract': dict(contract),
            'train_sampler_rng_state': train_sampler.rng.bit_generator.state,
            'validation_sampler_rng_state': validation_sampler.rng.bit_generator.state,
            'initial_query_checkpoint': str(query_checkpoint_path),
            'wandb_run_id': wandb_id,
            'experiment_id': provenance['experiment_id'],
            'transient_fast_state_saved': False,
        },
        replicated=False,
    )


def train_metaworld_ttt(cfg: ConfigDict) -> Path:
    requested_cfg = ConfigDict(config_to_dict(cfg))
    requested_benchmark = benchmark_from_config(requested_cfg)
    resume_payload = None
    if str(requested_cfg.train.resume_path):
        resume_payload = load_checkpoint(str(requested_cfg.train.resume_path))
        if (
            resume_payload.get('extra', {}).get('checkpoint_type')
            != requested_benchmark.ttt_mode
        ):
            raise ValueError('Resume path is not a MetaWorld TTT checkpoint.')
        cfg = _resume_config(requested_cfg, resume_payload)
    else:
        cfg = requested_cfg
    validate_metaworld_ttt_config(cfg)
    benchmark = benchmark_from_config(cfg)

    source_payload = (
        resume_payload
        if resume_payload is not None
        else load_checkpoint(str(cfg.train.initial_checkpoint_path))
    )
    source_extra = source_payload.get('extra', {})
    expected_type = benchmark.ttt_mode if resume_payload is not None else benchmark.query_mode
    if source_extra.get('checkpoint_type') != expected_type:
        raise ValueError(
            f'Expected initialization checkpoint type {expected_type!r}, got '
            f'{source_extra.get("checkpoint_type")!r}.'
        )
    normalization = source_extra.get('normalization')
    if not isinstance(normalization, Mapping):
        raise ValueError('Initialization checkpoint lacks normalization statistics.')
    dataset = MetaWorldTaskDataset(
        str(cfg.dataset.cache_root),
        integration_name=benchmark.integration_name,
        protocol=str(cfg.dataset.get('protocol', 'default')),
        horizon_buckets=tuple(cfg.dataset.get('horizon_buckets', ())),
        normalization=normalization,
        cache_prepared_episodes=bool(cfg.dataset.cache_prepared_episodes),
    )
    integrity = dataset.integrity_report()
    if not bool(integrity['normalizer_uses_exact_training_task_episodes']):
        raise RuntimeError('Train-task-only normalization check failed.')
    if not bool(integrity['unique_episode_seeds']):
        raise RuntimeError('Cache contains duplicate episode seeds.')
    if float(integrity['expert_success_rate']) != 1.0:
        raise RuntimeError('Cache contains unsuccessful expert episodes.')
    required_episodes = int(cfg.train.support_episodes_per_task) + int(
        cfg.train.query_episodes_per_task
    )
    if required_episodes > int(dataset.task_index.episodes_per_task):
        raise ValueError(
            f'Training needs {required_episodes} episodes per task, but the cache has '
            f'{dataset.task_index.episodes_per_task}.'
        )

    model_cfg = metaworld_model_config_from(cfg, dataset)
    adapt_cfg = adaptation_config_from(cfg)
    step_cfg = step_config_from(cfg)
    _validate_checkpoint_dataset(
        source_payload, dataset, asdict(model_cfg)
    )
    params = jax.tree_util.tree_map(jnp.asarray, source_payload['params'])
    optimizer = _optimizer(cfg, params)
    if resume_payload is None:
        state = create_ttt_train_state(
            params,
            optimizer,
            jax.random.key(int(cfg.train.seed) + 1),
        )
        query_checkpoint_path = str(
            Path(cfg.train.initial_checkpoint_path).expanduser().resolve()
        )
    else:
        state = create_ttt_train_state(
            params,
            optimizer,
            jnp.asarray(source_payload['rng']),
            step=int(source_payload['step']),
            opt_state=jax.tree_util.tree_map(jnp.asarray, source_payload['opt_state']),
        )
        query_checkpoint_path = str(source_extra.get('initial_query_checkpoint', ''))

    contract = _training_contract(cfg, dataset, model_cfg, adapt_cfg, step_cfg)
    if (
        resume_payload is not None
        and _normalized_training_contract(source_extra.get('training_contract', {}))
        != _normalized_training_contract(contract)
    ):
        raise ValueError('Resolved resume contract differs from the checkpoint contract.')

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
    if resume_payload is not None:
        train_rng_state = source_extra.get('train_sampler_rng_state')
        validation_rng_state = source_extra.get('validation_sampler_rng_state')
        if train_rng_state is None or validation_rng_state is None:
            raise ValueError('Resume checkpoint lacks sampler RNG state.')
        train_sampler.rng.bit_generator.state = train_rng_state
        validation_sampler.rng.bit_generator.state = validation_rng_state

    eager_debug = bool(cfg.train.get('eager_debug', False))
    debug_max_time_steps = int(cfg.train.get('debug_max_time_steps', 0))
    if eager_debug:
        jax.config.update('jax_disable_jit', True)
        _LOGGER.warning(
            'Eager debug mode enabled; this run is not suitable for performance measurements.'
        )
    train_step = create_ttt_train_step(
        optimizer,
        model_cfg,
        adapt_cfg,
        step_cfg,
        distributed=False,
        jit=not eager_debug,
    )
    eval_objective = lambda value, batch: ttt_meta_objective(
        value, batch, model_cfg, adapt_cfg, step_cfg
    )
    if not eager_debug:
        eval_objective = jax.jit(eval_objective)

    wandb_mod = _maybe_wandb(cfg, resume_payload)
    run_id = _run_id(cfg, wandb_mod, resumed=resume_payload is not None)
    run_dir = Path(cfg.train.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    parent_checkpoint = (
        str(cfg.train.resume_path)
        if resume_payload is not None
        else str(cfg.train.initial_checkpoint_path)
    )
    provenance = collect_experiment_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        config=cfg,
        experiment_id=run_id,
        dataset=dataset.provenance(),
        parent_checkpoint=parent_checkpoint,
        adaptation_mode=(
            f'{cfg.adaptation.write_objective}_'
            f'{"fomaml" if cfg.adaptation.first_order else "full_second_order"}'
        ),
        reset_policy=str(cfg.adaptation.reset_policy),
    )
    write_experiment_ledger(run_dir, config=cfg, provenance=provenance)
    _write_json(run_dir / 'normalization.json', dataset.normalization.to_dict())
    _write_json(run_dir / 'dataset_integrity.json', integrity)
    _write_json(run_dir / 'training_contract.json', contract)
    _write_json(
        run_dir / 'task_splits.json',
        {
            split: list(dataset.task_ids(split))
            for split in dataset.benchmark.split_names
        },
    )
    _LOGGER.info('MetaWorld TTT run directory: %s', run_dir)

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
            batch = _meta_batch_to_jax(
                train_sampler.build_batch(
                    int(cfg.train.batch_size),
                    support_episodes=int(cfg.train.support_episodes_per_task),
                    query_episodes=int(cfg.train.query_episodes_per_task),
                ),
                max_time_steps=debug_max_time_steps,
            )
            state, metrics = train_step(state, batch)
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
                    validation_batch = _meta_batch_to_jax(
                        validation_sampler.build_batch(
                            int(cfg.train.batch_size),
                            support_episodes=int(cfg.train.support_episodes_per_task),
                            query_episodes=int(cfg.train.query_episodes_per_task),
                        ),
                        max_time_steps=debug_max_time_steps,
                    )
                    _, validation_metrics = eval_objective(
                        state.params, validation_batch
                    )
                    values.append(_host_metrics(validation_metrics))
                averaged = {
                    name: float(np.mean([value[name] for value in values]))
                    for name in values[0]
                }
                diagnostic_batch = _meta_batch_to_jax(
                    validation_sampler.build_batch(
                        1,
                        support_episodes=int(cfg.train.support_episodes_per_task),
                        query_episodes=int(cfg.train.query_episodes_per_task),
                    ),
                    max_time_steps=debug_max_time_steps,
                )
                support = jax.tree_util.tree_map(
                    lambda value: value[0], diagnostic_batch['support']
                )
                query = jax.tree_util.tree_map(
                    lambda value: value[0], diagnostic_batch['query']
                )
                alignment = write_query_gradient_alignment(
                    state.params, support, query, model_cfg, adapt_cfg
                )
                adapted, _ = adapt_fast_state(
                    state.params, support, model_cfg, adapt_cfg
                )
                fast_metrics = named_fast_tensor_metrics(
                    state.params, adapted, model_cfg
                )
                fast_metrics['fast/effective_rank'] = fast_state_effective_rank(
                    adapted
                )
                averaged.update(
                    {
                        name: float(np.asarray(value))
                        for name, value in jax.device_get(
                            {**alignment, **fast_metrics}
                        ).items()
                    }
                )
                _log_metrics('validation', step, averaged, wandb_mod)

            if step % int(cfg.train.ckpt_every) == 0:
                _save_state(
                    run_dir / f'step_{step:07d}.pkl',
                    state,
                    step=step,
                    cfg=cfg,
                    dataset=dataset,
                    model_cfg=model_cfg,
                    adapt_cfg=adapt_cfg,
                    step_cfg=step_cfg,
                    contract=contract,
                    provenance=provenance,
                    train_sampler=train_sampler,
                    validation_sampler=validation_sampler,
                    query_checkpoint_path=query_checkpoint_path,
                    wandb_mod=wandb_mod,
                )
        _save_state(
            run_dir / 'last.pkl',
            state,
            step=target_step,
            cfg=cfg,
            dataset=dataset,
            model_cfg=model_cfg,
            adapt_cfg=adapt_cfg,
            step_cfg=step_cfg,
            contract=contract,
            provenance=provenance,
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            query_checkpoint_path=query_checkpoint_path,
            wandb_mod=wandb_mod,
        )
        _LOGGER.info('MetaWorld TTT training complete: %s', run_dir / 'last.pkl')
        return run_dir / 'last.pkl'
    finally:
        if wandb_mod is not None:
            wandb_mod.finish()


__all__ = [
    'CHECKPOINT_TYPE',
    'train_metaworld_ttt',
    'validate_metaworld_ttt_config',
]
