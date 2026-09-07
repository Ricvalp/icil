from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import ConfigDict
import numpy as np
import optax

from icil_jax_rlbench.data.hidden_goal import (
    HiddenGoalConfig,
    HiddenGoalMetaSampler,
    HiddenGoalTaskBank,
    StateNormalizer,
    benchmark_integrity_report,
    fit_state_normalizer,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    READ_MODES,
    FastWeightTTTConfig,
    adapt_fast_state,
    fast_state_effective_rank,
    init_fast_weight_ttt_params,
    named_fast_tensor_metrics,
)
from icil_jax_rlbench.train.checkpoints import load_checkpoint, save_checkpoint
from icil_jax_rlbench.train.provenance import (
    collect_experiment_provenance,
    config_to_dict,
    write_experiment_ledger,
)
from icil_jax_rlbench.train.ttt_step import (
    TTTTrainState,
    create_ttt_train_state,
    create_ttt_train_step,
    ttt_meta_objective,
    write_query_gradient_alignment,
)
from icil_jax_rlbench.train.ttt_config import (
    adaptation_config_from,
    step_config_from,
)


def _dataclass_from_section(cls, section: Any, **overrides):
    values = {
        field.name: getattr(section, field.name)
        for field in fields(cls)
        if hasattr(section, field.name)
    }
    values.update(overrides)
    return cls(**values)


def hidden_goal_config_from(cfg: ConfigDict) -> HiddenGoalConfig:
    return _dataclass_from_section(HiddenGoalConfig, cfg.benchmark)


def fast_weight_config_from(
    cfg: ConfigDict, benchmark_cfg: HiddenGoalConfig
) -> FastWeightTTTConfig:
    return FastWeightTTTConfig(
        observation_dim=int(benchmark_cfg.observation_dim),
        action_dim=int(benchmark_cfg.action_dim),
        translation_dim=2,
        hidden_dim=int(cfg.model.hidden_dim),
        fast_dim=int(cfg.model.fast_dim),
        fast_hidden_dim=int(cfg.model.fast_hidden_dim),
        fast_model=str(cfg.model.fast_model),
        gate_init=float(cfg.model.gate_init),
        inner_lr_init=float(cfg.model.inner_lr_init),
        inner_lr_min=float(cfg.model.inner_lr_min),
        translation_loss_weight=float(cfg.action.translation_loss_weight),
        gripper_loss_weight=float(cfg.action.gripper_loss_weight),
        translation_huber_delta=float(cfg.action.translation_huber_delta),
    )


def validate_adaptation_only_config(cfg: ConfigDict) -> None:
    if str(cfg.mode) != 'ttt_adaptation_only':
        raise ValueError("The state TTT runner requires mode='ttt_adaptation_only'.")
    forbidden = {
        'support_token_cross_attention': cfg.conditioning.support_token_cross_attention,
        'support_summary_film': cfg.conditioning.support_summary_film,
        'support_trajectory_tokens': cfg.conditioning.support_trajectory_tokens,
        'support_memory_initialization': cfg.conditioning.support_memory_initialization,
    }
    enabled = [name for name, value in forbidden.items() if bool(value)]
    if enabled:
        raise ValueError(
            'Adaptation-only mode forbids direct support paths: ' + ', '.join(enabled)
        )
    if bool(cfg.conditioning.query_history):
        raise ValueError('The controlled mechanism experiment disables query history.')
    if str(cfg.adaptation.read_objective) != 'robotics_action_imitation':
        raise ValueError('The initial READ objective must be robotics_action_imitation.')
    read_mode = str(cfg.adaptation.get('read_mode', 'absolute_gated'))
    if read_mode not in READ_MODES:
        raise ValueError(f'adaptation.read_mode must be one of {READ_MODES}.')
    read_scale = float(cfg.adaptation.get('read_scale', 1.0))
    if not np.isfinite(read_scale) or read_scale <= 0.0:
        raise ValueError('adaptation.read_scale must be finite and positive.')
    if str(cfg.action.translation_loss) != 'huber':
        raise ValueError('The controlled benchmark uses normalized Huber translation loss.')
    if str(cfg.action.gripper_loss) != 'binary_cross_entropy':
        raise ValueError('The controlled benchmark uses BCE for the binary gripper.')
    if int(cfg.adaptation.write_segment_size) <= 0:
        raise ValueError('adaptation.write_segment_size must be positive.')
    if int(cfg.adaptation.write_steps_per_segment) <= 0:
        raise ValueError('adaptation.write_steps_per_segment must be positive.')


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


def _run_id(cfg: ConfigDict, wandb_mod) -> str:
    if wandb_mod is not None and wandb_mod.run is not None:
        return str(wandb_mod.run.id)
    objective = str(cfg.adaptation.write_objective)
    order = 'fo' if bool(cfg.adaptation.first_order) else 'full'
    return f'{objective}_{order}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'


def _task_split_payload(task_bank: HiddenGoalTaskBank) -> Dict[str, Any]:
    return {
        split: task_bank.goals(split).tolist()
        for split in ('train', 'validation', 'test')
    }


def _task_split_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
    return 'hidden_goal_split_' + hashlib.sha256(encoded).hexdigest()[:12]


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def _to_jax(batch: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, batch)


def _shard_batch(batch: Any, num_devices: int) -> Any:
    def shard(value):
        if not hasattr(value, 'shape') or value.ndim == 0:
            return value
        if int(value.shape[0]) % int(num_devices) != 0:
            raise ValueError(
                f'Leading batch dimension {value.shape[0]} is not divisible by {num_devices} devices.'
            )
        return value.reshape(
            (int(num_devices), int(value.shape[0]) // int(num_devices))
            + value.shape[1:]
        )

    return jax.tree_util.tree_map(shard, batch)


def _unreplicate(state: TTTTrainState) -> TTTTrainState:
    return jax.tree_util.tree_map(lambda value: value[0], state)


def _replicate_state(state: TTTTrainState, devices) -> TTTTrainState:
    devices = tuple(devices)
    mesh = Mesh(np.asarray(devices, dtype=object), ('devices',))

    def replicate(value):
        value = jnp.asarray(value)
        expanded = jnp.broadcast_to(value, (len(devices),) + value.shape)
        sharding = NamedSharding(
            mesh, PartitionSpec('devices', *([None] * value.ndim))
        )
        return jax.device_put(expanded, sharding)

    return jax.tree_util.tree_map(replicate, state)


def _host_metrics(metrics: Mapping[str, Any], distributed: bool) -> Dict[str, float]:
    host = jax.device_get(metrics)
    if distributed:
        host = jax.tree_util.tree_map(lambda value: value[0], host)
    return {name: float(np.asarray(value)) for name, value in host.items()}


def _log_metrics(
    prefix: str,
    step: int,
    metrics: Mapping[str, float],
    wandb_mod,
) -> None:
    ordered = ' | '.join(
        f'{name} {value:.6f}'
        for name, value in sorted(metrics.items())
        if name in {
            'loss',
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
    )
    logging.info('%s step %d | %s', prefix, int(step), ordered)
    if wandb_mod is not None:
        wandb_mod.log(
            {f'{prefix}/{name}': value for name, value in metrics.items()},
            step=int(step),
        )


def _restore_state(
    checkpoint_path: str,
    optimizer,
) -> tuple[TTTTrainState, StateNormalizer | None]:
    payload = load_checkpoint(checkpoint_path)
    state = create_ttt_train_state(
        jax.tree_util.tree_map(jnp.asarray, payload['params']),
        optimizer,
        jnp.asarray(payload['rng']),
        step=int(payload['step']),
        opt_state=jax.tree_util.tree_map(jnp.asarray, payload['opt_state']),
    )
    normalizer_payload = payload.get('extra', {}).get('normalizer')
    normalizer = (
        None
        if normalizer_payload is None
        else StateNormalizer.from_dict(normalizer_payload)
    )
    return state, normalizer


def _save_state(
    path: Path,
    state: TTTTrainState,
    *,
    step: int,
    cfg: ConfigDict,
    normalizer: StateNormalizer,
    provenance: Mapping[str, Any],
    distributed: bool,
) -> None:
    save_checkpoint(
        path,
        state=state,
        step=int(step),
        config=cfg,
        extra={
            'checkpoint_type': 'fast_weight_ttt_state',
            'normalizer': normalizer.to_dict(),
            'experiment_id': provenance['experiment_id'],
            'task_split_id': provenance['dataset']['task_split_id'],
            'transient_fast_state_saved': False,
        },
        replicated=bool(distributed),
    )


def train_ttt_state(cfg: ConfigDict) -> Path:
    validate_adaptation_only_config(cfg)
    benchmark_cfg = hidden_goal_config_from(cfg)
    task_bank = HiddenGoalTaskBank(benchmark_cfg)
    normalizer = fit_state_normalizer(
        benchmark_cfg,
        task_bank,
        episodes_per_task=int(cfg.benchmark.normalizer_episodes_per_task),
    )
    split_payload = _task_split_payload(task_bank)
    split_id = _task_split_id(split_payload)
    integrity = benchmark_integrity_report(
        benchmark_cfg, task_bank, normalizer, samples=512, seed=int(cfg.train.seed) + 71
    )
    if integrity['support_query_episode_overlap']:
        raise RuntimeError('Support/query episode separation check failed.')
    if integrity['support_query_identical_initial_state']:
        raise RuntimeError('Support/query initial-state separation check failed.')
    if integrity['support_outer_loss_mask_nonzero']:
        raise RuntimeError('Support outer-loss masking check failed.')
    if float(integrity['oracle_expert_success_rate']) < 0.99:
        raise RuntimeError('Expert integrity check failed.')

    model_cfg = fast_weight_config_from(cfg, benchmark_cfg)
    adapt_cfg = adaptation_config_from(cfg)
    step_cfg = step_config_from(cfg)
    params = init_fast_weight_ttt_params(jax.random.key(int(cfg.train.seed)), model_cfg)
    optimizer = _optimizer(cfg, params)
    state = create_ttt_train_state(
        params, optimizer, jax.random.key(int(cfg.train.seed) + 1)
    )
    if str(cfg.train.resume_path):
        state, checkpoint_normalizer = _restore_state(str(cfg.train.resume_path), optimizer)
        if checkpoint_normalizer is not None:
            if checkpoint_normalizer.identifier != normalizer.identifier:
                raise ValueError(
                    'Checkpoint normalizer differs from the current train-split normalizer: '
                    f'{checkpoint_normalizer.identifier} != {normalizer.identifier}.'
                )
            normalizer = checkpoint_normalizer

    distributed = bool(cfg.train.distributed)
    num_devices = jax.local_device_count() if distributed else 1
    if int(cfg.train.batch_size) % num_devices != 0:
        raise ValueError(
            f'train.batch_size={cfg.train.batch_size} must be divisible by {num_devices} devices.'
        )
    train_step = create_ttt_train_step(
        optimizer,
        model_cfg,
        adapt_cfg,
        step_cfg,
        distributed=distributed,
    )
    if distributed:
        state = _replicate_state(state, jax.local_devices())

    train_sampler = HiddenGoalMetaSampler(
        benchmark_cfg,
        task_bank,
        normalizer,
        split='train',
        seed=int(cfg.train.seed) + 1001,
    )
    validation_sampler = HiddenGoalMetaSampler(
        benchmark_cfg,
        task_bank,
        normalizer,
        split='validation',
        seed=int(cfg.train.seed) + 2001,
    )
    wandb_mod = _maybe_wandb(cfg)
    run_id = _run_id(cfg, wandb_mod)
    run_dir = Path(cfg.train.output_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset_provenance = {
        'name': 'hidden_goal_reach_and_grasp',
        'task_split_id': split_id,
        'task_split_sizes': integrity['split_sizes'],
        'normalizer_id': normalizer.identifier,
        'normalizer_fit_split': 'train',
        'support_query_separate_episodes': True,
    }
    provenance = collect_experiment_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        config=cfg,
        experiment_id=run_id,
        dataset=dataset_provenance,
        parent_checkpoint=str(cfg.train.resume_path),
        adaptation_mode=str(cfg.mode),
        reset_policy=str(cfg.adaptation.reset_policy),
    )
    write_experiment_ledger(run_dir, config=cfg, provenance=provenance)
    _write_json(run_dir / 'normalizer.json', normalizer.to_dict())
    _write_json(run_dir / 'task_splits.json', split_payload)
    _write_json(run_dir / 'benchmark_integrity.json', integrity)
    logging.info('TTT run directory: %s', run_dir)
    logging.info('Benchmark integrity: %s', integrity)

    start_step = int(jax.device_get(state.step[0] if distributed else state.step))
    target_step = int(cfg.train.num_steps)
    if start_step >= target_step:
        raise ValueError(
            f'Checkpoint step {start_step} is already >= train.num_steps {target_step}.'
        )
    last_log_time = time.time()
    last_log_step = start_step
    try:
        for step in range(start_step + 1, target_step + 1):
            batch = _to_jax(
                train_sampler.build_batch(
                    int(cfg.train.batch_size),
                    support_episodes=int(benchmark_cfg.support_episodes),
                    query_episodes=int(benchmark_cfg.query_episodes),
                )
            )
            if distributed:
                batch = _shard_batch(batch, num_devices)
            state, metrics = train_step(state, batch)
            if step == start_step + 1:
                jax.block_until_ready(metrics['loss'])
            if step == start_step + 1 or step % int(cfg.train.log_every) == 0:
                now = time.time()
                host_metrics = _host_metrics(metrics, distributed)
                host_metrics['step_s'] = (now - last_log_time) / max(
                    1, step - last_log_step
                )
                _log_metrics('train', step, host_metrics, wandb_mod)
                last_log_time = now
                last_log_step = step

            if step % int(cfg.train.eval_every) == 0:
                eval_state = _unreplicate(state) if distributed else state
                eval_metrics_accumulator = []
                for _ in range(int(cfg.train.eval_batches)):
                    eval_batch = _to_jax(
                        validation_sampler.build_batch(int(cfg.train.batch_size))
                    )
                    _, eval_metrics = ttt_meta_objective(
                        eval_state.params,
                        eval_batch,
                        model_cfg,
                        adapt_cfg,
                        step_cfg,
                    )
                    eval_metrics_accumulator.append(jax.device_get(eval_metrics))
                eval_metrics = {
                    name: float(
                        np.mean([np.asarray(item[name]) for item in eval_metrics_accumulator])
                    )
                    for name in eval_metrics_accumulator[0]
                }
                diagnostic_batch = _to_jax(validation_sampler.build_batch(1))
                one_support = jax.tree_util.tree_map(
                    lambda value: value[0], diagnostic_batch['support']
                )
                one_query = jax.tree_util.tree_map(
                    lambda value: value[0], diagnostic_batch['query']
                )
                alignment = write_query_gradient_alignment(
                    eval_state.params,
                    one_support,
                    one_query,
                    model_cfg,
                    adapt_cfg,
                )
                adapted, _ = adapt_fast_state(
                    eval_state.params, one_support, model_cfg, adapt_cfg
                )
                tensor_metrics = named_fast_tensor_metrics(
                    eval_state.params, adapted, model_cfg
                )
                tensor_metrics['fast/effective_rank'] = fast_state_effective_rank(
                    adapted
                )
                eval_metrics.update(
                    {
                        name: float(np.asarray(value))
                        for name, value in jax.device_get(
                            {**alignment, **tensor_metrics}
                        ).items()
                    }
                )
                _log_metrics('validation', step, eval_metrics, wandb_mod)

            if step % int(cfg.train.ckpt_every) == 0:
                _save_state(
                    run_dir / f'step_{step:07d}.pkl',
                    state,
                    step=step,
                    cfg=cfg,
                    normalizer=normalizer,
                    provenance=provenance,
                    distributed=distributed,
                )
        _save_state(
            run_dir / 'last.pkl',
            state,
            step=target_step,
            cfg=cfg,
            normalizer=normalizer,
            provenance=provenance,
            distributed=distributed,
        )
        logging.info('TTT training complete: %s', run_dir / 'last.pkl')
        return run_dir / 'last.pkl'
    finally:
        if wandb_mod is not None:
            wandb_mod.finish()
