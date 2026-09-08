"""Small JAX training runner with manifest-bound, exact resumable sampling."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from icil_jax_rlbench.quickdraw.data import SketchSampler, SketchStore, load_manifest
from icil_jax_rlbench.quickdraw.model import (
    SketchModelConfig, init_sketch_params, meta_objective,
)
from icil_jax_rlbench.quickdraw.pen import PenConfig
from icil_jax_rlbench.train.checkpoints import load_checkpoint, save_checkpoint
from icil_jax_rlbench.train.provenance import (
    collect_experiment_provenance, write_experiment_ledger,
)
from icil_jax_rlbench.train.ttt_step import create_ttt_train_state


MODEL_TYPES = (
    'ttt_kvb_full', 'explicit_context', 'no_support',
    'ttt_supervised_write', 'ttt_kvb_first_order',
)
TRAJECTORY_FIELDS = frozenset(('tokens', 'event_mask', 'point_mask', 'knot_mask', 'frame', 'state'))


def default_config() -> dict[str, Any]:
    return {
        'cache_root': '', 'manifest_path': '', 'experiment': 'a',
        'model_type': 'ttt_kvb_full', 'model': {}, 'seed': 0,
        'batch_size': 2, 'support_count': 4, 'query_count': 1,
        'num_steps': 100, 'learning_rate': 3e-4, 'weight_decay': 1e-5,
        'slow_grad_clip_norm': 1.0, 'log_every': 10, 'checkpoint_every': 100,
        'output_dir': 'outputs/quickdraw', 'resume_path': '',
        'b_partition': 'familiar', 'fixed_batch': False,
        'a_protocol': 'from_manifest',
    }


def resolve_config(value: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(value) - set(default_config())
    if unknown:
        raise ValueError(f'Unknown training fields: {sorted(unknown)}')
    cfg = {**default_config(), **copy.deepcopy(dict(value))}
    if cfg['experiment'] not in ('a', 'b1', 'b2'):
        raise ValueError('Only experiments a, b1 and b2 are enabled.')
    if cfg['model_type'] not in MODEL_TYPES:
        raise ValueError(f'Unknown model_type: {cfg["model_type"]}')
    if cfg['a_protocol'] not in ('from_manifest', 'a_nn', 'a_local', 'a_category'):
        raise ValueError('A protocol must be a_nn, a_local, a_category, or from_manifest.')
    for key in ('batch_size', 'support_count', 'query_count', 'num_steps',
                'log_every', 'checkpoint_every'):
        if int(cfg[key]) < 1:
            raise ValueError(f'{key} must be positive.')
    for key in ('learning_rate', 'slow_grad_clip_norm'):
        if not np.isfinite(cfg[key]) or float(cfg[key]) <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    model = {**cfg['model'], 'experiment': cfg['experiment'],
             'model_type': cfg['model_type']}
    cfg['model'] = asdict(SketchModelConfig(**model))
    return cfg


def numeric_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Metadata never crosses the compiled policy boundary."""
    for role in ('support', 'query'):
        unexpected = set(batch[role]) - TRAJECTORY_FIELDS
        if unexpected:
            raise ValueError(f'Forbidden non-trajectory {role} fields: {sorted(unexpected)}')
    return {
        role: {name: jnp.asarray(value) for name, value in batch[role].items()}
        for role in ('support', 'query')
    }


def create_train_step(optimizer, model_cfg: SketchModelConfig):
    def step(state, batch):
        next_key, objective_key = jax.random.split(state.rng)
        (loss, metrics), grads = jax.value_and_grad(
            lambda params: meta_objective(
                params, batch, model_cfg, model_cfg.model_type, objective_key
            ), has_aux=True,
        )(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return state.replace(step=state.step + 1, params=params,
                             opt_state=opt_state, rng=next_key), {
            **metrics, 'loss': loss, 'slow_grad_norm': optax.global_norm(grads),
            'slow_update_norm': optax.global_norm(updates),
        }
    # Single-device initially: no unverified distributed clipping convention.
    return jax.jit(step)


def _optimizer(cfg, params):
    def decay(path, value):
        group = str(getattr(path[0], 'key', '')) if path else ''
        return value.ndim >= 2 and group not in ('fast_init', 'inner_lr_raw')
    return optax.chain(
        optax.clip_by_global_norm(float(cfg['slow_grad_clip_norm'])),
        optax.adamw(float(cfg['learning_rate']),
                    weight_decay=float(cfg['weight_decay']),
                    mask=jax.tree_util.tree_map_with_path(decay, params)),
    )


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
                    + '\n', encoding='utf-8')


def _checkpoint_type(cfg) -> str:
    return f'quickdraw_{cfg["experiment"]}_{cfg["model_type"]}'


def load_run(checkpoint_path, *, cache_root=None, manifest_path=None):
    payload = load_checkpoint(checkpoint_path)
    cfg = resolve_config(payload['config'])
    if payload['extra'].get('checkpoint_type') != _checkpoint_type(cfg):
        raise ValueError('Checkpoint does not match its sketch model type.')
    store = SketchStore.open(cache_root or cfg['cache_root'])
    manifest = load_manifest(manifest_path or cfg['manifest_path'])
    if store.identifier != payload['extra']['cache_id']:
        raise ValueError('Checkpoint and sketch cache differ.')
    if manifest['identifier'] != payload['extra']['manifest_id']:
        raise ValueError('Checkpoint and experimental manifest differ.')
    if asdict(SketchModelConfig(**cfg['model'])) != payload['extra']['model_config']:
        raise ValueError('Checkpoint model schema differs from configuration.')
    return payload, cfg, store, manifest


def train(config: Mapping[str, Any]) -> Path:
    requested = resolve_config(config)
    resumed = None
    if requested['resume_path']:
        resumed, cfg, store, manifest = load_run(
            requested['resume_path'], cache_root=requested['cache_root'] or None,
            manifest_path=requested['manifest_path'] or None,
        )
        # Scientific settings and all stochastic streams come from the snapshot.
        for key in ('num_steps', 'log_every', 'checkpoint_every', 'output_dir',
                    'resume_path'):
            cfg[key] = requested[key]
        cfg['cache_root'] = str(Path(requested['cache_root'] or cfg['cache_root']).resolve())
        cfg['manifest_path'] = str(Path(requested['manifest_path'] or cfg['manifest_path']).resolve())
    else:
        cfg = requested
        store = SketchStore.open(cfg['cache_root'])
        manifest = load_manifest(cfg['manifest_path'])
        cfg['cache_root'] = str(Path(cfg['cache_root']).resolve())
        cfg['manifest_path'] = str(Path(cfg['manifest_path']).resolve())
    model_cfg = SketchModelConfig(**cfg['model'])
    if cfg['experiment'] == 'a':
        protocol = manifest.get('a_protocol', 'a_category')
        if cfg['a_protocol'] not in ('from_manifest', protocol):
            raise ValueError('Training A protocol disagrees with the immutable task manifest.')
        cfg['a_protocol'] = protocol
    sampler = SketchSampler(
        store, manifest, experiment=cfg['experiment'], split='train',
        seed=int(cfg['seed']) + 1001, max_steps=model_cfg.max_steps,
        support_count=int(cfg['support_count']), query_count=int(cfg['query_count']),
        b_partition=cfg['b_partition'],
        pen_config=PenConfig(max_motion=model_cfg.motion_bound),
    )
    init_key, training_key = jax.random.split(jax.random.PRNGKey(int(cfg['seed'])))
    params = init_sketch_params(init_key, model_cfg) if resumed is None else (
        jax.tree_util.tree_map(jnp.asarray, resumed['params'])
    )
    optimizer = _optimizer(cfg, params)
    state = create_ttt_train_state(
        params, optimizer, training_key if resumed is None else jnp.asarray(resumed['rng']),
        step=0 if resumed is None else int(resumed['step']),
        opt_state=None if resumed is None else jax.tree_util.tree_map(jnp.asarray, resumed['opt_state']),
    )
    if resumed is not None:
        sampler.load_state_dict(resumed['extra']['sampler_state'])
    if int(state.step) >= int(cfg['num_steps']):
        raise ValueError('num_steps is a final target and must exceed the checkpoint step.')
    counts = copy.deepcopy(resumed['extra']['exposure']) if resumed else {
        'support_events': 0, 'query_events': 0, 'write_segments': 0,
        'task_episodes': 0, 'optimizer_steps': 0,
        'consumed_support_events': 0,
    }
    observed = {key:set(value) for key,value in
                (resumed['extra'].get('observed_ids',{}) if resumed else {}).items()}
    for name in ('categories','programs','neighborhoods','support_drawings','query_drawings'):
        observed.setdefault(name,set())
    reuse = copy.deepcopy(resumed['extra'].get('reuse_counts', {})) if resumed else {}
    for name in ('tasks', 'categories', 'support_drawings', 'query_drawings', 'support_target_pairs'):
        reuse.setdefault(name, {})
    accounting_start = (resumed['extra'].get('reuse_accounting_start_step',
                        int(resumed['step']) if 'reuse_counts' not in resumed['extra'] else 0)
                        if resumed else 0)
    fixed = resumed['extra'].get('fixed_batch') if resumed else None
    if cfg['fixed_batch'] and fixed is None:
        fixed = sampler.build_batch(int(cfg['batch_size']))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    run_dir = Path(cfg['output_dir']).resolve() / f'{cfg["experiment"]}_{cfg["model_type"]}_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=False)
    provenance = collect_experiment_provenance(
        repo_root=Path(__file__).resolve().parents[2], config=cfg,
        experiment_id=run_dir.name,
        dataset={'cache_id': store.identifier, 'manifest_id': manifest['identifier']},
        parent_checkpoint=cfg['resume_path'], adaptation_mode=cfg['model_type'],
        reset_policy='W0_per_task_then_freeze_across_queries',
    )
    provenance['output_family'] = 'autoregressive_diagonal_gaussian_mixture_events' if cfg['experiment'] == 'a' else 'deterministic_control_events'
    provenance['task_protocol'] = cfg['a_protocol'] if cfg['experiment'] == 'a' else cfg['experiment']
    provenance['task_curation'] = manifest.get('curation', {})
    provenance['diversity_budget'] = manifest['budget']
    provenance['reuse_accounting_start_step'] = accounting_start
    provenance['rng_contract'] = {
        'initialization': 'JAX seed split', 'training': 'saved JAX key; AR NLL is deterministic',
        'sampling_and_transforms': 'saved independent sampler streams',
        'generation': 'explicit per-episode key paired across support conditions',
    }
    provenance['parameters'] = sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
    provenance['parameters_by_group'] = {
        name:sum(int(x.size) for x in jax.tree_util.tree_leaves(group))
        for name,group in params.items()
    }
    package_root = Path(__file__).resolve().parents[1]
    source_paths = [*Path(__file__).parent.glob('*.py'),
                    package_root/'models/fast_weight_ttt.py',
                    package_root/'train/ttt_step.py',package_root/'train/checkpoints.py']
    provenance['source_sha256'] = {
        str(path.relative_to(package_root)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_paths)
    }
    provenance['sampler'] = {
        'identifier': sampler.identifier, 'pool_identifier': sampler.pool_identifier,
        'manifest_reservoir_counts': sampler.manifest_reservoir_counts,
        'eligible_reservoir_counts': sampler.reservoir_counts,
    }
    write_experiment_ledger(run_dir, config=cfg, provenance=provenance)
    _json(run_dir / 'manifest.json', manifest)
    train_step = create_train_step(optimizer, model_cfg)

    def save(step):
        save_checkpoint(run_dir / 'last.pkl', state=state, step=step, config=cfg,
                        replicated=False, extra={
            'checkpoint_type': _checkpoint_type(cfg), 'schema_version': 1,
            'cache_id': store.identifier, 'manifest_id': manifest['identifier'],
            'model_config': asdict(model_cfg), 'sampler_state': sampler.state_dict(),
            'exposure': counts, 'fixed_batch': fixed, 'transient_fast_state_saved': False,
            'observed_ids': {key:sorted(value) for key,value in observed.items()},
            'reuse_counts': reuse, 'reuse_accounting_start_step': accounting_start,
            'experiment_id': run_dir.name,
        })

    with (run_dir / 'metrics.jsonl').open('w', encoding='utf-8') as log:
        start = time.monotonic()
        for step in range(int(state.step) + 1, int(cfg['num_steps']) + 1):
            batch = fixed if fixed is not None else sampler.build_batch(int(cfg['batch_size']))
            state, metrics = train_step(state, numeric_batch(batch))
            loss = float(metrics['loss'])
            if not np.isfinite(loss):
                raise FloatingPointError(f'Nonfinite query loss at step {step}.')
            counts['support_events'] += int(np.count_nonzero(batch['support']['event_mask']))
            counts['query_events'] += int(np.count_nonzero(batch['query']['event_mask']))
            counts['task_episodes'] += int(cfg['batch_size'])
            counts['optimizer_steps'] += 1
            for record in batch['meta']['tasks']:
                observed['categories'].add(record['intended_category'])
                observed['support_drawings'].update(record['support_ids'])
                observed['query_drawings'].update(record['query_ids'])
                if record.get('intended_base_id') is not None:
                    observed['programs'].add(record['intended_base_id'])
                if record.get('intended_neighborhood_id') is not None:
                    observed['neighborhoods'].add(record['intended_neighborhood_id'])
                used = {'tasks': [record['task_id']], 'categories': [record['intended_category']],
                        'support_drawings': record['support_ids'], 'query_drawings': record['query_ids'],
                        'support_target_pairs': [json.dumps([support, query], separators=(',', ':'))
                                                for support in record['support_ids']
                                                for query in record['query_ids']]}
                for role, ids in used.items():
                    for item in ids:
                        reuse[role][item] = reuse[role].get(item, 0) + 1
            counts.update({f'observed_unique_{key}':len(value) for key,value in observed.items()})
            counts['observed_unique_drawings'] = len(observed['support_drawings']|observed['query_drawings'])
            counts['sampled_unique_drawings'] = counts['observed_unique_drawings']
            counts['observed_unique_support_target_pairs'] = len(reuse['support_target_pairs'])
            counts['consumed_unique_support_target_pairs'] = (
                len(reuse['support_target_pairs']) if cfg['model_type'] != 'no_support' else 0)
            consumed_support = observed['support_drawings'] if cfg['model_type'] != 'no_support' else set()
            counts['consumed_unique_support_drawings'] = len(consumed_support)
            counts['consumed_unique_drawings'] = len(consumed_support|observed['query_drawings'])
            counts['scored_query_points'] = counts.get('scored_query_points',0)+int(np.count_nonzero(batch['query']['point_mask']))
            if cfg['model_type'] != 'no_support':
                counts['consumed_support_events'] = counts.get('consumed_support_events',0)+int(np.count_nonzero(batch['support']['event_mask']))
            mask = np.asarray(batch['support']['event_mask'])
            padding = (-mask.shape[-1]) % model_cfg.segment_size
            segmented = np.pad(mask, [(0, 0), (0, 0), (0, padding)]).reshape(
                mask.shape[:2] + (-1, model_cfg.segment_size)
            )
            if cfg['model_type'].startswith('ttt_'):
                counts['write_segments'] += int(np.any(segmented, axis=-1).sum()) * model_cfg.write_steps_per_segment
            if step == 1 or step % int(cfg['log_every']) == 0 or step == int(cfg['num_steps']):
                row = {key: float(value) for key, value in metrics.items() if np.asarray(value).ndim == 0}
                row.update(step=step, elapsed_seconds=time.monotonic() - start, **counts)
                log.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                log.flush()
                print(json.dumps({'step': step, 'loss': loss, 'run': run_dir.name}), flush=True)
            if step % int(cfg['checkpoint_every']) == 0 or step == int(cfg['num_steps']):
                save(step)
        _json(run_dir/'runtime.json',{
            'wall_seconds_including_first_compile':time.monotonic()-start,
            'optimizer_steps_this_process':int(state.step)-(int(resumed['step']) if resumed else 0),
            'exposure':counts,'synchronization':'Loss read blocks each JAX update.',
        })
        _json(run_dir/'exposure.json', {
            'a_protocol': cfg['a_protocol'] if cfg['experiment'] == 'a' else None,
            'split_regime': manifest.get('split_regime'),
            'available_budget': manifest['budget'], 'exposure': counts,
            'reuse_counts': reuse, 'reuse_accounting_start_step': accounting_start,
            'interpretation': ('A-NN task IDs are permissible query targets; varying their count also '
                               'changes the target reservoir. Overlapping neighborhoods are not independent tasks.'),
        })
    return run_dir
