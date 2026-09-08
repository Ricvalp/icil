"""Full-data supervised ICIL training for autoregressive and diffusion Transformers.

Every epoch visits all training targets. There are no inner-loop updates or
task-adapted parameters. The NumPy loader supplies trajectories, never features.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
import importlib.util
from importlib.metadata import version
import json
import math
from pathlib import Path
import time
import uuid

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .full_data import FullDataset
from .supervised_models import SupervisedModelConfig, init_model, loss
from .metrics import file_sha256
from icil_jax_rlbench.train.checkpoints import load_checkpoint, save_checkpoint


CHECKPOINT_TYPE = 'quickdraw_full_supervised_v1'
# This reviewed predecessor differs only in observational plotting/logging.
# Remove this migration if numerical training or data sampling changes.
PRE_PLOT_TRAINER_SHA256 = '2bff18d7e4a07fa2a819675509903a3ff6de0727a00f320deaf313fbf22255b8'
BATCH_FIELDS = frozenset(('support_tokens', 'support_mask', 'query_tokens',
                         'query_mask', 'query_point_mask', 'example_mask'))


@struct.dataclass
class TrainState:
    step: jax.Array
    params: object
    opt_state: object
    rng: jax.Array


def default_config(architecture='autoregressive'):
    return {
        'dataset_root': 'datasets/quickdraw_full_nn_v1',
        'output_dir': f'outputs/quickdraw_icil/{architecture}_v1',
        'model': asdict(SupervisedModelConfig(architecture=architecture)),
        'seed': 0, 'support_count': 4, 'selection_mode': 'exact_top_k',
        'condition_on_support': True,
        'epochs': 20, 'batch_size': 64, 'micro_batch_size': 16,
        'learning_rate': 3e-4, 'min_learning_rate': 3e-5,
        'weight_decay': .01, 'grad_clip_norm': 1.0,
        'warmup_steps': 2000, 'schedule_steps': None,
        'log_every': 50, 'checkpoint_every': 1000,
        'validation_every_steps': 0, 'validation_examples_per_category': 0,
        'validation_seed': 1729, 'prefetch_batches': 2,
        'plot_every': 10000, 'plot_examples': 4, 'plot_seed': 2027,
        'resume_path': '', 'max_steps': None,
        'wandb_project': None, 'wandb_name': None, 'wandb_entity': None,
        'wandb_mode': 'online',
    }


def resolve_config(value):
    unknown = set(value) - set(default_config())
    if unknown:
        raise ValueError(f'Unknown supervised training fields: {sorted(unknown)}')
    cfg = {**default_config(), **dict(value)}
    cfg['model'] = asdict(SupervisedModelConfig(**cfg['model']))
    for name in ('epochs', 'batch_size', 'micro_batch_size', 'support_count',
                 'log_every', 'checkpoint_every', 'prefetch_batches'):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if cfg['batch_size'] % cfg['micro_batch_size']:
        raise ValueError('batch_size must be divisible by micro_batch_size')
    for name in ('validation_every_steps', 'validation_examples_per_category', 'warmup_steps',
                 'plot_every', 'plot_seed'):
        if not isinstance(cfg[name], int) or cfg[name] < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if not isinstance(cfg['plot_examples'], int) or not 1 <= cfg['plot_examples'] <= 8:
        raise ValueError('plot_examples must be an integer between 1 and 8')
    if cfg['plot_seed'] >= 2 ** 32:
        raise ValueError('plot_seed must fit an unsigned 32-bit integer')
    if cfg['max_steps'] is not None and cfg['max_steps'] < 1:
        raise ValueError('max_steps is an optional positive diagnostic limit')
    for name in ('learning_rate', 'min_learning_rate', 'weight_decay', 'grad_clip_norm'):
        if not np.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if cfg['learning_rate'] <= 0 or cfg['grad_clip_norm'] <= 0:
        raise ValueError('learning_rate and grad_clip_norm must be positive')
    if cfg['min_learning_rate'] > cfg['learning_rate']:
        raise ValueError('min_learning_rate cannot exceed learning_rate')
    if cfg['selection_mode'] not in ('exact_top_k', 'sample_top_m'):
        raise ValueError('Use exact_top_k or sample_top_m')
    if cfg['wandb_mode'] not in ('online', 'offline', 'disabled'):
        raise ValueError('Invalid W&B mode')
    return cfg


def numeric_batch(batch, *, condition_on_support=True):
    unexpected = set(batch) - BATCH_FIELDS - {'metadata'}
    missing = BATCH_FIELDS - set(batch)
    if unexpected or missing:
        raise ValueError(f'Invalid supervised trajectory fields: missing={sorted(missing)}, '
                         f'unexpected={sorted(unexpected)}')
    result = {name: jnp.asarray(batch[name]) for name in BATCH_FIELDS}
    if not condition_on_support:
        result['support_mask'] = jnp.zeros_like(result['support_mask'])
        result['support_tokens'] = jnp.zeros_like(result['support_tokens'])
    return result


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _optimizer(cfg, params):
    if cfg['schedule_steps'] <= cfg['warmup_steps']:
        raise ValueError('schedule_steps must exceed warmup_steps')
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0 if cfg['warmup_steps'] else cfg['learning_rate'],
        peak_value=cfg['learning_rate'], warmup_steps=cfg['warmup_steps'],
        decay_steps=cfg['schedule_steps'], end_value=cfg['min_learning_rate'])
    decay = jax.tree_util.tree_map(lambda value: value.ndim > 1, params)
    return optax.chain(optax.clip_by_global_norm(cfg['grad_clip_norm']),
                       optax.adamw(schedule, weight_decay=cfg['weight_decay'], mask=decay)), schedule


def create_train_step(optimizer, model_cfg, micro_batch_size):
    """Accumulate weighted microbatch gradients without storing all gradients."""
    def step(state, batch):
        next_rng, objective_key = jax.random.split(state.rng)
        n = batch['example_mask'].shape[0] // micro_batch_size
        # [batch,...] -> [accumulation,microbatch,...], preserving demo/time axes.
        micro = jax.tree_util.tree_map(
            lambda value: value.reshape((n, micro_batch_size) + value.shape[1:]), batch)

        def evaluate(index, inputs):
            (_, metrics), grads = jax.value_and_grad(
                lambda p: loss(p, inputs, model_cfg, jax.random.fold_in(objective_key, index), training=True),
                has_aux=True)(state.params)
            weight = jnp.sum(inputs['example_mask']).astype(jnp.float32)
            return jax.tree_util.tree_map(lambda x: x * weight, grads), {
                name: value * weight for name, value in metrics.items()}, weight

        first = jax.tree_util.tree_map(lambda x: x[0], micro)
        carry = evaluate(jnp.asarray(0), first)

        def accumulate(carry, values):
            index, inputs = values
            result = evaluate(index, inputs)
            return jax.tree_util.tree_map(jnp.add, carry, result), None

        (grads, metrics, count), _ = jax.lax.scan(
            accumulate, carry, (jnp.arange(1, n), jax.tree_util.tree_map(lambda x: x[1:], micro)))
        grads = jax.tree_util.tree_map(lambda x: x / jnp.maximum(count, 1), grads)
        metrics = {name: value / jnp.maximum(count, 1) for name, value in metrics.items()}
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return state.replace(step=state.step + 1, params=params, opt_state=opt_state, rng=next_rng), {
            **metrics, 'gradient_norm': optax.global_norm(grads),
        }
    return jax.jit(step)


def _host_batch(dataset, selected, cfg, epoch, batch_index, *, validation=False):
    valid = len(selected)
    padded = np.pad(selected, (0, cfg['batch_size'] - valid), mode='edge')
    seed = cfg['validation_seed'] if validation else cfg['seed']
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0 if validation else epoch, batch_index, 31]))
    batch = dataset.batch(padded, support_count=cfg['support_count'],
                          selection_mode=cfg['selection_mode'], rng=rng)
    batch['example_mask'] = np.arange(cfg['batch_size']) < valid
    return batch


def _batches(dataset, rows, cfg, epoch=0, start=0, *, validation=False):
    """Bounded thread prefetch; randomness is derived from immutable batch indices."""
    total = math.ceil(len(rows) / cfg['batch_size'])
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='quickdraw-input') as executor:
        pending = {}
        next_index = start
        for index in range(start, total):
            while next_index < min(total, index + cfg['prefetch_batches']):
                selected = rows[next_index * cfg['batch_size']:(next_index + 1) * cfg['batch_size']]
                pending[next_index] = executor.submit(
                    _host_batch, dataset, selected, cfg, epoch, next_index, validation=validation)
                next_index += 1
            yield index, pending.pop(index).result()


def validation_rows(dataset, cfg):
    rows = dataset.rows('development')
    cap = cfg['validation_examples_per_category']
    if not cap:
        return rows
    rng = np.random.default_rng(cfg['validation_seed'])
    return np.concatenate([np.sort(rng.permutation(rows[dataset.category_ids[rows] == category])[:cap])
                           for category in range(len(dataset.categories))])


@lru_cache(maxsize=8)
def create_validation_step(model_cfg):
    return jax.jit(lambda p, b, key: loss(p, b, model_cfg, key, training=False)[1])


def validate(params, dataset, cfg, model_cfg, rows=None):
    rows = validation_rows(dataset, cfg) if rows is None else rows
    evaluate = create_validation_step(model_cfg)
    validation_cfg = {**cfg, 'batch_size': cfg['micro_batch_size']}
    totals, count = {}, 0
    started = last_status = time.monotonic()
    print(f'Validating {len(rows):,} development targets ...', flush=True)
    for index, host in _batches(dataset, rows, validation_cfg, validation=True):
        batch = numeric_batch(host, condition_on_support=cfg['condition_on_support'])
        key = jax.random.fold_in(jax.random.PRNGKey(cfg['validation_seed']), index)
        metrics = jax.device_get(evaluate(params, batch, key))
        weight = int(np.sum(host['example_mask']))
        for name, value in metrics.items():
            if not np.isfinite(value):
                raise ValueError(f'Nonfinite validation metric: {name}')
            totals[name] = totals.get(name, 0.0) + float(value) * weight
        count += weight
        now = time.monotonic()
        if now - last_status >= 30:
            print(f'validation {count:,}/{len(rows):,} ({100 * count / len(rows):.1f}%) | '
                  f'{count / max(now - started, 1e-8):.1f} examples/s', flush=True)
            last_status = now
    if not count:
        raise ValueError('Validation must contain examples')
    return {name: value / count for name, value in totals.items()}, count


@contextmanager
def _wandb_run(cfg, output, history):
    if not cfg['wandb_project'] or cfg['wandb_mode'] == 'disabled':
        yield None
        return
    import wandb
    path = output / 'wandb_run.json'
    saved = json.loads(path.read_text()) if path.exists() else None
    if saved and saved['project'] != cfg['wandb_project']:
        raise ValueError('Resume W&B with the original project')
    resume = bool(saved and saved['mode'] == cfg['wandb_mode'] == 'online')
    run = wandb.init(project=cfg['wandb_project'],
                     entity=cfg['wandb_entity'] or (saved.get('entity') if saved else None),
                     name=cfg['wandb_name'] or output.name,
                     id=saved['id'] if resume else uuid.uuid4().hex,
                     resume='allow' if resume else 'never', mode=cfg['wandb_mode'],
                     dir=str(output), job_type='supervised-icil',
                     config={key: value for key, value in cfg.items() if key != 'resume_path'})
    exit_code = 1
    try:
        _json(path, {'id': run.id, 'entity': run.entity, 'project': cfg['wandb_project'], 'mode': cfg['wandb_mode']})
        run.define_metric('optimizer_step')
        for name in ('train/*', 'validation/*', 'epoch', 'exposure/*', 'samples/*'):
            run.define_metric(name, step_metric='optimizer_step')
        for index, row in enumerate(history):
            if index >= run.step:
                run.log(_wandb_record(row, output), step=index, commit=True)
        yield run
        exit_code = 0
    finally:
        run.finish(exit_code=exit_code)


def _wandb_record(row, output):
    """Keep checkpoints/JSON numeric or paths; materialize images only for W&B."""
    result = dict(row)
    key = 'samples/context_and_generated'
    if key in result:
        path = Path(output) / result[key]
        if path.is_file():
            import wandb
            result[key] = wandb.Image(str(path), caption=(
                f'Fixed development contexts and generated sketches | step {row["optimizer_step"]}'))
        else:
            # Scalar history remains replayable if old local plots were removed.
            result.pop(key)
    return result


def _scientific_config(cfg):
    mutable = {'output_dir', 'dataset_root', 'resume_path', 'epochs', 'max_steps',
               'log_every', 'checkpoint_every', 'prefetch_batches',
               'plot_every', 'plot_examples', 'plot_seed',
               'wandb_project', 'wandb_name', 'wandb_entity', 'wandb_mode'}
    return {key: value for key, value in cfg.items() if key not in mutable}


def _execution_signature():
    return {
        'versions': {name: version(name) for name in ('jax', 'jaxlib', 'flax', 'optax', 'numpy')},
        'backend': jax.default_backend(),
        'device_kinds': [device.device_kind for device in jax.local_devices()],
        'source_hashes': {path.name: file_sha256(path) for path in
                          (Path(__file__), Path(__file__).with_name('supervised_models.py'),
                           Path(__file__).with_name('full_data.py'))},
    }


def _compatible_execution(previous, current):
    if previous == current:
        return True
    if not isinstance(previous, dict):
        return False
    sources = previous.get('source_hashes', {})
    if sources.get('supervised_train.py') != PRE_PLOT_TRAINER_SHA256:
        return False
    upgraded = {**previous, 'source_hashes': {
        **sources, 'supervised_train.py': current['source_hashes']['supervised_train.py']}}
    return upgraded == current


def load_run(checkpoint, *, dataset_root=None):
    payload = load_checkpoint(checkpoint)
    if payload.get('extra', {}).get('checkpoint_type') != CHECKPOINT_TYPE:
        raise ValueError('Expected a full-data supervised Transformer checkpoint')
    cfg = resolve_config(payload['config'])
    dataset = FullDataset.open(dataset_root or cfg['dataset_root'])
    if dataset.identifier != payload['extra']['dataset_id']:
        raise ValueError('Checkpoint and dataset identities differ')
    return payload, cfg, SupervisedModelConfig(**cfg['model']), dataset


def train(value):
    cfg = resolve_config(value)
    dataset = FullDataset.open(cfg['dataset_root'])
    model_cfg = SupervisedModelConfig(**cfg['model'])
    if dataset.max_steps != model_cfg.max_steps or cfg['support_count'] > dataset.top_m:
        raise ValueError('Model length/support count must agree with the full dataset')
    train_rows = dataset.rows('train')
    steps_per_epoch = math.ceil(len(train_rows) / cfg['batch_size'])
    output = Path(cfg['output_dir']).resolve()
    payload = load_checkpoint(cfg['resume_path']) if cfg['resume_path'] else None
    execution = _execution_signature()
    if cfg['schedule_steps'] is None:
        cfg['schedule_steps'] = (payload['config']['schedule_steps'] if payload
                                 else max(cfg['epochs'] * steps_per_epoch, cfg['warmup_steps'] + 1))
    if payload:
        extra = payload['extra']
        if extra.get('checkpoint_type') != CHECKPOINT_TYPE or extra['dataset_id'] != dataset.identifier:
            raise ValueError('Resume requires the same supervised dataset and checkpoint type')
        if _scientific_config(cfg) != _scientific_config(payload['config']):
            raise ValueError('Resume requires identical model, data sampling, optimizer, and validation settings')
        if Path(cfg['resume_path']).resolve().parent != output:
            raise ValueError('Resume in the original output directory to retain the historical best checkpoint')
        if not _compatible_execution(extra.get('execution'), execution):
            raise ValueError('Exact resume requires the original source, package versions, backend, and device kinds')
        selection_path = output / 'plots' / 'selection.json'
        if cfg['plot_every'] and selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if (selection['seed'] != cfg['plot_seed'] or
                    len(selection['target_rows']) != min(cfg['plot_examples'], len(dataset.rows('development')))):
                raise ValueError('Keep plot_examples and plot_seed fixed after the first plot')
        params = jax.device_put(payload['params'])
    else:
        if output.exists():
            raise FileExistsError(f'Use a new output directory or --resume: {output}')
        params = init_model(jax.random.PRNGKey(cfg['seed']), model_cfg, support_count=cfg['support_count'])
    optimizer, schedule = _optimizer(cfg, params)
    state = TrainState(
        step=jnp.asarray(payload['step'] if payload else 0, jnp.int32), params=params,
        opt_state=jax.device_put(payload['opt_state']) if payload else optimizer.init(params),
        rng=jnp.asarray(payload['rng']) if payload else jax.random.PRNGKey(cfg['seed'] + 1))
    n = len(dataset.lengths)
    if payload:
        epoch, offset = extra['next_epoch'], extra['next_batch']
        history, best_loss = extra['history'], extra['best_validation_loss']
        exposure = extra['exposure']
        epoch_totals, epoch_count = extra['epoch_totals'], extra['epoch_count']
    else:
        epoch, offset, history, best_loss = 0, 0, [], None
        exposure = {'targets': np.zeros(n, np.uint32), 'supports': np.zeros(n, np.uint32),
                    'pairs': np.zeros((n, dataset.top_m), np.uint32),
                    'query_events': 0, 'support_events': 0}
        epoch_totals, epoch_count = {}, 0
    if epoch >= cfg['epochs'] or (cfg['max_steps'] and int(state.step) >= cfg['max_steps']):
        raise ValueError('The requested training budget is already complete')
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        'checkpoint_type': CHECKPOINT_TYPE, 'dataset_id': dataset.identifier,
        'dataset_manifest': str(dataset.root / 'manifest.json'),
        'dataset_manifest_sha256': file_sha256(dataset.root / 'manifest.json'),
        'training': 'ordinary_supervised_query_objective_no_inner_loop_or_fast_weights',
        'architecture': model_cfg.architecture, 'model': asdict(model_cfg),
        'conditioning': 'support_trajectories_only' if cfg['condition_on_support'] else 'independently_trained_no_context',
        'parameter_count': sum(int(x.size) for x in jax.tree_util.tree_leaves(params)),
        'execution': execution,
        'devices': [str(device) for device in jax.local_devices()],
        'training_target_count': len(train_rows), 'category_count': len(dataset.categories),
        'validation_example_count': len(validation_rows(dataset, cfg)),
        'test_used_for_selection': False, 'config': cfg,
    }
    _json(output / 'config.json', cfg)
    if not payload:
        _json(output / 'provenance.json', provenance)
    elif extra['execution'] != execution:
        _json(output / 'plotting_upgrade.json', {
            'change': 'periodic_fixed_development_plots_with_independent_rng',
            'checkpoint_step': int(state.step), 'previous_execution': extra['execution'],
            'current_execution': execution,
        })
    metrics_path = output / 'metrics.jsonl'
    metrics_path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in history))
    step_fn = create_train_step(optimizer, model_cfg, cfg['micro_batch_size'])
    step = int(state.step)
    window = []
    started, window_started = time.monotonic(), time.monotonic()
    previous_validation_step = -1
    plot_batch = None
    print(f'{model_cfg.architecture}: {len(train_rows):,} training targets across '
          f'{len(dataset.categories)} categories; {steps_per_epoch:,} updates/epoch; '
          f'{provenance["parameter_count"]:,} parameters', flush=True)

    with _wandb_run(cfg, output, history) as wandb_run:
        def record(row):
            history.append(row)
            with metrics_path.open('a') as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            index = len(history) - 1
            if wandb_run is not None and index >= wandb_run.step:
                wandb_run.log(_wandb_record(row, output), step=index, commit=True)

        def flush_window():
            nonlocal window_started, epoch_count
            if not window:
                return
            host = jax.device_get([item[0] for item in window])
            count = sum(item[1] for item in window)
            totals = {}
            for values, (_, weight) in zip(host, window):
                for name, value in values.items():
                    if not np.isfinite(value):
                        raise ValueError(f'Nonfinite training metric at step {step}: {name}')
                    totals[name] = totals.get(name, 0.0) + float(value) * weight
            for name, value in totals.items():
                epoch_totals[name] = epoch_totals.get(name, 0.0) + value
            epoch_count += count
            elapsed = max(time.monotonic() - window_started, 1e-8)
            row = {'optimizer_step': step, 'epoch': epoch + offset / steps_per_epoch,
                   **{'train/' + name: value / count for name, value in totals.items()},
                   'train/examples_per_second': count / elapsed,
                   'train/learning_rate': float(schedule(max(step - 1, 0))),
                   'exposure/query_events': int(exposure['query_events']),
                   'exposure/support_events': int(exposure['support_events'])}
            record(row)
            print(f"step {step:,} | epoch {row['epoch']:.3f} | train loss={row['train/loss']:.5f} | "
                  f"{row['train/examples_per_second']:.1f} examples/s", flush=True)
            window.clear()
            window_started = time.monotonic()

        def checkpoint(name):
            save_checkpoint(output / name, state=state, step=step, config=cfg, replicated=False,
                extra={'checkpoint_type': CHECKPOINT_TYPE, 'dataset_id': dataset.identifier,
                       'execution': execution,
                       'next_epoch': epoch, 'next_batch': offset, 'history': history,
                       'best_validation_loss': best_loss, 'exposure': exposure,
                       'epoch_totals': epoch_totals, 'epoch_count': epoch_count})

        def evaluate():
            nonlocal best_loss, previous_validation_step, window_started
            if previous_validation_step == step:
                return
            values, count = validate(state.params, dataset, cfg, model_cfg)
            record({'optimizer_step': step, 'epoch': epoch + offset / steps_per_epoch,
                    **{'validation/' + name: value for name, value in values.items()},
                    'validation/sample_count': count})
            improved = best_loss is None or values['loss'] < best_loss
            if improved:
                best_loss = values['loss']
                checkpoint('best.pkl')
            previous_validation_step = step
            print(f"step {step:,} | validation loss={values['loss']:.5f} | "
                  f"{count:,} examples | best={best_loss:.5f}", flush=True)
            window_started = time.monotonic()

        def plot():
            nonlocal plot_batch, window_started
            from .supervised_plots import prepare_plot_batch, generate_plot
            if plot_batch is None:
                plot_batch = prepare_plot_batch(
                    dataset, count=cfg['plot_examples'], support_count=cfg['support_count'],
                    selection_mode=cfg['selection_mode'], seed=cfg['plot_seed'],
                    condition_on_support=cfg['condition_on_support'])
                selection_path = output / 'plots' / 'selection.json'
                selection_path.parent.mkdir(parents=True, exist_ok=True)
                if selection_path.exists():
                    if json.loads(selection_path.read_text()) != plot_batch['metadata']:
                        raise ValueError('Keep plot_examples and plot_seed fixed after the first plot')
                else:
                    _json(selection_path, plot_batch['metadata'])
            relative = f'plots/step_{step:09d}.png'
            print(f'step {step:,} | generating fixed development sketches ...', flush=True)
            generate_plot(state.params, model_cfg, plot_batch, output / relative, step=step)
            record({'optimizer_step': step, 'epoch': epoch + offset / steps_per_epoch,
                    'samples/context_and_generated': relative,
                    'samples/example_count': len(plot_batch['support_tokens'])})
            print(f'step {step:,} | saved {output / relative}', flush=True)
            window_started = time.monotonic()

        while epoch < cfg['epochs'] and (cfg['max_steps'] is None or step < cfg['max_steps']):
            order = np.random.default_rng(np.random.SeedSequence([cfg['seed'], epoch, 13])).permutation(train_rows)
            for batch_index, host in _batches(dataset, order, cfg, epoch, offset):
                selected = np.asarray(host['metadata']['target_rows'])[host['example_mask']]
                support_rows = np.asarray(host['metadata']['support_rows'])[host['example_mask']]
                batch = numeric_batch(host, condition_on_support=cfg['condition_on_support'])
                state, metrics = step_fn(state, batch)
                step += 1
                offset = batch_index + 1
                window.append((metrics, len(selected)))
                np.add.at(exposure['targets'], selected, 1)
                exposure['query_events'] += int(np.sum(host['query_mask'][host['example_mask']]))
                if cfg['condition_on_support']:
                    np.add.at(exposure['supports'], support_rows.reshape(-1), 1)
                    slots = np.argmax(dataset.neighbors[selected, None, :] == support_rows[:, :, None], axis=-1)
                    np.add.at(exposure['pairs'], (np.repeat(selected, cfg['support_count']), slots.reshape(-1)), 1)
                    exposure['support_events'] += int(np.sum(host['support_mask'][host['example_mask']]))
                end_epoch = offset == steps_per_epoch
                stop = cfg['max_steps'] is not None and step >= cfg['max_steps']
                validation_due = cfg['validation_every_steps'] and step % cfg['validation_every_steps'] == 0
                save_due = step % cfg['checkpoint_every'] == 0
                plot_due = cfg['plot_every'] and step % cfg['plot_every'] == 0
                if step % cfg['log_every'] == 0 or end_epoch or stop or validation_due or save_due or plot_due:
                    flush_window()
                if end_epoch:
                    record({'optimizer_step': step, 'epoch': epoch + 1,
                            **{'train/epoch_' + name: value / epoch_count for name, value in epoch_totals.items()},
                            'train/epoch_sample_count': epoch_count})
                    epoch += 1
                    offset, epoch_totals, epoch_count = 0, {}, 0
                if plot_due:
                    plot()
                if end_epoch or validation_due:
                    evaluate()
                if end_epoch or save_due or stop:
                    checkpoint('last.pkl')
                if end_epoch or stop:
                    break
        flush_window()
        checkpoint('last.pkl')
    _json(output / 'exposure.json', {
        'dataset_id': dataset.identifier, 'optimizer_steps': step,
        'completed_epochs': epoch, 'next_batch': offset,
        'target_uses': int(np.sum(exposure['targets'], dtype=np.uint64)),
        'support_uses': int(np.sum(exposure['supports'], dtype=np.uint64)),
        'unique_targets': int(np.count_nonzero(exposure['targets'])),
        'unique_supports': int(np.count_nonzero(exposure['supports'])),
        'unique_support_target_pairs': int(np.count_nonzero(exposure['pairs'])),
        'query_events': exposure['query_events'], 'support_events': exposure['support_events'],
        'available_training_targets': len(train_rows), 'runtime_seconds': time.monotonic() - started,
        'count_arrays': 'Stored in checkpoint extra.exposure; pairs columns follow the immutable neighbor table',
    })
    return output / 'last.pkl'


def _configuration(path):
    path = Path(path)
    if path.suffix == '.json':
        return json.loads(path.read_text())
    spec = importlib.util.spec_from_file_location('_quickdraw_supervised_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config()
    return config.to_dict() if hasattr(config, 'to_dict') else config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='FIELD=JSON')
    parser.add_argument('--resume', action='store_true', help='Resume output_dir/last.pkl')
    args = parser.parse_args()
    cfg = _configuration(args.config)
    for value in args.set:
        name, encoded = value.split('=', 1)
        destination = cfg
        fields = name.split('.')
        for field in fields[:-1]:
            destination = destination.setdefault(field, {})
        destination[fields[-1]] = json.loads(encoded)
    if args.resume:
        cfg['resume_path'] = str(Path(cfg['output_dir']) / 'last.pkl')
    print(train(cfg))


if __name__ == '__main__':
    main()
