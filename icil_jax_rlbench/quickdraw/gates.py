"""Bounded Q4 implementation diagnostics on declared caches and fixed tasks.

These gates test tiny-data fitting and ordinary support adaptation in the exact
fast subspace. They make no held-out QuickDraw or diversity-threshold claim.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from icil_jax_rlbench.models.fast_weight_ttt import (
    apply_fast_gradient, initial_fast_state, tree_difference_norm, tree_l2_norm,
)
from icil_jax_rlbench.quickdraw.data import SketchSampler, SketchStore, load_manifest
from icil_jax_rlbench.quickdraw.model import (
    SketchModelConfig, init_sketch_params, meta_objective, query_loss,
    teacher_forced_distribution,
)
from icil_jax_rlbench.quickdraw.pen import PenConfig
from icil_jax_rlbench.quickdraw.train import load_run, numeric_batch, resolve_config, train


def _dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def _fingerprint(tree):
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        value = np.asarray(value)
        digest.update(str(path).encode('ascii'))
        digest.update(str((value.shape, value.dtype.str)).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def ordinary_fast_adaptation(params, batch, cfg, *, steps=25, learning_rate=.1):
    """Optimize only W using supervised support loss, then score frozen query.

    Slow parameters, W0, architecture, and delta READ remain unchanged. Query
    loss is diagnostic only and never selects updates, rates, or stopping time.
    """
    if (not cfg.model_type.startswith('ttt_') or steps < 1
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Provide a TTT model, positive steps and learning_rate.')
    initial = initial_fast_state(params)
    rates = jax.tree_util.tree_map(lambda value: jnp.asarray(learning_rate, value.dtype), initial)
    before_fingerprint = _fingerprint(params)

    def task(support, query):
        support_before = query_loss(params, initial, support, cfg)[0]
        query_before = query_loss(params, initial, query, cfg)[0]
        distribution_before = teacher_forced_distribution(params, initial, query, cfg)
        def step(fast, _):
            support_loss, gradient = jax.value_and_grad(
                lambda state: query_loss(params, state, support, cfg)[0]
            )(fast)
            fast, _, update = apply_fast_gradient(fast, gradient, rates, cfg.adapt_config())
            return fast, {'support_loss_before_step': support_loss,
                          'query_loss_after_step': query_loss(params, fast, query, cfg)[0],
                          'gradient_norm': tree_l2_norm(gradient), 'update_norm': tree_l2_norm(update)}
        final, trace = jax.lax.scan(step, initial, xs=None, length=int(steps))
        distribution_after = teacher_forced_distribution(params, final, query, cfg)
        functional_delta = jax.tree_util.tree_map(lambda a, b: a - b, distribution_after, distribution_before)
        support_after = query_loss(params, final, support, cfg)[0]
        query_after = query_loss(params, final, query, cfg)[0]
        return {
            'support_loss_before': support_before, 'support_loss_after': support_after,
            'query_loss_before': query_before, 'query_loss_after': query_after,
            'support_loss_improvement': support_before - support_after,
            'query_loss_improvement': query_before - query_after,
            'fast_delta_norm': tree_difference_norm(final, initial),
            'functional_delta_norm': tree_l2_norm(functional_delta),
            **trace,
        }
    values = jax.jit(jax.vmap(task))(batch['support'], batch['query'])
    values = {name: np.asarray(value) for name, value in values.items()}
    if before_fingerprint != _fingerprint(params):
        raise AssertionError('Ordinary fast adaptation changed slow parameters.')
    if not all(np.all(np.isfinite(value)) for value in values.values()):
        raise FloatingPointError('Ordinary fast adaptation produced nonfinite diagnostics.')
    return values


def run_gates(cache_root, manifest_path, output, *, experiment='a', model_config=None,
              steps=50, seed=0, tasks=2, support_count=2, query_count=1,
              learning_rate=3e-3, adaptation_steps=25, adaptation_learning_rate=.1,
              train_no_support_reference=True):
    """Run fixed-meta-batch full-gradient fitting and a same-subspace upper bound."""
    for name, value in (('steps', steps), ('tasks', tasks), ('support_count', support_count),
                        ('query_count', query_count), ('adaptation_steps', adaptation_steps)):
        if int(value) < 1:
            raise ValueError(f'{name} must be positive.')
    if (not np.isfinite(learning_rate) or learning_rate <= 0
            or not np.isfinite(adaptation_learning_rate) or adaptation_learning_rate <= 0):
        raise ValueError('Learning rates must be finite and positive.')
    model = {'hidden_dim': 16, 'fast_dim': 8, 'fast_hidden_dim': 12,
             'mixture_components': 2, 'max_steps': 96 if experiment == 'b2' else 32,
             'segment_size': 16, 'inner_lr_init': .1, **(model_config or {})}
    if model.get('model_type', 'ttt_kvb_full') != 'ttt_kvb_full':
        raise ValueError('The fixed-meta-batch gate requires full-second-order KVB.')
    output = Path(output).resolve()
    cfg = resolve_config({
        'cache_root': str(Path(cache_root).resolve()), 'manifest_path': str(Path(manifest_path).resolve()),
        'experiment': experiment, 'model_type': 'ttt_kvb_full', 'model': model,
        'seed': int(seed), 'batch_size': int(tasks), 'support_count': int(support_count),
        'query_count': int(query_count), 'num_steps': int(steps), 'learning_rate': float(learning_rate),
        'fixed_batch': True, 'output_dir': str(output / 'models'),
        'log_every': max(1, int(steps) // 5), 'checkpoint_every': int(steps),
    })
    store = SketchStore.open(cfg['cache_root'])
    manifest = load_manifest(cfg['manifest_path'], store)
    model_cfg = SketchModelConfig(**cfg['model'])
    sampler = SketchSampler(store, manifest, experiment=experiment, split='train',
                            seed=int(seed) + 1001, max_steps=model_cfg.max_steps,
                            support_count=support_count, query_count=query_count,
                            pen_config=PenConfig(max_motion=model_cfg.motion_bound))
    raw_batch = sampler.build_batch(int(tasks))
    batch = numeric_batch(raw_batch)
    init_key, _ = jax.random.split(jax.random.PRNGKey(int(seed)))
    initial_params = init_sketch_params(init_key, model_cfg)
    initial_loss = float(meta_objective(initial_params, batch, model_cfg)[0])
    output.mkdir(parents=True, exist_ok=False)
    arrays = {f'{role}.{name}': np.asarray(value) for role in ('support', 'query')
              for name, value in batch[role].items()}
    np.savez_compressed(output / 'fixed_batch.npz', **arrays)
    _dump(output / 'fixed_batch_records.json', raw_batch['meta'])
    _dump(output / 'config.json', {
        'training': cfg, 'adaptation_steps': int(adaptation_steps),
        'adaptation_learning_rate': float(adaptation_learning_rate),
        'train_no_support_reference': bool(train_no_support_reference),
        'cache_provenance': store.provenance, 'cache_id': store.identifier,
        'manifest_id': manifest['identifier'],
    })
    run = train(cfg)
    checkpoint = run / 'last.pkl'
    payload, _, _, _ = load_run(checkpoint)
    for role in ('support', 'query'):
        for name in batch[role]:
            np.testing.assert_array_equal(batch[role][name], payload['extra']['fixed_batch'][role][name])
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])
    final_loss = float(meta_objective(params, batch, model_cfg)[0])
    feasibility = ordinary_fast_adaptation(params, batch, model_cfg,
                                           steps=adaptation_steps, learning_rate=adaptation_learning_rate)
    np.savez_compressed(output / 'ordinary_fast_adaptation.npz', **feasibility)
    no_support_reference = {'enabled': False}
    if train_no_support_reference:
        no_support_cfg = resolve_config({**cfg, 'model_type': 'no_support'})
        reference_cfg = SketchModelConfig(**no_support_cfg['model'])
        reference_initial = init_sketch_params(init_key, reference_cfg)
        reference_initial_loss = float(meta_objective(reference_initial, batch, reference_cfg)[0])
        reference_run = train(no_support_cfg)
        reference_payload, _, _, _ = load_run(reference_run / 'last.pkl')
        for role in ('support', 'query'):
            for name in batch[role]:
                np.testing.assert_array_equal(batch[role][name], reference_payload['extra']['fixed_batch'][role][name])
        reference_params = jax.tree_util.tree_map(jnp.asarray, reference_payload['params'])
        reference_final_loss = float(meta_objective(reference_params, batch, reference_cfg)[0])
        no_support_reference = {'enabled': True, 'checkpoint': str(reference_run / 'last.pkl'),
                                'initial_query_loss': reference_initial_loss,
                                'final_query_loss': reference_final_loss,
                                'independently_trained': True, 'same_fixed_query_batch': True}
    improvement = initial_loss - final_loss
    fit_passed = bool(np.isfinite(final_loss) and improvement > max(1e-4, abs(initial_loss) * 1e-3))
    read_changed = bool(np.any(feasibility['functional_delta_norm'] > 1e-8))
    report = {
        'schema_version': 1, 'experiment': experiment, 'checkpoint': str(checkpoint),
        'data_kind': store.provenance.get('kind', 'provided_cache'),
        'synthetic_fixture': store.provenance.get('kind') == 'synthetic_correctness_fixture',
        'cache_id': store.identifier, 'manifest_id': manifest['identifier'],
        'model_config': asdict(model_cfg),
        'fixed_meta_batch': {'tasks': int(tasks), 'optimizer_steps': int(steps),
                            'initial_query_loss': initial_loss, 'final_query_loss': final_loss,
                            'query_loss_improvement': improvement, 'passed': fit_passed,
                            'objective': 'Full-second-order query loss after support KVB updates; no support loss in outer objective.'},
        'ordinary_fast_adaptation': {
            'checkpoint_phase': 'After fixed-meta-batch meta-training',
            'steps': int(adaptation_steps), 'learning_rate': float(adaptation_learning_rate),
            'optimized_parameters': 'Exactly the existing fast-state leaves; slow params and W0 frozen.',
            'objective': 'Same supervised support prediction loss and delta READ as query.',
            'query_used_for_updates_or_selection': False,
            'same_fast_subspace': True, 'slow_parameters_unchanged': True,
            'fast_parameter_count': sum(int(value.size) for value in jax.tree_util.tree_leaves(initial_fast_state(params))),
            'read_influence_passed': read_changed,
            'mean_support_loss_improvement': float(feasibility['support_loss_improvement'].mean()),
            'mean_query_loss_improvement': float(feasibility['query_loss_improvement'].mean()),
            'support_fit_improved': bool(feasibility['support_loss_improvement'].mean() > 1e-8),
            'independent_query_improved': bool(feasibility['query_loss_improvement'].mean() > 1e-8),
            'per_task_query_improvement': feasibility['query_loss_improvement'].tolist(),
        },
        'no_support_reference': no_support_reference,
        'interpretation': 'Bounded implementation fixture. Fixed tasks were used for optimization; this is not held-out adaptation, QuickDraw quality, or diversity-threshold evidence.',
    }
    _dump(output / 'report.json', report)
    return output
