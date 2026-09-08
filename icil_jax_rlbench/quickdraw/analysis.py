"""Task-level diversity diagnostics; no simulator or evaluator imports."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def participation_ratio(updates: np.ndarray) -> dict[str, float | int]:
    """Centered across-task covariance rank via a task Gram matrix.

    This differs from matrix rank inside one fast model. All-zero or identical
    updates have dimension zero. The sample limit is min(tasks - 1, parameters).
    """
    values = np.asarray(updates, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError('Updates must be finite [task, coordinate] vectors.')
    shifted = values - values[:1] if len(values) else values
    centered = shifted - shifted.mean(axis=0, keepdims=True) if len(values) else values
    gram = centered @ centered.T
    trace = float(np.trace(gram))
    squared_trace = float(np.sum(gram * gram))
    effective = trace * trace / squared_trace if squared_trace > 0 else 0.0
    return {'effective_dimension': effective, 'tasks': len(values),
            'coordinates': values.shape[1],
            'maximum_identifiable_dimension': min(max(0, len(values) - 1), values.shape[1]),
            'centered_total_energy': trace}


def actual_update_alignment(query_gradient, actual_update) -> dict[str, float]:
    gradient = np.asarray(query_gradient, dtype=np.float64).reshape(-1)
    delta = np.asarray(actual_update, dtype=np.float64).reshape(-1)
    if gradient.shape != delta.shape or not np.all(np.isfinite((gradient, delta))):
        raise ValueError('Gradient and actual update must be finite aligned vectors.')
    improvement = -float(np.dot(gradient, delta))
    norm_product = float(np.linalg.norm(gradient) * np.linalg.norm(delta))
    return {
        'predicted_local_query_improvement': improvement,
        'descent_alignment_cosine': improvement / norm_product if norm_product else 0.0,
    }


def paired_bootstrap(values: Sequence[float], *, seed=0, draws=2000):
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.all(np.isfinite(data)):
        raise ValueError('Provide one finite paired value per independent task.')
    rng = np.random.default_rng(seed)
    estimates = data[rng.integers(0, len(data), (draws, len(data)))].mean(axis=1)
    low, high = np.quantile(estimates, [.025, .975])
    return {'mean': float(data.mean()), 'ci95': [float(low), float(high)],
            'independent_tasks': len(data), 'bootstrap_seed': int(seed)}


def compare_diversity_curves(levels, errors) -> dict:
    """Compare smooth log-linear and continuous segmented fits with BIC.

    This exploratory fit is not a threshold declaration. Repeated seed/subset
    curves must be resampled as units by the caller before claiming a location.
    """
    levels = np.asarray(levels, np.float64)
    errors = np.asarray(errors, np.float64)
    if levels.ndim != 1 or errors.shape != levels.shape or len(levels) < 5:
        raise ValueError('At least five aligned diversity levels are required.')
    if np.any(levels <= 0) or len(np.unique(levels)) != len(levels):
        raise ValueError('Diversity levels must be distinct and positive.')
    if not np.all(np.isfinite((levels, errors))):
        raise ValueError('Curve inputs must be finite.')
    order = np.argsort(levels)
    x, y = np.log(levels[order]), errors[order]

    def fit(design, parameters):
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = float(np.sum(np.square(design @ coef - y)))
        bic = len(y) * np.log(max(residual / len(y), 1e-14)) + parameters * np.log(len(y))
        return {'bic': float(bic), 'squared_error': residual, 'coefficients': coef.tolist()}

    smooth = fit(np.column_stack((np.ones_like(x), x)), 2)
    candidates = []
    for knot in x[1:-1]:
        result = fit(np.column_stack((np.ones_like(x), x, np.maximum(x-knot, 0))), 4)
        result['candidate_diversity'] = float(np.exp(knot))
        candidates.append(result)
    segmented = min(candidates, key=lambda entry: entry['bic'])
    return {'smooth': smooth, 'segmented': segmented,
            'segmented_bic_advantage': smooth['bic'] - segmented['bic'],
            'threshold_claim': False,
            'limitation': 'Exploratory fit; independent model/subset replication and location uncertainty required.'}


def summarize_update_archive(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if 'final_fast_delta' not in archive:
            raise ValueError('Archive must contain final_fast_delta[task,coordinate].')
        result = {'across_task_fast_delta': participation_ratio(archive['final_fast_delta'])}
        if 'functional_delta' in archive:
            result['across_task_functional_delta'] = participation_ratio(archive['functional_delta'])
        if 'oracle_query_gradient' in archive and 'actual_first_update' in archive:
            result['first_update_alignment'] = [
                actual_update_alignment(g, d)
                for g, d in zip(archive['oracle_query_gradient'], archive['actual_first_update'])
            ]
    return result


def neighborhood_response_diversity(features, records):
    """Average repeated task episodes, then separate local and category axes."""
    values = np.asarray(features, np.float64)
    if values.ndim != 2 or len(values) != len(records) or not np.isfinite(values).all():
        raise ValueError('Response vectors must align with finite task records.')
    if any(not row.get('intended_neighborhood_id') for row in records):
        return {'status': 'not_applicable', 'reason': 'Coarse-category or motor-program protocol.'}
    tasks, categories = {}, {}
    for index, row in enumerate(records):
        task, category = str(row['intended_neighborhood_id']), str(row['intended_category'])
        if task in categories and categories[task] != category:
            raise ValueError('Neighborhood category labels conflict.')
        categories[task] = category
        tasks.setdefault(task, []).append(index)
    means = {task: values[indices].mean(axis=0) for task, indices in tasks.items()}
    category_means, local = [], {}
    for category in sorted(set(categories.values())):
        local_means = np.stack([means[task] for task in sorted(means) if categories[task] == category])
        category_means.append(local_means.mean(axis=0))
        local[category] = participation_ratio(local_means)
    return {
        'status': 'descriptive', 'neighborhood_count': len(tasks), 'category_count': len(local),
        'across_neighborhoods': participation_ratio(np.stack(list(means.values()))),
        'within_category_neighborhoods': local,
        'across_category_means': participation_ratio(np.stack(category_means)),
        'episode_reduction': 'equal mean per intended neighborhood before covariance summaries',
        'interpretation': 'functional response differences within categories; category decodability is not a failure',
    }


def grouped_category_probe(features, labels, groups, *, folds=3, ridge=1.0, seed=0):
    """Fixed-ridge category decoding with group-held-out, train-only scaling.

    Groups must join all reused base programs/duplicate clusters. Hyperparameters
    are fixed before fitting; this is an analysis probe, never policy selection.
    """
    values = np.asarray(features, np.float64)
    labels, groups = np.asarray(labels).astype(str), np.asarray(groups).astype(str)
    if (values.ndim != 2 or labels.ndim != 1 or groups.ndim != 1
            or len(values) != len(labels) or labels.shape != groups.shape
            or not np.all(np.isfinite(values)) or not np.isfinite(ridge) or ridge <= 0 or folds < 2):
        raise ValueError('Provide finite aligned features, labels, groups and positive ridge.')
    classes = np.unique(labels)
    if len(classes) < 2:
        return {'status': 'skipped', 'reason': 'At least two categories are required.'}
    per_class = []
    for category in classes:
        selected = np.unique(groups[labels == category])
        if any(len(np.unique(labels[groups == group])) != 1 for group in selected):
            return {'status': 'skipped', 'reason': 'An overlap group crosses category labels.'}
        per_class.append(selected)
    folds = min(int(folds), min(map(len, per_class)))
    if folds < 2:
        return {'status': 'skipped', 'reason': 'Fewer than two independent overlap groups per category.'}
    fold_ids = np.full(len(values), -1, dtype=np.int32)
    rng = np.random.default_rng(seed)
    for group_list in per_class:
        for index, group in enumerate(rng.permutation(group_list)):
            fold_ids[groups == group] = index % folds
    encoded = np.searchsorted(classes, labels)
    targets = np.eye(len(classes), dtype=np.float64)[encoded]
    predictions = np.full(len(values), -1, dtype=np.int32)
    for fold in range(folds):
        fit, held = fold_ids != fold, fold_ids == fold
        mean, scale = values[fit].mean(axis=0), values[fit].std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        training, validation = (values[fit] - mean) / scale, (values[held] - mean) / scale
        target_mean = targets[fit].mean(axis=0)
        # Dual ridge avoids a parameters-by-parameters solve for fast updates.
        coefficients = np.linalg.solve(training @ training.T + ridge * np.eye(sum(fit)),
                                       targets[fit] - target_mean)
        scores = validation @ training.T @ coefficients + target_mean
        predictions[held] = np.argmax(scores, axis=1)
    correct = predictions == encoded
    return {
        'status': 'ok', 'accuracy': float(np.mean(correct)),
        'macro_category_accuracy': float(np.mean([np.mean(correct[labels == label]) for label in classes])),
        'categories': classes.tolist(), 'tasks': len(values), 'overlap_groups': len(np.unique(groups)),
        'folds': folds, 'fold_ids': fold_ids.tolist(),
        'predicted_categories': classes[predictions].tolist(),
        'ridge': float(ridge), 'fold_seed': int(seed),
        'scaling': 'mean and standard deviation fitted on each training fold only',
        'protocol': 'fixed hyperparameters; duplicate/program groups never cross folds',
        'limitation': 'Within-analysis-split decoding; no adaptation competence claim.',
    }


def _overlap_groups(records, store):
    """Conservatively join shared support OR query clusters across episodes."""
    parents = list(range(len(records)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    seen, clusters_per_record = {}, []
    for index, record in enumerate(records):
        ids = list(record['support_ids']) + list(record['query_ids'])
        clusters = sorted({store.get(item).duplicate_cluster_id for item in ids})
        clusters_per_record.append(clusters)
        for cluster in clusters:
            if cluster in seen:
                left, right = find(index), find(seen[cluster])
                parents[max(left, right)] = min(left, right)
            else:
                seen[cluster] = index
    return np.asarray([find(index) for index in range(len(records))], np.int32), clusters_per_record


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def extract_checkpoint_updates(checkpoint, episodes, output, cache_root=None, manifest_path=None):
    """Export frozen-checkpoint WRITE diagnostics and shared public responses.

    Original query labels are used only for offline oracle gradients and losses.
    Across-task functional vectors use one fixed public history, never each
    task's distinct teacher-forced query prefix. No policy update is persisted.
    """
    # Keep lightweight array analysis usable without importing the train stack.
    import jax
    import jax.numpy as jnp
    from icil_jax_rlbench.models.fast_weight_ttt import fast_read_residual, initial_fast_state
    from icil_jax_rlbench.quickdraw.evaluate import load_evaluation
    from icil_jax_rlbench.quickdraw.model import (
        SketchModelConfig, first_write_diagnostics, initial_query_carry, query_step,
    )
    from icil_jax_rlbench.quickdraw.train import load_run

    payload, cfg, store, manifest = load_run(checkpoint, cache_root=cache_root, manifest_path=manifest_path)
    model_cfg = SketchModelConfig(**cfg['model'])
    if not model_cfg.model_type.startswith('ttt_'):
        raise ValueError('Checkpoint update extraction requires a TTT model.')
    sections, evaluation = load_evaluation(episodes)
    if evaluation['cache_id'] != store.identifier or evaluation['manifest_id'] != manifest['identifier']:
        raise ValueError('Evaluation episodes and checkpoint manifests differ.')
    if evaluation['experiment'] != cfg['experiment'] or evaluation['max_steps'] != model_cfg.max_steps:
        raise ValueError('Evaluation public protocol differs from checkpoint.')
    if cfg['experiment'] == 'b2' and evaluation.get('motion_bound') != model_cfg.motion_bound:
        raise ValueError('Evaluation motion bound differs from the checkpoint target schedule.')
    records = evaluation['records']
    if not records:
        raise ValueError('At least one saved evaluation task is required.')
    for role, count_key in (('support', 'support_count'), ('query', 'query_count')):
        expected = (len(records), int(evaluation[count_key]), model_cfg.max_steps)
        if sections[role]['tokens'].shape != expected + (4,):
            raise ValueError(f'{role} shape differs from the saved evaluation protocol.')
        if any(len(record[f'{role}_ids']) != int(evaluation[count_key]) for record in records):
            raise ValueError(f'{role} IDs differ from the saved evaluation protocol.')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])
    initial = initial_fast_state(params)
    diagnostic = jax.jit(lambda support, query: first_write_diagnostics(params, support, query, model_cfg))
    probe_steps = min(8, model_cfg.max_steps)
    probe_previous = jnp.zeros((probe_steps, 4), jnp.float32)
    probe_states = jnp.zeros((probe_steps, 3), jnp.float32)
    probe_frame = jnp.asarray([0., 0., 0., 1.], jnp.float32)
    probe_times = jnp.arange(probe_steps)

    def response(fast):
        def step(carry, item):
            previous, state, time = item
            hidden, distribution = query_step(params, fast, model_cfg, carry, previous,
                                               time, probe_frame, state)
            read = fast_read_residual(params, fast, hidden, model_cfg.fast_config(),
                                      read_mode='delta', read_scale=model_cfg.read_scale)
            return hidden, (distribution, read)
        _, result = jax.lax.scan(step, initial_query_carry(model_cfg),
                                 (probe_previous, probe_states, probe_times))
        return result
    response = jax.jit(response)
    def vector(tree):
        return np.concatenate([np.asarray(value).reshape(-1) for value in jax.tree_util.tree_leaves(tree)])
    def layout(tree):
        leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
        offset, rows = 0, []
        for path, value in leaves:
            size = int(value.size)
            rows.append({'path': '/'.join(str(getattr(item, 'key', getattr(item, 'idx', item))) for item in path),
                         'shape': list(value.shape), 'offset': offset, 'size': size})
            offset += size
        return rows
    baseline_tree, baseline_read = response(initial)
    baseline = vector(baseline_tree)
    features = {}
    mapping = {
        'raw_support_statistics': 'raw_support_statistics',
        'raw_first_write_gradient': 'raw_write_gradient',
        'clipped_first_write_gradient': 'clipped_write_gradient',
        'actual_first_update': 'first_update', 'final_fast_delta': 'final_fast_delta',
        'oracle_query_gradient': 'query_gradient',
    }
    scalar_fields = ('write_loss', 'query_loss_before', 'query_loss_after_first_update',
                     'query_loss_after_adaptation', 'first_query_improvement',
                     'local_improvement_prediction', 'raw_write_gradient_query_cosine',
                     'first_update_query_gradient_cosine')
    for index in range(len(records)):
        support, query = ({name: jnp.asarray(value[index]) for name, value in sections[role].items()}
                          for role in ('support', 'query'))
        result = diagnostic(support, query)
        for target, source in mapping.items():
            features.setdefault(target, []).append(vector(result[source]))
        for name in scalar_fields:
            features.setdefault(name, []).append(np.asarray(result[name]))
        adapted = jax.tree_util.tree_map(lambda w0, delta: w0 + delta, initial, result['final_fast_delta'])
        prediction, read = response(adapted)
        features.setdefault('functional_delta', []).append(vector(prediction) - baseline)
        features.setdefault('public_read_delta', []).append(vector(read - baseline_read))
    arrays = {name: np.stack(value) for name, value in features.items()}
    arrays.update({f'raw_support_{name}': np.asarray(value) for name, value in sections['support'].items()})
    arrays.update(public_probe_previous_tokens=np.asarray(probe_previous),
                  public_probe_states=np.asarray(probe_states), public_probe_frame=np.asarray(probe_frame),
                  public_probe_timesteps=np.asarray(probe_times), public_functional_baseline=baseline)
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise FloatingPointError('Nonfinite checkpoint update diagnostics.')
    groups, clusters = _overlap_groups(records, store)
    arrays['probe_overlap_groups'] = groups
    labels = [record['intended_category'] for record in records]
    category_probes = {}
    for name in ('raw_support_statistics', 'raw_first_write_gradient', 'actual_first_update',
                 'final_fast_delta', 'functional_delta'):
        category_probes[name] = (
            {'status': 'skipped', 'reason': 'No analysis probes are fitted on untouched final-test episodes.'}
            if evaluation['split'] == 'test' else grouped_category_probe(arrays[name], labels, groups)
        )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'features.npz', **arrays)
    provenance = {
        'schema_version': 1, 'checkpoint_sha256': _sha256(checkpoint),
        'checkpoint_step': int(payload['step']), 'model_config': cfg['model'],
        'cache_id': store.identifier, 'manifest_id': manifest['identifier'],
        'evaluation_id': evaluation['identifier'], 'evaluation_split': evaluation['split'],
        'evaluation_data_sha256': evaluation['data_sha256'],
        'feature_sha256': _sha256(output / 'features.npz'),
        'analysis_source_sha256': _sha256(__file__),
        'model_source_sha256': _sha256(Path(__file__).with_name('model.py')),
        'public_probe': {'steps': probe_steps, 'previous_tokens': 'fixed zeros at every step',
                         'state': 'fixed zero XY/pen', 'frame': 'identity',
                         'clock': 'integer timestep divided by the checkpoint max_steps'},
        'functional_vector_order': 'JAX sorted PyTree leaves of output distribution parameters',
        'fast_vector_order': 'JAX sorted PyTree leaves, flattened in row-major order',
        'fast_tensor_layout': layout(initial), 'functional_tensor_layout': layout(baseline_tree),
        'raw_support_statistics_layout': ['mean_x', 'mean_y', 'mean_incoming_pen', 'mean_stop',
                                          'variance_x', 'variance_y', 'variance_incoming_pen', 'variance_stop'],
        'query_label_boundary': 'Original query targets appear only in offline oracle gradients/losses; never WRITE or common probes.',
        'probe_grouping': 'Connected overlap of support and query duplicate clusters; repeated B programs share a group.',
        'records': [{**record, 'analysis_row': index, 'condition': 'correct_support',
                     'probe_overlap_group': int(groups[index]),
                     'duplicate_clusters': clusters[index]} for index, record in enumerate(records)],
    }
    summary = summarize_update_archive(output / 'features.npz')
    summary.update(schema_version=1, checkpoint_sha256=provenance['checkpoint_sha256'],
                   evaluation_id=evaluation['identifier'], tasks=len(records),
                   category_probes=category_probes,
                   actual_first_query_improvement={
                       'mean': float(arrays['first_query_improvement'].mean()),
                       'positive_fraction': float(np.mean(arrays['first_query_improvement'] > 0)),
                       'statistical_unit': 'Saved task episodes; repeated categories/programs are not independent research replicates.',
                   },
                   interpretation='Single-checkpoint descriptive diagnostics; no threshold or held-out adaptation claim.')
    summary['neighborhood_diversity'] = {
        name: neighborhood_response_diversity(arrays[name], records)
        for name in ('actual_first_update', 'final_fast_delta', 'functional_delta')
    }
    for name, value in (('records.json', provenance), ('summary.json', summary)):
        (output / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return output
