"""Frozen, paired evaluation of supervised full-dataset sketch policies.

Classifier embeddings select evaluation tasks and references offline. Model
generation sees only the shown sketch tokens and their masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .classifier_data import _progress, _status
from .metrics import (
    RENDERER_CONFIG, file_sha256, geometry_descriptor, load_trajectories,
    paired_geometry, save_trajectories, trajectory_statistics,
)


CONDITIONS = ('correct_support', 'no_context', 'same_category_wrong_neighborhood',
              'wrong_support', 'support_copy', 'nearest_training_example')


def _identifier(value):
    payload = {key: item for key, item in value.items() if key != 'identifier'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def _category_name(dataset, category):
    categories = dataset.manifest['categories']
    return str(categories[str(category)] if isinstance(categories, dict) else categories[category])


def _ids(dataset, rows):
    return [str(dataset.base_ids[int(row)]) for row in rows]


def _sequences(dataset, rows):
    rows = np.asarray(rows, np.int64)
    tokens = np.asarray(dataset.tokens[rows]).copy()
    lengths = np.asarray(dataset.lengths[rows], np.int32)
    time = np.arange(tokens.shape[-2])
    return {'tokens': tokens, 'event_mask': time < lengths[..., None] + 1,
            'point_mask': time < lengths[..., None], 'lengths': lengths,
            'stopped': np.ones(rows.shape, bool)}


def _feature_rows(dataset, embeddings, rows):
    return np.asarray([embeddings.row(str(dataset.base_ids[int(row)])) for row in rows], np.int64)


def _task_description(dataset, embeddings, target):
    neighbors = np.asarray(dataset.neighbors[int(target)], np.int64)
    neighbors = neighbors[neighbors >= 0]
    members = np.r_[int(target), neighbors]
    center = np.asarray(embeddings.cosine[_feature_rows(dataset, embeddings, members)], np.float64).mean(0)
    center /= max(float(np.linalg.norm(center)), 1e-12)
    lengths = np.asarray(dataset.lengths[members])
    tokens = np.asarray(dataset.tokens[members])
    strokes = ((tokens[..., 2] < .5) & (np.arange(tokens.shape[1]) < lengths[:, None])).sum(1)
    return {'target': int(target), 'members': set(map(int, members)),
            'target_group': int(dataset.duplicate_ids[int(target)]),
            'groups': set(map(int, dataset.duplicate_ids[members])), 'centroid': center,
            'complexity': np.asarray([lengths.mean(), strokes.mean()])}


def _wrong_neighborhood(intended, candidates, *, minimum_distance, maximum_jaccard):
    choices = []
    for candidate in candidates:
        if intended['target_group'] in candidate['groups']:
            continue
        union = intended['groups'] | candidate['groups']
        overlap = len(intended['groups'] & candidate['groups']) / max(len(union), 1)
        distance = float(np.clip(1. - intended['centroid'] @ candidate['centroid'], 0., 2.))
        if distance < minimum_distance or overlap > maximum_jaccard:
            continue
        mismatch = float(np.linalg.norm((candidate['complexity'] - intended['complexity']) /
                                        np.maximum(intended['complexity'], 1.)))
        choices.append((mismatch, -distance, candidate['target'], overlap))
    if not choices:
        raise ValueError(f"No separated same-category wrong neighborhood for target row {intended['target']}; "
                         'increase --negative-candidates or explicitly revise the separation thresholds.')
    mismatch, negative_distance, target, overlap = min(choices)
    return target, {'source_target_row': target, 'centroid_cosine_distance': -negative_distance,
                    'member_jaccard': overlap, 'relative_length_stroke_mismatch': mismatch,
                    'minimum_centroid_cosine_distance': minimum_distance,
                    'maximum_member_jaccard': maximum_jaccard}


def _select_references(dataset, embeddings, target, candidates, count):
    vectors = np.asarray(embeddings.cosine[_feature_rows(dataset, embeddings, candidates)], np.float64)
    vector = np.asarray(embeddings.cosine[embeddings.row(str(dataset.base_ids[int(target)]))], np.float64)
    scores = vectors @ vector
    # Float64 cosine score, then immutable ID for exact ties.
    order = np.lexsort((np.asarray(_ids(dataset, candidates)), -scores))
    return np.asarray(candidates, np.int64)[order[:count]]


def prepare_evaluation(dataset_root, embedding_root, output, *, split='development',
                       tasks_per_category=4, support_count=4, selection_mode='exact_top_k',
                       seed=0, reference_count=8, allow_test=False,
                       negative_candidates=128, wrong_neighborhood_min_distance=.05,
                       wrong_neighborhood_max_jaccard=.1, progress=True):
    """Freeze balanced held-out targets, all controls, and local real references.

    Evaluation caps never restrict the training reservoir. Global reference
    halves are assigned before query-local retrieval, allowing overlapping local
    neighborhoods while keeping real_a and real_b globally disjoint.
    """
    from .embeddings import EmbeddingStore
    from .full_data import FullDataset

    selection_mode = 'sample_top_m' if selection_mode == 'sampled_top_m' else selection_mode
    if split not in ('development', 'test') or (split == 'test' and not allow_test):
        raise ValueError('Use development evaluation; test requires explicit --allow-test.')
    if (tasks_per_category < 1 or support_count < 1 or negative_candidates < 2 or
            reference_count < 4 or reference_count % 2):
        raise ValueError('Positive tasks/supports, >=2 negative candidates, and an even reference count >=4 are required.')
    if not 0 <= wrong_neighborhood_min_distance <= 2 or not 0 <= wrong_neighborhood_max_jaccard <= 1:
        raise ValueError('Wrong-neighborhood distance/Jaccard thresholds must lie in [0,2]/[0,1].')
    _status('Verifying full dataset and frozen embeddings ...', progress)
    dataset, embeddings = FullDataset.open(dataset_root), EmbeddingStore.open(embedding_root)
    if dataset.manifest.get('embedding_identifier') != embeddings.identifier:
        raise ValueError('Evaluation embeddings differ from the dataset curation resource.')
    if support_count > dataset.neighbors.shape[1]:
        raise ValueError('Support count exceeds the stored nearest-neighbor table width.')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Evaluation outputs are immutable: {output}')
    rng = np.random.default_rng(seed)
    split_rows = np.asarray(dataset.rows(split), np.int64)
    category_ids = np.unique(dataset.category_ids[split_rows])
    if len(category_ids) < 2:
        raise ValueError('Wrong-category controls require at least two categories.')
    selected, category_pools, descriptions, reference_halves = [], {}, {}, {}
    for category in _progress(category_ids, 'Constructing evaluation candidate neighborhoods',
                              unit='categories', enabled=progress):
        category = int(category)
        candidates = split_rows[dataset.category_ids[split_rows] == category]
        if len(candidates) < tasks_per_category:
            raise ValueError(f'Category {category} has fewer targets than tasks_per_category.')
        candidates = rng.permutation(candidates)
        _, first_duplicate = np.unique(dataset.duplicate_ids[candidates], return_index=True)
        candidates = candidates[np.sort(first_duplicate)]
        if len(candidates) < tasks_per_category:
            raise ValueError(f'Category {category} has fewer unique target groups than tasks_per_category.')
        targets = candidates[:tasks_per_category]
        selected.append(targets)
        category_pools[category] = candidates[:max(tasks_per_category, negative_candidates)]
        descriptions[category] = [_task_description(dataset, embeddings, row)
                                  for row in category_pools[category]]
        references = np.asarray(dataset.reference_rows(split, category), np.int64)
        # This partition is global per split/category, independent of target.
        references = references[np.argsort(np.asarray(_ids(dataset, references)))]
        _, first_duplicate = np.unique(dataset.duplicate_ids[references], return_index=True)
        references = references[np.sort(first_duplicate)]
        half_a, half_b = references[::2], references[1::2]
        if min(len(half_a), len(half_b)) < reference_count // 2:
            raise ValueError(f'Category {category} needs >= {reference_count} reserved references.')
        if np.intersect1d(references, split_rows).size:
            raise ValueError('Reserved references overlap policy examples.')
        reference_halves[category] = (half_a, half_b)
    # Round-robin categories; each contributes exactly tasks_per_category.
    targets = np.stack(selected, axis=1).reshape(-1)
    proper = dataset.batch(targets, support_count=support_count, selection_mode=selection_mode, rng=rng)
    supports = np.asarray(proper['metadata']['support_rows'], np.int64)
    negative_pool = [item for category in category_ids for item in descriptions[int(category)]]
    negative_targets = np.asarray([item['target'] for item in negative_pool], np.int64)
    negative_complexity = np.stack([item['complexity'] for item in negative_pool])
    negative_categories = dataset.category_ids[negative_targets]
    negatives, other_categories, refs_a, refs_b, records, pairings = [], [], [], [], [], []
    for target, support in _progress(zip(targets, supports), 'Freezing controls and local references',
                                     total=len(targets), unit='tasks', enabled=progress):
        category = int(dataset.category_ids[target])
        intended = _task_description(dataset, embeddings, target)
        negative, pairing = _wrong_neighborhood(intended, descriptions[category],
            minimum_distance=wrong_neighborhood_min_distance,
            maximum_jaccard=wrong_neighborhood_max_jaccard)
        mismatches = np.linalg.norm((negative_complexity - intended['complexity']) /
                                    np.maximum(intended['complexity'], 1.), axis=1)
        mismatches[negative_categories == category] = np.inf
        other = int(negative_targets[np.lexsort((negative_targets, mismatches))[0]])
        negatives.append(negative)
        other_categories.append(other)
        first, second = reference_halves[category]
        refs_a.append(_select_references(dataset, embeddings, target, first, reference_count // 2))
        refs_b.append(_select_references(dataset, embeddings, target, second, reference_count // 2))
        record = {'task_id': f'ann/{int(target)}', 'intended_neighborhood_id': f'ann/{int(target)}',
                  'intended_category': _category_name(dataset, category), 'query_id': str(dataset.base_ids[target]),
                  'query_ids': [str(dataset.base_ids[target])], 'target_row': int(target),
                  'support_ids': _ids(dataset, support), 'eligible_member_ids': _ids(dataset, sorted(intended['members'])),
                  'a_protocol': 'a_nn', 'local_reference_role': 'offline_target_local_distribution_proxy',
                  'category_id': category, 'centroid': intended['centroid'].tolist(),
                  'duplicate_cluster_id': str(int(dataset.duplicate_ids[target])),
                  'synthetic_fixture': bool(dataset.manifest.get('allow_subset_fixture', False)),
                  'reference_ids': {'real_a': _ids(dataset, refs_a[-1]), 'real_b': _ids(dataset, refs_b[-1])}}
        records.append(record)
        pairings.append(pairing)
    local = dataset.batch(np.asarray(negatives), support_count=support_count,
                          selection_mode=selection_mode, rng=rng)
    wrong = dataset.batch(np.asarray(other_categories), support_count=support_count,
                          selection_mode=selection_mode, rng=rng)
    arrays = {'target_rows': targets, 'support_rows': supports,
              'same_category_wrong_neighborhood_rows': np.asarray(local['metadata']['support_rows'], np.int64),
              'wrong_support_rows': np.asarray(wrong['metadata']['support_rows'], np.int64),
              'reference_rows_real_a': np.stack(refs_a), 'reference_rows_real_b': np.stack(refs_b),
              'negative_target_rows': np.asarray(negatives, np.int64),
              'wrong_category_target_rows': np.asarray(other_categories, np.int64)}
    all_references = set(map(int, np.r_[arrays['reference_rows_real_a'].ravel(), arrays['reference_rows_real_b'].ravel()]))
    for role in ('support_rows', 'same_category_wrong_neighborhood_rows', 'wrong_support_rows'):
        if all_references & set(map(int, arrays[role].ravel())):
            raise ValueError('Reference rows overlap shown support rows.')
    if set(map(int, arrays['reference_rows_real_a'].ravel())) & set(map(int, arrays['reference_rows_real_b'].ravel())):
        raise ValueError('Global reference halves overlap.')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'episodes.npz', **arrays)
    metadata = {'schema_version': 1, 'evaluation_version': 'supervised_full_ann_v1',
                'dataset_identifier': dataset.identifier, 'embedding_identifier': embeddings.identifier,
                'split': split, 'seed': int(seed), 'support_count': int(support_count),
                'selection_mode': selection_mode, 'tasks_per_category': int(tasks_per_category),
                'reference_count': int(reference_count), 'records': records,
                'same_category_wrong_pairings': pairings, 'max_steps': int(dataset.tokens.shape[1]),
                'negative_candidate_count_per_category': int(negative_candidates),
                'negative_selection': 'bounded split-local candidate tasks; separated neighborhood; minimum relative length/stroke mismatch',
                'reference_protocol': 'globally_disjoint_reserved_halves_then_query_local_cosine_retrieval',
                'data_sha256': file_sha256(output / 'episodes.npz'),
                'a_protocol': 'a_nn', 'experiment': 'a', 'model_inputs': 'support tokens and masks only'}
    metadata['identifier'] = _identifier(metadata)
    _dump(output / 'metadata.json', metadata)
    for half in ('real_a', 'real_b'):
        rows = arrays[f'reference_rows_{half}']
        reference_records = [{**record, 'drawing_id': str(dataset.base_ids[row]),
                              'base_id': str(dataset.base_ids[row]), 'support_ids': [],
                              'duplicate_cluster_id': str(int(dataset.duplicate_ids[row])),
                              'condition': 'independent_real_reference'}
                             for record, task_rows in zip(records, rows) for row in task_rows]
        save_trajectories(output / half, _sequences(dataset, rows.ravel()), {
            'schema_version': 1, 'coordinate_mode': 'absolute', 'pen_semantics': 'incoming',
            'renderer': RENDERER_CONFIG, 'records': reference_records,
            'expected_count': len(reference_records), 'manifest_id': metadata['identifier'],
            'dataset_identifier': dataset.identifier, 'reference_half': half,
            'reference_protocol': metadata['reference_protocol']})
    shown = np.unique(np.concatenate([arrays[role].ravel() for role in (
        'support_rows', 'same_category_wrong_neighborhood_rows', 'wrong_support_rows')]))
    save_trajectories(output / 'shown_supports', _sequences(dataset, shown), {
        'schema_version': 1, 'coordinate_mode': 'absolute', 'pen_semantics': 'incoming',
        'renderer': RENDERER_CONFIG, 'manifest_id': metadata['identifier'],
        'records': [{'drawing_id': str(dataset.base_ids[row]),
                     'intended_category': _category_name(dataset, int(dataset.category_ids[row])),
                     'support_ids': [], 'condition': 'shown_support_population'} for row in shown]})
    return output


def load_evaluation(path, dataset=None):
    path = Path(path)
    metadata = json.loads((path / 'metadata.json').read_text())
    if metadata.get('evaluation_version') != 'supervised_full_ann_v1' or metadata.get('identifier') != _identifier(metadata):
        raise ValueError('Frozen supervised evaluation metadata changed.')
    if metadata['data_sha256'] != file_sha256(path / 'episodes.npz'):
        raise ValueError('Frozen supervised evaluation arrays changed.')
    with np.load(path / 'episodes.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if dataset is not None:
        if metadata['dataset_identifier'] != dataset.identifier:
            raise ValueError('Frozen evaluation and checkpoint datasets differ.')
        if metadata['max_steps'] != dataset.tokens.shape[1]:
            raise ValueError('Frozen evaluation sequence length differs.')
        allowed = set(map(int, dataset.rows(metadata['split'])))
        for role in ('target_rows', 'support_rows', 'same_category_wrong_neighborhood_rows', 'wrong_support_rows'):
            if not set(map(int, arrays[role].ravel())) <= allowed:
                raise ValueError('Evaluation examples leave the declared split.')
    return arrays, metadata


def _numeric_batch(dataset, targets, support_rows, *, drop_context=False):
    supports, query = _sequences(dataset, support_rows), _sequences(dataset, targets)
    return {'support_tokens': np.zeros_like(supports['tokens']) if drop_context else supports['tokens'],
            'support_mask': np.zeros_like(supports['event_mask']) if drop_context else supports['event_mask'],
            'query_tokens': query['tokens'], 'query_mask': query['event_mask'],
            'query_point_mask': query['point_mask'], 'example_mask': np.ones(len(targets), bool)}


def _geometry_descriptors(dataset, rows, *, chunk_size=4096):
    """Vectorized raw 16-point XY and length/stroke descriptors, in bounded RAM."""
    rows = np.asarray(rows, np.int64)
    result = np.empty((len(rows), 34), np.float32)
    for start in range(0, len(rows), chunk_size):
        selected = rows[start:start + chunk_size]
        tokens = np.asarray(dataset.tokens[selected])
        lengths = np.asarray(dataset.lengths[selected])
        positions = (lengths[:, None] - 1) * np.linspace(0., 1., 16)[None]
        left = np.floor(positions).astype(np.int64)
        right = np.minimum(left + 1, lengths[:, None] - 1)
        weight = (positions - left)[..., None]
        batch_index = np.arange(len(selected))[:, None]
        xy = tokens[batch_index, left, :2] * (1. - weight) + tokens[batch_index, right, :2] * weight
        strokes = ((tokens[..., 2] < .5) & (np.arange(tokens.shape[1]) < lengths[:, None])).sum(1)
        result[start:start + len(selected)] = np.concatenate((xy.reshape(len(selected), 32),
                                                            lengths[:, None] / 128., strokes[:, None] / 16.), axis=1)
    return result


def _nearest_descriptors(queries, candidates, *, chunk_size=32768):
    """Exact raw-descriptor search; ties retain the earliest immutable row."""
    queries = np.asarray(queries, np.float64)
    best = np.full(len(queries), np.inf)
    indices = np.zeros(len(queries), np.int64)
    for start in range(0, len(candidates), chunk_size):
        block = np.asarray(candidates[start:start + chunk_size], np.float64)
        distance = np.maximum((queries * queries).sum(1)[:, None] +
                              (block * block).sum(1)[None] - 2. * queries @ block.T, 0.)
        index = distance.argmin(1)
        values = distance[np.arange(len(queries)), index]
        update = values < best
        best[update], indices[update] = values[update], start + index[update]
    return indices, np.sqrt(best)


def generate_run(checkpoint, episodes, output, *, dataset_root=None, conditions=None,
                 samples_per_task=4, batch_size=8, seed=0, allow_test=False, progress=True):
    """Generate each condition with identical per-task/sample random keys."""
    import jax
    import jax.numpy as jnp
    from .supervised_models import generate, loss
    from .supervised_train import load_run

    conditions = tuple(conditions or CONDITIONS[:-1])
    if not conditions or set(conditions) - set(CONDITIONS) or len(set(conditions)) != len(conditions):
        raise ValueError('Use distinct known evaluation conditions.')
    if samples_per_task < 1 or batch_size < 1:
        raise ValueError('Positive samples-per-task and batch size are required.')
    _status('Verifying checkpoint, full dataset, and frozen evaluation ...', progress)
    payload, cfg, model_cfg, dataset = load_run(checkpoint, dataset_root=dataset_root)
    arrays, evaluation = load_evaluation(episodes, dataset)
    if evaluation['split'] == 'test' and not allow_test:
        raise ValueError('Test generation requires explicit --allow-test.')
    if evaluation['support_count'] != cfg['support_count']:
        raise ValueError('Evaluation support count differs from training configuration.')
    if evaluation['selection_mode'] != cfg['selection_mode']:
        raise ValueError('Evaluation neighbor-selection mode differs from training configuration.')
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])

    def generate_one(tokens, mask, key):
        result = generate(params, tokens[None], mask[None], model_cfg, key)
        return {name: result[name][0] for name in ('tokens', 'event_mask', 'point_mask', 'length', 'stopped')}

    def loss_one(batch, key):
        expanded = jax.tree_util.tree_map(lambda value: value[None], batch)
        value, parts = loss(params, expanded, model_cfg, key, training=False)
        return {'loss': value, **parts}

    generate_compiled = jax.jit(jax.vmap(generate_one))
    loss_compiled = jax.jit(jax.vmap(loss_one))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    checkpoint_id = file_sha256(checkpoint)
    target_rows = arrays['target_rows']
    base_key = jax.random.PRNGKey(seed)
    train_rows, train_descriptors, nearest_rows, nearest_distances = None, None, None, None
    if 'nearest_training_example' in conditions:
        _status('Finding nearest training examples using only shown support geometry ...', progress)
        train_rows = np.sort(np.asarray(dataset.rows('train'), np.int64))
        train_descriptors = _geometry_descriptors(dataset, train_rows)
        support_descriptors = _geometry_descriptors(dataset, arrays['support_rows'].ravel()).reshape(
            len(target_rows), evaluation['support_count'], -1).mean(1)
        indices, nearest_distances = _nearest_descriptors(support_descriptors, train_descriptors)
        nearest_rows = train_rows[indices]
    for condition in conditions:
        role = condition + '_rows' if condition in ('wrong_support', 'same_category_wrong_neighborhood') else 'support_rows'
        shown_rows = arrays[role]
        drop_context = condition == 'no_context' or not cfg.get('condition_on_support', True)
        is_baseline = condition in ('support_copy', 'nearest_training_example')
        task_losses = []
        if not is_baseline:
            for start in _progress(range(0, len(target_rows), batch_size),
                                    f'{condition}: held-out target loss', unit='batches', enabled=progress):
                count = min(batch_size, len(target_rows) - start)
                # Padding stabilizes compiled shapes; unused outputs are dropped.
                indices = np.minimum(np.arange(start, start + batch_size), len(target_rows) - 1)
                batch = _numeric_batch(dataset, target_rows[indices], shown_rows[indices], drop_context=drop_context)
                keys = jnp.stack([jax.random.fold_in(jax.random.fold_in(base_key, 0x4C4F5353), int(i)) for i in indices])
                losses = jax.tree_util.tree_map(np.asarray, loss_compiled(batch, keys))
                task_losses.extend([{name: float(value[i]) for name, value in losses.items()} for i in range(count)])
        results, records = [], []
        total = len(target_rows) * samples_per_task
        for start in _progress(range(0, total, batch_size), f'{condition}: generation',
                                unit='batches', enabled=progress):
            count = min(batch_size, total - start)
            sample_numbers = np.minimum(np.arange(start, start + batch_size), total - 1)
            task_indices = sample_numbers // samples_per_task
            sample_indices = sample_numbers % samples_per_task
            # Fold task and sample separately: stable across chunk sizes and number of samples.
            keys = jnp.stack([jax.random.fold_in(jax.random.fold_in(base_key, int(task)), int(sample))
                              for task, sample in zip(task_indices, sample_indices)])
            batch = _numeric_batch(dataset, target_rows[task_indices], shown_rows[task_indices], drop_context=drop_context)
            if is_baseline:
                copied = (arrays['support_rows'][task_indices, sample_indices % evaluation['support_count']]
                          if condition == 'support_copy' else nearest_rows[task_indices])
                generated = _sequences(dataset, copied)
                generated['length'] = generated.pop('lengths')
            else:
                generated = jax.tree_util.tree_map(np.asarray, generate_compiled(
                    batch['support_tokens'], batch['support_mask'], keys))
            for i in range(count):
                task, sample = int(task_indices[i]), int(sample_indices[i])
                result = {name: value[i] for name, value in generated.items()}
                results.append(result)
                shown = [] if drop_context and not is_baseline else _ids(dataset, shown_rows[task])
                record = {**evaluation['records'][task], 'condition': condition,
                          'support_ids': shown, 'original_support_ids': evaluation['records'][task]['support_ids'],
                          'query_seed': np.asarray(keys[i]).astype(int).tolist(), 'task_index': task,
                          'sample_index': sample, 'checkpoint_id': checkpoint_id,
                          'length': int(result['length']), 'stopped': bool(result['stopped'])}
                source_row = (arrays['negative_target_rows'][task] if condition == 'same_category_wrong_neighborhood'
                              else arrays['wrong_category_target_rows'][task] if condition == 'wrong_support'
                              else target_rows[task])
                record['source_neighborhood_id'] = f'ann/{int(source_row)}' if shown else None
                if condition == 'nearest_training_example':
                    record.update(copied_training_id=str(dataset.base_ids[nearest_rows[task]]),
                                  nearest_training_support_geometry_distance=float(nearest_distances[task]),
                                  nearest_training_selection='all training rows; mean shown-support raw 16-point XY and length/stroke descriptor')
                if not is_baseline:
                    record['conditional_target_loss'] = task_losses[task]
                if shown:
                    points = result['tokens'][result['point_mask'], :3]
                    support_sequences = _sequences(dataset, shown_rows[task])
                    geometry = [paired_geometry(points, tokens[mask, :3]) for tokens, mask in zip(
                        support_sequences['tokens'], support_sequences['point_mask'])]
                    valid = [item for item in geometry if item.get('valid_pair')]
                    record['raw_support_geometry'] = {
                        'evaluable': bool(valid),
                        'minimum_chamfer_squared': min((item['symmetric_chamfer_squared'] for item in valid), default=None),
                        'minimum_ordered_point_rmse': min((item['resampled_ordered_point_rmse'] for item in valid), default=None),
                        'generated_stroke_starts': int(np.sum(points[:, 2] < .5)),
                        'interpretation': 'copying diagnostic; not paired target reconstruction'}
                records.append(record)
        trajectories = {name: np.stack([item[name] for item in results])
                        for name in ('tokens', 'event_mask', 'point_mask')}
        trajectories.update(lengths=np.asarray([item['length'] for item in results], np.int32),
                            stopped=np.asarray([item['stopped'] for item in results], bool))
        save_trajectories(output / condition, trajectories, {
            'schema_version': 1, 'coordinate_mode': 'absolute', 'pen_semantics': 'incoming',
            'renderer': RENDERER_CONFIG, 'records': records, 'expected_count': total,
            'manifest_id': evaluation['identifier'], 'evaluation_id': evaluation['identifier'],
            'dataset_identifier': dataset.identifier, 'checkpoint_id': checkpoint_id,
            'model_type': model_cfg.architecture, 'experiment': 'a', 'a_protocol': 'a_nn',
            'protocol': 'empty_prefix_generation_from_context_only',
            'pairing': 'identical task/sample PRNG keys and intended target/reference identities across conditions'})
        summaries[condition] = trajectory_statistics(trajectories)
        if task_losses:
            summaries[condition]['conditional_target_loss'] = {
                'mean': float(np.mean([item['loss'] for item in task_losses])),
                'per_task': task_losses, 'task_count': len(task_losses),
                'objective': ('paired held-out denoising MSE; not a likelihood or comparable to AR NLL'
                              if model_cfg.architecture == 'diffusion' else
                              'teacher-forced mixture XY NLL plus pen/STOP BCE'),
                'pairing': 'same held-out targets and PRNG keys; diffusion time/noise paired across support conditions'}
    _dump(output / 'summary.json', {'checkpoint_id': checkpoint_id, 'evaluation_id': evaluation['identifier'],
                                  'conditions': summaries, 'samples_per_task': samples_per_task,
                                  'seed': seed, 'test_opt_in': allow_test})
    return output


def geometry_copying(dataset_root, generated, output, *, query_batch_size=64, progress=True):
    """Exact descriptor proximity to all training rows and actual shown supports."""
    from .full_data import FullDataset

    if query_batch_size < 1:
        raise ValueError('Positive query batch size required.')
    dataset = FullDataset.open(dataset_root)
    arrays, metadata = load_trajectories(generated)
    if metadata.get('dataset_identifier') != dataset.identifier:
        raise ValueError('Generated trajectories and geometry-copying dataset differ.')
    training_rows = np.sort(np.asarray(dataset.rows('train'), np.int64))
    _status(f'Building raw geometry descriptors for {len(training_rows):,} training examples ...', progress)
    descriptors = _geometry_descriptors(dataset, training_rows)
    by_id = {str(value): i for i, value in enumerate(dataset.base_ids)}
    reports = []
    for start in _progress(range(0, len(arrays['tokens']), query_batch_size), 'Checking raw geometry copying',
                            unit='batches', enabled=progress):
        points = [tokens[mask, :3] for tokens, mask in zip(
            arrays['tokens'][start:start + query_batch_size], arrays['point_mask'][start:start + query_batch_size])]
        queries = [geometry_descriptor(value) for value in points]
        valid = [i for i, value in enumerate(queries) if value is not None]
        found = {}
        if valid:
            nearest, distances = _nearest_descriptors(np.stack([queries[i] for i in valid]), descriptors)
            found = {i: (int(training_rows[row]), float(distance)) for i, row, distance in zip(valid, nearest, distances)}
        for i, (query, raw) in enumerate(zip(queries, points)):
            record = metadata['records'][start + i]
            report = {'sample_index': start + i, 'intended_neighborhood_id': record.get('intended_neighborhood_id'),
                      'evaluable': query is not None, 'nearest_training_id': None,
                      'nearest_training_descriptor_distance': None, 'nearest_support_descriptor_distance': None,
                      'exact_support_point_sequence': None, 'exact_training_point_sequence': None}
            if query is not None:
                train_row, distance = found[i]
                nearest_raw = dataset.tokens[train_row, :int(dataset.lengths[train_row]), :3]
                report.update(nearest_training_id=str(dataset.base_ids[train_row]),
                              nearest_training_descriptor_distance=distance,
                              exact_training_point_sequence=bool(np.array_equal(raw, nearest_raw)))
                support_rows = [by_id[item] for item in record.get('support_ids', [])]
                if support_rows:
                    support_descriptors = _geometry_descriptors(dataset, support_rows)
                    report['nearest_support_descriptor_distance'] = float(np.linalg.norm(support_descriptors - query, axis=1).min())
                    report['exact_support_point_sequence'] = any(np.array_equal(
                        raw, dataset.tokens[row, :int(dataset.lengths[row]), :3]) for row in support_rows)
            reports.append(report)
    report = {'dataset_identifier': dataset.identifier, 'manifest_id': metadata.get('manifest_id'),
              'training_population_count': len(training_rows), 'statistics': trajectory_statistics(arrays),
              'descriptor': '16 linearly interpolated raw XY points, length/128, stroke_count/16; no alignment',
              'nearest_training_scope': 'all permitted training examples; no category or query metadata used',
              'near_copy_threshold': None, 'exact_training_check_scope': 'nearest descriptor exemplar only',
              'per_generated': reports}
    _dump(output, report)
    return report


def score_run(generation, feature_root, reference, output, *, reference_repeat=(), seed=0):
    """Score all frozen conditions and paired context effects within one model."""
    from .aggregate import paired_neighborhood_summary
    from .metrics import load_features, score_feature_artifacts

    generation, feature_root = Path(generation), Path(feature_root)
    summary = json.loads((generation / 'summary.json').read_text())
    conditions = tuple(summary['conditions'])
    if 'correct_support' not in conditions:
        raise ValueError('Paired support scoring requires correct_support generations.')
    metadata = {condition: load_trajectories(generation / condition)[1] for condition in conditions}
    intended = metadata['correct_support']
    signature_keys = ('task_index', 'sample_index', 'query_seed', 'intended_category',
                      'intended_neighborhood_id', 'query_ids', 'reference_ids')
    signature = [[row.get(key) for key in signature_keys] for row in intended['records']]
    scores = {}
    for condition, item in metadata.items():
        if any(item.get(key) != intended.get(key) for key in (
                'checkpoint_id', 'evaluation_id', 'manifest_id', 'dataset_identifier', 'model_type')):
            raise ValueError('Control provenance differs from the correct-support run.')
        if any(item.get(key) != summary.get(key) for key in ('checkpoint_id', 'evaluation_id')):
            raise ValueError('Generated controls differ from their run summary.')
        if [[row.get(key) for key in signature_keys] for row in item['records']] != signature:
            raise ValueError('Controls must retain identical intended targets, references, and random keys.')
        _, feature_metadata = load_features(feature_root / condition)
        if (feature_metadata.get('artifact_sha256') != file_sha256(generation / condition / 'trajectories.npz') or
                feature_metadata.get('artifact_metadata_sha256') != file_sha256(generation / condition / 'metadata.json')):
            raise ValueError('Features must correspond exactly to the generated condition artifacts.')
        scores[condition] = score_feature_artifacts(feature_root / condition, reference,
                                                     reference_repeats=reference_repeat)
    # Reference membership comes from the fixed evaluation construction, never
    # from whichever real samples happen to improve a model's score.
    for path in (reference, *reference_repeat):
        _, real = load_features(path)
        half = real.get('reference_half')
        if half not in ('real_a', 'real_b'):
            raise ValueError('Use the frozen evaluation real_a/real_b reference artifacts.')
        expected = {row['intended_neighborhood_id']: set(row['reference_ids'][half])
                    for row in intended['records']}
        actual = {}
        for row in real['records']:
            actual.setdefault(row['intended_neighborhood_id'], set()).add(str(row['drawing_id']))
        if expected != actual:
            raise ValueError('Reference features differ from the intended frozen local populations.')
    comparisons = {}
    correct = scores['correct_support']['neighborhood_conditional']['per_neighborhood']
    for name, condition in (('context_gain', 'no_context'), ('category_specificity', 'wrong_support'),
                            ('within_category_specificity', 'same_category_wrong_neighborhood'),
                            ('support_copy_comparison', 'support_copy'),
                            ('nearest_training_comparison', 'nearest_training_example')):
        if condition not in scores:
            continue
        control = scores[condition]['neighborhood_conditional']['per_neighborhood']
        effect = paired_neighborhood_summary(control, correct, intended['records'], seed=seed)
        effect['positive_difference_favors'] = 'correct_support'
        records = metadata[condition]['records']
        if all('conditional_target_loss' in row for row in records + intended['records']):
            def losses(items):
                return {row['intended_neighborhood_id']: {'sketch_mmd': row['conditional_target_loss']['loss']}
                        for row in items}
            target_effect = paired_neighborhood_summary(losses(records), losses(intended['records']),
                                                        intended['records'], seed=seed)
            target_effect.update(metric='conditional_target_loss', positive_difference_favors='correct_support',
                                 objective=summary['conditions']['correct_support']['conditional_target_loss']['objective'])
            effect['conditional_target_loss'] = target_effect
        comparisons[name] = effect
    result = {'checkpoint_id': intended['checkpoint_id'], 'evaluation_id': intended['evaluation_id'],
              'dataset_identifier': intended['dataset_identifier'], 'architecture': intended['model_type'],
              'feature_scores': scores, 'paired_controls': comparisons,
              'uncertainty': 'overlap-connected neighborhoods within one training run; not independent seed uncertainty',
              'raw_output_statistics': summary['conditions']}
    _dump(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--dataset-root', required=True)
    prepare.add_argument('--embedding-root', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--split', choices=('development', 'test'), default='development')
    prepare.add_argument('--tasks-per-category', type=int, default=4)
    prepare.add_argument('--support-count', type=int, default=4)
    prepare.add_argument('--selection-mode', choices=('exact_top_k', 'sample_top_m'), default='exact_top_k')
    prepare.add_argument('--seed', type=int, default=0)
    prepare.add_argument('--reference-count', type=int, default=8)
    prepare.add_argument('--allow-test', action='store_true')
    prepare.add_argument('--negative-candidates', type=int, default=128)
    prepare.add_argument('--wrong-neighborhood-min-distance', type=float, default=.05)
    prepare.add_argument('--wrong-neighborhood-max-jaccard', type=float, default=.1)
    prepare.add_argument('--no-progress', action='store_false', dest='progress', default=True)
    generation = commands.add_parser('generate')
    generation.add_argument('--checkpoint', required=True)
    generation.add_argument('--episodes', required=True)
    generation.add_argument('--output', required=True)
    generation.add_argument('--dataset-root')
    generation.add_argument('--conditions', nargs='+', choices=CONDITIONS)
    generation.add_argument('--samples-per-task', type=int, default=4)
    generation.add_argument('--batch-size', type=int, default=8)
    generation.add_argument('--seed', type=int, default=0)
    generation.add_argument('--allow-test', action='store_true')
    generation.add_argument('--no-progress', action='store_false', dest='progress', default=True)
    copying = commands.add_parser('geometry-copying')
    copying.add_argument('--dataset-root', required=True)
    copying.add_argument('--generated', required=True)
    copying.add_argument('--output', required=True)
    copying.add_argument('--query-batch-size', type=int, default=64)
    copying.add_argument('--no-progress', action='store_false', dest='progress', default=True)
    scoring = commands.add_parser('score')
    scoring.add_argument('--generation', required=True)
    scoring.add_argument('--feature-root', required=True)
    scoring.add_argument('--reference', required=True)
    scoring.add_argument('--reference-repeat', action='append', default=[])
    scoring.add_argument('--output', required=True)
    scoring.add_argument('--seed', type=int, default=0)
    arguments = vars(parser.parse_args())
    command = arguments.pop('command')
    function = {'prepare': prepare_evaluation, 'generate': generate_run,
                'geometry-copying': geometry_copying, 'score': score_run}[command]
    result = function(**arguments)
    if isinstance(result, Path):
        print(result)


if __name__ == '__main__':
    main()
