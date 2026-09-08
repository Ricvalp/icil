"""Offline cosine-defined tasks, without tensor-framework or index imports.

The policy reads only the resulting trajectory IDs. Complete-target retrieval
is an explicitly declared offline privilege of A-NN, never a policy input.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Iterable

import numpy as np

from .data import (
    DUPLICATE_RULE, SPLITS, SketchStore, _digest, _partition, _rank,
    build_manifest, validate_manifest,
)


PROTOCOLS = ('a_nn', 'a_local', 'a_category')
SPLIT_REGIMES = ('familiar_drawings', 'heldout_regions', 'heldout_categories')
TIE_RULE = 'descending_cosine_then_ascending_base_id'


def cosine_neighbors(
    query: np.ndarray, candidate_ids: Iterable[str], vectors: np.ndarray, *,
    exclude_ids: Iterable[str] = (), duplicate_clusters: dict[str, str] | None = None,
    exclude_clusters: Iterable[str] = (), count: int | None = None,
) -> tuple[list[str], list[float]]:
    """Exact normalized-vector search with stable ties and explicit exclusions."""
    ids = np.asarray(list(candidate_ids), dtype=str)
    values = np.asarray(vectors, np.float32)
    query = np.asarray(query, np.float32)
    if values.ndim != 2 or values.shape[0] != len(ids) or query.shape != (values.shape[1],):
        raise ValueError('Cosine query/candidate shapes do not agree.')
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(query)):
        raise ValueError('Cosine vectors must be finite.')
    if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-5) or not np.isclose(np.linalg.norm(query), 1, atol=2e-5):
        raise ValueError('Cosine retrieval requires separate L2-normalized vectors.')
    excluded, clusters = set(exclude_ids), set(exclude_clusters)
    allowed = np.asarray([item not in excluded and (
        not clusters or duplicate_clusters is not None and duplicate_clusters[item] not in clusters
    ) for item in ids], dtype=bool)
    if clusters and duplicate_clusters is None:
        raise ValueError('Duplicate exclusions require candidate duplicate-cluster metadata.')
    scores = np.asarray(values, np.float64) @ np.asarray(query, np.float64)
    indices = np.flatnonzero(allowed)
    order = np.lexsort((ids[indices], -scores[indices]))
    indices = indices[order[:count]]
    return ids[indices].tolist(), [float(value) for value in scores[indices]]


def _round_robin(groups: dict[str, list[str]]) -> list[str]:
    return [values[index] for index in range(max(map(len, groups.values()), default=0))
            for values in groups.values() if index < len(values)]


def _regions(ids: list[str], vectors: np.ndarray, *, count: int, seed: int,
             category: str, train_fraction: float, development_fraction: float) -> tuple[dict, dict]:
    """Farthest-first spherical Voronoi regions; reserve entire cells."""
    if count < 3 or len(ids) < count:
        raise ValueError('Held-out regions require at least three populated construction regions.')
    order = {item: index for index, item in enumerate(ids)}
    first = order[_rank(ids, seed, 'first_region_center:' + category)[0]]
    selected = [first]
    nearest_similarity = vectors @ vectors[first]
    for _ in range(1, count):
        candidates = np.asarray([index for index in range(len(ids)) if index not in selected])
        ranked = np.lexsort((np.asarray(ids)[candidates], nearest_similarity[candidates]))
        next_index = int(candidates[ranked[0]])
        selected.append(next_index)
        nearest_similarity = np.maximum(nearest_similarity, vectors @ vectors[next_index])
    centers = vectors[selected]
    # [drawing, region]; center order fixes exact equidistance ties.
    assignments = np.argmax(vectors @ centers.T, axis=1)
    groups = {ids[selected[index]]: [item for row, item in enumerate(ids) if assignments[row] == index]
              for index in range(count)}
    groups = {key: values for key, values in groups.items() if values}
    if len(groups) < 3:
        raise ValueError('Embedding geometry yields fewer than three nonempty regions; no random fallback.')
    region_split = _partition(_rank(groups, seed, 'region_split:' + category), train_fraction, development_fraction)
    pools = {split: [item for region in region_split[split] for item in groups[region]] for split in SPLITS}
    metadata = {'rule': 'farthest_first_cosine_voronoi_v1', 'centers': [ids[i] for i in selected],
                'region_splits': region_split, 'members': groups,
                'tie_policy': 'construction_center_order', 'requested_regions': count}
    return pools, metadata


def _task_statistics(store: SketchStore, ids: list[str], vectors: np.ndarray) -> dict:
    center = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(center))
    center = center / norm if norm > 1e-12 else vectors[0]
    geometry, lengths, strokes = [], [], []
    for item in ids:
        record = store.get(item)
        progress = np.linspace(0, 1, record.length)
        geometry.append(np.stack([np.interp(np.linspace(0, 1, 16), progress, record.absolute[:, axis])
                                  for axis in range(2)], axis=-1))
        lengths.append(record.length)
        strokes.append(int(np.count_nonzero(record.incoming_pen == 0)))
    return {'centroid': center.tolist(),
            'feature_dispersion': float(np.maximum(0, 1 - vectors @ center).mean()),
            'geometry_dispersion': float(np.var(np.asarray(geometry), axis=0).mean()),
            'geometry_definition': 'variance_of_16_point_index_interpolated_absolute_xy',
            'mean_length': float(np.mean(lengths)), 'mean_strokes': float(np.mean(strokes)),
            'length_histogram': dict(Counter(map(str, lengths))),
            'stroke_count_histogram': dict(Counter(map(str, strokes)))}


def _overlap(tasks: dict[str, dict], *, limit: int = 256) -> dict:
    """Bound diagnostic quadratic work and disclose deterministic subsampling."""
    selected = sorted(tasks)[:limit]
    overlaps, distances, anchor_distances = [], [], []
    for i, left in enumerate(selected):
        for right in selected[i + 1:]:
            if tasks[left]['category'] != tasks[right]['category']:
                continue
            a, b = set(tasks[left]['member_ids']), set(tasks[right]['member_ids'])
            overlaps.append(len(a & b) / len(a | b))
            distances.append(float(max(0, 1 - np.dot(tasks[left]['centroid'], tasks[right]['centroid']))))
            if 'construction_vector' in tasks[left] and 'construction_vector' in tasks[right]:
                anchor_distances.append(float(max(0, 1 - np.dot(tasks[left]['construction_vector'], tasks[right]['construction_vector']))))
    return {'task_count': len(tasks), 'diagnostic_task_count': len(selected),
            'diagnostic_selection': 'first_256_lexical_task_ids', 'same_category_pairs': len(overlaps),
            'mean_member_jaccard': float(np.mean(overlaps)) if overlaps else None,
            'max_member_jaccard': max(overlaps) if overlaps else None,
            'mean_centroid_cosine_distance': float(np.mean(distances)) if distances else None,
            'mean_anchor_cosine_distance': float(np.mean(anchor_distances)) if anchor_distances else None,
            'identical_member_pairs': sum(value == 1 for value in overlaps)}


def build_neighborhood_manifest(
    store: SketchStore, embeddings, *, a_protocol: str = 'a_nn',
    split_regime: str = 'familiar_drawings', seed: int = 0, subset_seed: int = 0,
    family_count: int | None = None, neighborhood_count: int | None = None,
    evaluation_neighborhood_count: int | None = None,
    neighborhood_size: int = 64, reference_per_neighborhood: int = 8,
    reference_per_category: int = 8, unique_budget: int | None = None,
    drawings_per_category: int | None = None, train_fraction: float = .7,
    development_fraction: float = .15, max_points: int = 255,
    selection_mode: str = 'exact_top_k', top_m: int = 32, region_count: int = 8,
    min_neighbor_cosine: float = -1.0,
    budget_regime: str = 'fixed_reservoir',
) -> dict:
    """Split duplicates first, then curate fixed split-local distributions.

    A-local construction anchors and real references never enter model-example
    pools. The training eligible reservoir and all evaluation tasks are fixed
    as N changes. N selects nested category-balanced local tasks; A-NN records
    explicitly that the same operation also reduces distinct training targets.
    """
    if a_protocol not in PROTOCOLS or split_regime not in SPLIT_REGIMES:
        raise ValueError('Unknown A protocol or policy split regime.')
    if budget_regime not in ('fixed_reservoir', 'per_neighborhood'):
        raise ValueError('Select fixed_reservoir or per_neighborhood budget_regime.')
    if budget_regime == 'per_neighborhood' and (a_protocol != 'a_local' or unique_budget is not None
                                               or drawings_per_category is not None):
        raise ValueError('Per-neighborhood allowance requires A-local and no global/category drawing budget.')
    if selection_mode not in ('exact_top_k', 'sample_top_m'):
        raise ValueError('Select exact_top_k or sample_top_m.')
    if neighborhood_size < 2 or top_m < 1 or region_count < 3:
        raise ValueError('Require neighborhood_size >= 2, top_m >= 1, and region_count >= 3.')
    if reference_per_neighborhood < 2 or reference_per_neighborhood % 2:
        raise ValueError('Neighborhood reference count must be positive and even, at least two.')
    if not -1 <= min_neighbor_cosine <= 1:
        raise ValueError('Minimum neighbor cosine must lie in [-1,1].')
    if neighborhood_count is not None and neighborhood_count < 1:
        raise ValueError('neighborhood_count must be positive.')
    if evaluation_neighborhood_count is not None and evaluation_neighborhood_count < 1:
        raise ValueError('evaluation_neighborhood_count must be positive.')
    if embeddings.cache_id != store.identifier:
        raise ValueError('Embeddings were exported from a different policy cache.')
    row_by_id = {record['base_id']: index for index, record in enumerate(embeddings.records)}
    if len(row_by_id) != len(embeddings.records):
        raise ValueError('Embedding IDs must be unique.')
    if set(row_by_id) != {record.base_id for record in store.records}:
        raise ValueError('Frozen embeddings must cover exactly the source policy cache.')
    for record in store.records:
        if record.base_id not in row_by_id:
            raise ValueError('Policy drawing is missing its frozen embedding.')
        metadata = embeddings.records[row_by_id[record.base_id]]
        if metadata['category'] != record.category or metadata['duplicate_cluster_id'] != record.duplicate_cluster_id:
            raise ValueError('Embedding record metadata differs from the trajectory cache.')
    def vectors(ids):
        return np.asarray(embeddings.cosine[[row_by_id[item] for item in ids]], np.float32)

    # Preserve B's existing category/program split contract as a separate axis.
    manifest = build_manifest(store, seed=seed, subset_seed=subset_seed,
                              train_fraction=train_fraction, development_fraction=development_fraction,
                              family_count=family_count, reference_per_category=reference_per_category,
                              max_points=max_points)
    manifest['b_category_splits'] = deepcopy(manifest['categories'])
    manifest['b_reference_ids'] = deepcopy(manifest['reference_ids'])
    manifest['b_heldout_category_ids'] = deepcopy(manifest['a_ids'])
    clusters = defaultdict(list)
    for record in store.records:
        if record.length <= max_points:
            clusters[record.duplicate_cluster_id].append(record)
    category_ids = defaultdict(list)
    for records in clusters.values():
        if len({record.category for record in records}) == 1:
            representative = min(records, key=lambda record: record.base_id)
            category_ids[representative.category].append(representative.base_id)
    # A subset seed selects local tasks only; coarse-category coverage and the
    # available drawing reservoir remain fixed across subset replications.
    category_order = _rank(manifest['b_category_splits']['train'], seed, 'fixed_coarse_categories')
    selected_categories = category_order[:family_count] if family_count else category_order
    categories = (deepcopy(manifest['categories']) if split_regime == 'heldout_categories'
                  else {split: list(selected_categories) for split in SPLITS})
    categories['train'] = list(selected_categories)
    policy_splits = {split: {} for split in SPLITS}
    region_metadata = {}
    if split_regime == 'heldout_categories':
        for split in SPLITS:
            policy_splits[split] = {category: category_ids[category] for category in categories[split]}
    else:
        for category in selected_categories:
            ids = sorted(category_ids[category])
            if split_regime == 'familiar_drawings':
                pools = _partition(_rank(ids, seed, 'policy_drawing_split'), train_fraction, development_fraction)
            else:
                pools, region_metadata[category] = _regions(
                    ids, vectors(ids), count=region_count, seed=seed, category=category,
                    train_fraction=train_fraction, development_fraction=development_fraction)
            for split in SPLITS:
                policy_splits[split][category] = pools[split]
    if unique_budget is not None and drawings_per_category is not None:
        raise ValueError('Choose unique_budget or drawings_per_category, not both.')
    if unique_budget is not None and unique_budget < len(selected_categories):
        raise ValueError('Unique drawing budget must cover every fixed training category.')
    a_ids, split_pools, tasks, references = ({split: {} for split in SPLITS} for _ in range(4))
    rejected = Counter()
    attempted = Counter()
    construction_anchors = {split: {} for split in SPLITS}
    for split in SPLITS:
        cap = neighborhood_count if split == 'train' else evaluation_neighborhood_count
        if cap is not None and (cap < len(categories[split]) or cap % len(categories[split])):
            raise ValueError('Neighborhood cap must be a positive multiple of its fixed category count.')
        per_category_cap = None if cap is None else cap // len(categories[split])
        for category_index, category in enumerate(categories[split]):
            ids = _rank(policy_splits[split][category], seed, 'role_split:' + split)
            # Reserve a fixed, split-local reference reservoir before tasks.
            reference_count = max(reference_per_category, reference_per_neighborhood, int(len(ids) * .15))
            anchor_count = max(1, int(len(ids) * .15)) if a_protocol == 'a_local' else 0
            reference_pool = ids[:reference_count]
            anchors = ids[reference_count:reference_count + anchor_count]
            examples = ids[reference_count + anchor_count:]
            if len(reference_pool) < reference_per_neighborhood or not examples:
                raise ValueError(f'{category}/{split} has insufficient split-local references and examples.')
            if split == 'train':
                requested = (unique_budget // len(selected_categories) + int(category_index < unique_budget % len(selected_categories))
                             if unique_budget is not None else drawings_per_category)
                if requested is not None:
                    if requested < 1 or requested > len(examples):
                        raise ValueError(f'Insufficient eligible drawings for requested budget in {category}/{split}.')
                    examples = _rank(examples, seed, 'fixed_eligible_reservoir')[:requested]
            a_ids[split][category] = examples
            construction_anchors[split][category] = anchors
            split_pools[split][category] = {'example_ids': examples, 'reference_ids': reference_pool, 'anchor_ids': anchors}
            references[split][category] = {'real_a': reference_pool[::2], 'real_b': reference_pool[1::2]}
            candidates = anchors if a_protocol == 'a_local' else examples if a_protocol == 'a_nn' else [category]
            def task_identity(anchor):
                return (category if a_protocol == 'a_category' else
                        a_protocol + '/' + _digest([split, category, anchor])[:24])
            candidate_by_task = {task_identity(anchor): anchor for anchor in candidates}
            candidates = [candidate_by_task[task_id] for task_id in _rank(
                candidate_by_task, subset_seed if split == 'train' else seed,
                'nested_neighborhoods' if split == 'train' else 'fixed_evaluation_neighborhoods')]
            example_vectors, reference_vectors = vectors(examples), vectors(reference_pool)
            accepted = 0
            for anchor in candidates:
                if per_category_cap is not None and accepted >= per_category_cap:
                    break
                attempted[split] += 1
                if a_protocol == 'a_category':
                    member_ids, scores = examples, []
                    selected_references = (reference_pool[::2][:reference_per_neighborhood // 2]
                                           + reference_pool[1::2][:reference_per_neighborhood // 2])
                else:
                    desired = neighborhood_size if a_protocol == 'a_local' else top_m
                    member_ids, scores = cosine_neighbors(
                        vectors([anchor])[0], examples, example_vectors, exclude_ids=[anchor],
                        duplicate_clusters={item: store.get(item).duplicate_cluster_id for item in examples},
                        exclude_clusters=[store.get(anchor).duplicate_cluster_id], count=desired)
                    if len(member_ids) != desired:
                        rejected[split + ':undersized'] += 1
                        continue
                    if min(scores) < min_neighbor_cosine:
                        rejected[split + ':below_minimum_cosine'] += 1
                        continue
                    selected_references = []
                    for offset in (0, 1):
                        half, _ = cosine_neighbors(vectors([anchor])[0], reference_pool[offset::2],
                                                   reference_vectors[offset::2], count=reference_per_neighborhood // 2)
                        selected_references.extend(half)
                task_id = task_identity(anchor)
                neighbor_ids = member_ids if a_protocol == 'a_nn' else []
                members = member_ids + [anchor] if a_protocol == 'a_nn' else member_ids
                task = {'task_id': task_id, 'category': category, 'split': split,
                        'anchor_id': anchor if a_protocol == 'a_local' else None,
                        'target_id': anchor if a_protocol == 'a_nn' else None,
                        'member_ids': members, 'neighbor_ids': neighbor_ids, 'neighbor_scores': scores,
                        'reference_ids': {'real_a': selected_references[:reference_per_neighborhood // 2],
                                          'real_b': selected_references[reference_per_neighborhood // 2:]},
                        'reference_role': 'offline_target_local_distribution_proxy' if a_protocol == 'a_nn' else 'reserved_local_distribution',
                        'duplicate_clusters': {item: store.get(item).duplicate_cluster_id for item in members},
                        **_task_statistics(store, members, vectors(members))}
                if a_protocol != 'a_category':
                    task['construction_vector'] = vectors([anchor])[0].tolist()
                tasks[split][task_id] = task
                accepted += 1
            if per_category_cap is not None and accepted < per_category_cap:
                raise ValueError(f'{category}/{split} has only {accepted} eligible tasks; requested {per_category_cap}.')
            if not any(task['category'] == category for task in tasks[split].values()):
                raise ValueError(f'No eligible {a_protocol} tasks in {category}/{split}; reduce declared size or add data.')
    available_tasks = deepcopy(tasks['train'])
    ranked_tasks = {category: _rank([key for key, task in available_tasks.items() if task['category'] == category],
                                   subset_seed, 'nested_neighborhoods') for category in selected_categories}
    task_order = _round_robin(ranked_tasks)
    if neighborhood_count is not None:
        if neighborhood_count < len(selected_categories) or neighborhood_count > len(task_order):
            raise ValueError(f'N must cover the {len(selected_categories)} fixed categories and cannot exceed {len(task_order)} eligible tasks.')
        # Exact balance makes category frequencies invariant along the N sweep.
        if neighborhood_count % len(selected_categories):
            raise ValueError('N must be a multiple of F for an exactly category-balanced sweep.')
        selected_ids = task_order[:neighborhood_count]
        expected = neighborhood_count // len(selected_categories)
        if any(sum(available_tasks[item]['category'] == category for item in selected_ids) != expected for category in selected_categories):
            raise ValueError('Some categories cannot supply the requested balanced neighborhood count.')
        tasks['train'] = {key: available_tasks[key] for key in selected_ids}
    # In the allowance track, construction still sees the same full split-local
    # reservoir at every N. Only selected task members are available to learning.
    # Overlap makes this union smaller than the N * neighborhood_size incidence.
    construction_candidate_ids = deepcopy(a_ids)
    if budget_regime == 'per_neighborhood':
        for category, candidates in construction_candidate_ids['train'].items():
            selected_members = {item for task in tasks['train'].values() if task['category'] == category
                                for item in task['member_ids']}
            examples = [item for item in candidates if item in selected_members]
            a_ids['train'][category] = examples
            split_pools['train'][category]['example_ids'] = examples
    # Existing consumers can still request pooled category references.
    pooled_references = {}
    for split in SPLITS:
        for category, halves in references[split].items():
            pooled = pooled_references.setdefault(category, {'real_a': [], 'real_b': []})
            for half in pooled:
                pooled[half].extend(halves[half])
    feature_manifest = deepcopy(embeddings.manifest)
    manifest.update({
        'neighborhood_schema_version': 'quickdraw-neighborhoods-v1',
        'a_protocol': a_protocol, 'split_regime': split_regime,
        'categories': categories, 'category_splits': deepcopy(categories),
        'nested_category_order': category_order,
        'a_ids': a_ids, 'a_tasks': tasks, 'split_pools': split_pools,
        'construction_candidate_ids': construction_candidate_ids,
        'policy_split_ids': policy_splits, 'construction_anchors': construction_anchors,
        'reference_ids': pooled_references, 'split_reference_ids': references,
        'region_construction': region_metadata, 'nested_neighborhood_order': task_order,
        'retrieval': {'backend': 'numpy_exact_cosine_v1', 'selection_mode': selection_mode,
                      'top_m': top_m, 'neighborhood_size': neighborhood_size,
                      'tie_policy': TIE_RULE, 'exclusion_mode': 'scientific_duplicate_cluster',
                      'min_neighbor_cosine': min_neighbor_cosine,
                      'rejection_rule': 'reject_then_next_anchor_in_fixed_pool_no_radius_expansion'},
        'curation': {'embedding_identifier': embeddings.identifier,
                     'index_identifier': _digest(['numpy_exact_cosine_v1', embeddings.identifier, a_ids]),
                     'index_version': 'numpy_exact_cosine_v1', 'feature_manifest': feature_manifest,
                     'retrieval_feature_convention': 'L2_normalized_frozen_classifier_penultimate_features',
                     'metric_feature_convention': 'raw_unnormalized_frozen_classifier_penultimate_features',
                     'classifier_role': 'fixed_offline_task_construction_and_evaluation',
                     'policy_inputs_include_classifier_features': False,
                     'all_exported_drawings': len(embeddings.records),
                     'construction_example_candidates': {
                         split: {category: len(ids) for category, ids in pools.items()}
                         for split, pools in construction_candidate_ids.items()},
                     'all_exported_categories': len({record['category'] for record in embeddings.records})},
        'construction_diagnostics': {'attempted_tasks': dict(attempted), 'rejected_tasks': dict(rejected),
                                     'rejection_rates': {split: sum(count for reason, count in rejected.items()
                                                                   if reason.startswith(split + ':')) / attempted[split]
                                                         for split in SPLITS},
                                     'overlap': {split: _overlap(tasks[split]) for split in SPLITS}},
    })
    proximity = {}
    for split in ('development', 'test'):
        values = []
        for category in set(a_ids['train']) & set(a_ids[split]):
            # Fixed bounded sample; regions may touch despite disjoint IDs.
            train = vectors(sorted(a_ids['train'][category])[:256])
            heldout = vectors(sorted(a_ids[split][category])[:256])
            values.extend(np.max(heldout @ train.T, axis=1).tolist())
        proximity[split] = {'sample_rule': 'first_256_lexical_ids_per_category_per_split',
                            'heldout_count': len(values),
                            'mean_nearest_train_cosine': float(np.mean(values)) if values else None,
                            'max_nearest_train_cosine': max(values) if values else None}
    manifest['construction_diagnostics']['train_heldout_proximity'] = proximity
    manifest['budget'].update({
        'family_count': len(selected_categories), 'fixed_training_categories': selected_categories,
        'evaluation_neighborhood_count': evaluation_neighborhood_count,
        'neighborhood_count': len(tasks['train']),
        'available_neighborhood_count': len(available_tasks) if neighborhood_count is None else None,
        'constructed_neighborhood_count': len(available_tasks),
        'construction_candidate_count': sum(len(pool['anchor_ids'] if a_protocol == 'a_local' else pool['example_ids'])
                                             for pool in split_pools['train'].values()),
        'unique_a_train_drawings': sum(map(len, a_ids['train'].values())),
        'unique_budget': unique_budget, 'drawings_per_category': drawings_per_category,
        'budget_regime': budget_regime,
        'regime': 'fixed_eligible_drawing_reservoir' if budget_regime == 'fixed_reservoir' else 'diversity_plus_data',
        'per_neighborhood_drawing_allowance': neighborhood_size if a_protocol == 'a_local' else None,
        'train_neighborhood_member_incidences': sum(len(task['member_ids']) for task in tasks['train'].values()),
        'construction_example_candidate_count': sum(map(len, construction_candidate_ids['train'].values())),
        'N_also_limits_training_targets': a_protocol == 'a_nn',
        'unique_eligible_targets': len(tasks['train']) if a_protocol == 'a_nn' else
        len({item for task in tasks['train'].values() for item in task['member_ids']}),
        'reference_drawings': sum(len(pool['reference_ids']) for split in split_pools.values() for pool in split.values()),
        'construction_anchors': sum(len(pool['anchor_ids']) for split in split_pools.values() for pool in split.values()),
        'empirical_task_category_probability': dict(Counter(task['category'] for task in tasks['train'].values())),
    })
    total = len(tasks['train'])
    manifest['budget']['empirical_task_category_probability'] = {
        category: count / total for category, count in manifest['budget']['empirical_task_category_probability'].items()}
    manifest['reservoir_identifiers']['a'] = {split: _digest(a_ids[split]) for split in SPLITS}
    manifest['identifier'] = _digest({key: value for key, value in manifest.items() if key != 'identifier'})
    validate_manifest(store, manifest)
    return manifest


def validate_neighborhood_manifest(store: SketchStore, manifest: dict) -> None:
    """Validate metadata privilege and duplicate/split boundaries from records."""
    protocol, regime = manifest['a_protocol'], manifest['split_regime']
    if protocol not in PROTOCOLS or regime not in SPLIT_REGIMES:
        raise ValueError('Unknown neighborhood protocol/split regime.')
    categories = manifest['categories']
    if any(not categories[split] for split in SPLITS):
        raise ValueError('All policy splits must have explicit categories.')
    if regime == 'heldout_categories':
        flat = [category for split in SPLITS for category in categories[split]]
        if len(flat) != len(set(flat)):
            raise ValueError('Held-out categories overlap.')
    elif any(set(categories[split]) != set(categories['train']) for split in SPLITS):
        raise ValueError('Familiar-category regimes must retain the exact category set.')
    split_clusters = set()
    for split in SPLITS:
        if set(manifest['policy_split_ids'][split]) != set(categories[split]):
            raise ValueError('Declared drawing splits differ from category membership.')
        for category, ids in manifest['policy_split_ids'][split].items():
            for item in ids:
                record = store.get(item)
                if record.category != category or record.duplicate_cluster_id in split_clusters:
                    raise ValueError('Policy drawing split or duplicate cluster overlaps.')
                split_clusters.add(record.duplicate_cluster_id)
    seen_clusters = set()
    for split in SPLITS:
        if set(manifest['a_ids'][split]) != set(categories[split]):
            raise ValueError('Policy reservoir categories differ from split categories.')
        if set(manifest['split_pools'][split]) != set(categories[split]):
            raise ValueError('Split-local role categories differ from split categories.')
        for category, pool in manifest['split_pools'][split].items():
            candidates = manifest.get('construction_candidate_ids', manifest['a_ids'])[split][category]
            if (len(candidates) != len(set(candidates))
                    or not set(candidates) <= set(manifest['policy_split_ids'][split][category])
                    or set(candidates) & set(pool['reference_ids'] + pool['anchor_ids'])
                    or not set(pool['example_ids']) <= set(candidates)):
                raise ValueError('Construction candidates cross split/role boundaries or omit available examples.')
            if pool['example_ids'] != manifest['a_ids'][split][category] or not pool['example_ids']:
                raise ValueError('Declared split pool does not match model-example reservoir.')
            for role in ('example_ids', 'reference_ids', 'anchor_ids'):
                for item in pool[role]:
                    record = store.get(item)
                    if record.category != category or record.duplicate_cluster_id in seen_clusters:
                        raise ValueError('Policy examples, references, anchors or duplicate clusters overlap across roles/splits.')
                    if item not in manifest['policy_split_ids'][split][category]:
                        raise ValueError('Role membership crosses the declared policy split.')
                    seen_clusters.add(record.duplicate_cluster_id)
        if not manifest['a_tasks'][split]:
            raise ValueError('Each split requires eligible declared tasks.')
        for task_id, task in manifest['a_tasks'][split].items():
            category = task['category']
            if task_id != task['task_id'] or task['split'] != split or category not in categories[split]:
                raise ValueError('Task identity/category/split mismatch.')
            pool = manifest['split_pools'][split][category]
            members = task['member_ids']
            if len(members) != len(set(members)) or not set(members) <= set(pool['example_ids']):
                raise ValueError('Task members must be distinct split-local eligible model examples.')
            references = task['reference_ids']['real_a'] + task['reference_ids']['real_b']
            if len(set(references)) != len(references) or not set(references) <= set(pool['reference_ids']):
                raise ValueError('Task references must be distinct reserved split-local drawings.')
            for half in ('real_a', 'real_b'):
                if not set(task['reference_ids'][half]) <= set(manifest['split_reference_ids'][split][category][half]):
                    raise ValueError('Local reference half crosses the globally fixed real-real split.')
            if protocol == 'a_local' and task['anchor_id'] not in pool['anchor_ids']:
                raise ValueError('A-local requires a reserved, example-excluded construction anchor.')
            if protocol == 'a_nn':
                if task['target_id'] not in members or task['target_id'] in task['neighbor_ids']:
                    raise ValueError('A-NN target cannot occur among support neighbors.')
                if set(task['neighbor_ids']) != set(members) - {task['target_id']}:
                    raise ValueError('A-NN candidate/member mapping differs.')
                scores = np.asarray(task['neighbor_scores'], np.float64)
                if (scores.shape != (len(task['neighbor_ids']),) or not np.all(np.isfinite(scores))
                        or np.any(np.abs(scores) > 1 + 2e-5)):
                    raise ValueError('Neighbor scores must be finite cosines aligned with neighbor IDs.')
                pairs = list(zip(task['neighbor_scores'], task['neighbor_ids']))
                if pairs != sorted(pairs, key=lambda pair: (-pair[0], pair[1])):
                    raise ValueError('Neighbor ranks violate the declared score/base-ID tie policy.')
            if task['duplicate_clusters'] != {item: store.get(item).duplicate_cluster_id for item in members}:
                raise ValueError('Task duplicate-cluster metadata mismatch.')
    if manifest['budget'].get('budget_regime') == 'per_neighborhood':
        if protocol != 'a_local':
            raise ValueError('Per-neighborhood allowance is defined only for A-local.')
        for category, ids in manifest['a_ids']['train'].items():
            members = {item for task in manifest['a_tasks']['train'].values() if task['category'] == category
                       for item in task['member_ids']}
            if set(ids) != members:
                raise ValueError('Per-neighborhood training availability must equal the selected member union.')
        if any(len(task['member_ids']) != manifest['budget']['per_neighborhood_drawing_allowance']
               for task in manifest['a_tasks']['train'].values()):
            raise ValueError('Tasks violate the fixed per-neighborhood data allowance.')
    if regime == 'heldout_regions':
        for category, regions in manifest['region_construction'].items():
            for split in SPLITS:
                expected = {item for region in regions['region_splits'][split] for item in regions['members'][region]}
                if expected != set(manifest['policy_split_ids'][split][category]):
                    raise ValueError('Held-out region membership crosses a split boundary.')
    # B uses its original independent category and program experiment pools.
    b_seen = set()
    b_references = {store.get(item).duplicate_cluster_id for halves in manifest['b_reference_ids'].values()
                    for ids in halves.values() for item in ids}
    for split in SPLITS:
        for category, ids in manifest['b_ids'][split].items():
            if category not in manifest['b_category_splits']['train'] or not ids:
                raise ValueError('B program reservoir category mismatch.')
            for item in ids:
                record = store.get(item)
                if record.category != category or record.duplicate_cluster_id in b_seen | b_references:
                    raise ValueError('B program duplicate split/reference overlap.')
                b_seen.add(record.duplicate_cluster_id)
