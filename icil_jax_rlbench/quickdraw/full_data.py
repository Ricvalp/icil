"""Full-category supervised sketch data with compact offline neighbor tables.

Build in the metrics interpreter with FAISS. Training imports only NumPy and
reads immutable memory maps; classifier features and identities are metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import file_sha256


VERSION = 'quickdraw_full_supervised_nn_v1'
SPLITS = ('train', 'development', 'test')
EXAMPLE, REFERENCE, EXCLUDED = 0, 1, 2
ARRAY_NAMES = ('tokens', 'lengths', 'category_ids', 'duplicate_ids', 'base_ids',
               'split_codes', 'roles', 'neighbors', 'neighbor_scores')


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _identifier(value) -> str:
    return hashlib.sha256(_canonical({key: item for key, item in value.items()
                                     if key != 'identifier'})).hexdigest()


def _split_name(split: str) -> str:
    split = 'development' if split == 'validation' else split
    if split not in SPLITS:
        raise ValueError(f'Unknown dataset split: {split}')
    return split


def _assign_pools(category_ids, duplicate_ids, categories, clusters, *, seed,
                  train_fraction, development_fraction, reference_fraction, top_m):
    """Partition whole duplicate groups before reserving references or retrieval."""
    count = len(category_ids)
    split_codes = np.full(count, -1, np.int8)
    roles = np.full(count, EXCLUDED, np.uint8)
    cluster_categories = defaultdict(set)
    for category, duplicate in zip(category_ids, duplicate_ids):
        cluster_categories[int(duplicate)].add(int(category))
    cross_category = {cluster for cluster, values in cluster_categories.items() if len(values) > 1}
    excluded_count = 0
    counts = {}
    for category, name in enumerate(categories):
        rows = np.flatnonzero(category_ids == category)
        groups = defaultdict(list)
        for row in rows:
            duplicate = int(duplicate_ids[row])
            if duplicate in cross_category:
                excluded_count += 1
            else:
                groups[duplicate].append(int(row))
        # Category-independent hashes are stable under source order changes.
        ordered = sorted(groups, key=lambda value: (
            hashlib.sha256(f'{seed}:split:{clusters[value]}'.encode()).digest(), clusters[value]))
        train_end = int(len(ordered) * train_fraction)
        development_end = train_end + int(len(ordered) * development_fraction)
        partitions = (ordered[:train_end], ordered[train_end:development_end], ordered[development_end:])
        counts[name] = {}
        for code, (split, partition) in enumerate(zip(SPLITS, partitions)):
            reserve_count = max(2, int(len(partition) * reference_fraction)) if reference_fraction else 0
            if len(partition) - reserve_count < top_m + 1:
                raise ValueError(f'{name!r} {split} has {len(partition)} duplicate groups: need '
                                 f'{top_m + 1} example groups plus {reserve_count} reference groups. '
                                 'Reduce top_m or use more drawings; categories are never silently dropped.')
            reference_order = sorted(partition, key=lambda value: (
                hashlib.sha256(f'{seed}:reference:{clusters[value]}'.encode()).digest(), clusters[value]))
            references = set(reference_order[:reserve_count])
            row_counts = Counter()
            for duplicate in partition:
                role = REFERENCE if duplicate in references else EXAMPLE
                group_rows = groups[duplicate]
                split_codes[group_rows], roles[group_rows] = code, role
                row_counts[role] += len(group_rows)
            counts[name][split] = {'examples': row_counts[EXAMPLE], 'references': row_counts[REFERENCE],
                                   'example_duplicate_groups': len(partition) - reserve_count,
                                   'reference_duplicate_groups': reserve_count}
    return split_codes, roles, counts, {'cross_category_duplicate_groups': len(cross_category),
                                       'excluded_cross_category_drawings': excluded_count}


def _aligned_sources(cache_root: Path, embedding_root: Path, progress: bool):
    """One metadata index plus streamed alignment; no million-record object graph."""
    from .classifier_data import _progress, _status
    from .embeddings import EMBEDDING_VERSION

    _status('Verifying source cache and frozen embedding hashes ...', progress)
    feature_manifest = json.loads((embedding_root / 'manifest.json').read_text())
    if (feature_manifest.get('embedding_version') != EMBEDDING_VERSION
            or feature_manifest.get('identifier') != _identifier(feature_manifest)):
        raise ValueError('Embedding manifest version/hash mismatch.')
    if set(feature_manifest.get('files', {})) != {'raw.npy', 'cosine.npy', 'records.jsonl'}:
        raise ValueError('Embedding manifest has incomplete files.')
    if set(feature_manifest.get('source_cache_files', {})) != {'index.json', 'records.npz'}:
        raise ValueError('Embedding export must attest both source cache files.')
    for name, digest in feature_manifest['files'].items():
        if file_sha256(embedding_root / name) != digest:
            raise ValueError(f'Embedding file hash mismatch: {name}')
    for name, digest in feature_manifest['source_cache_files'].items():
        if file_sha256(cache_root / name) != digest:
            raise ValueError(f'Source cache file hash mismatch: {name}')
    cache = json.loads((cache_root / 'index.json').read_text())
    if cache.get('schema_version') != 'quickdraw-v1' or cache.get('identifier') != feature_manifest.get('cache_identifier'):
        raise ValueError('Embedding and vector cache identifiers/schemas do not match.')
    records = cache['records']
    count = len(records)
    if count < 1 or count >= np.iinfo(np.int32).max or feature_manifest.get('actual_count') != count:
        raise ValueError('Embedding row count must match the nonempty int32-addressable vector cache.')
    previous_id = None
    with (embedding_root / 'records.jsonl').open() as handle:
        for row, metadata in _progress(enumerate(records), 'Checking drawing/embedding alignment',
                                       total=count, unit='drawings', enabled=progress):
            line = handle.readline()
            if not line:
                raise ValueError('Embedding metadata ended before the vector cache.')
            feature = json.loads(line)
            base_id = metadata.get('base_id')
            if not isinstance(base_id, str) or not base_id or (previous_id is not None and base_id <= previous_id):
                raise ValueError('Source vector IDs must be sorted and globally unique.')
            previous_id = base_id
            if feature.get('row_index') != row or any(feature.get(key) != metadata.get(key)
                                                     for key in ('base_id', 'category', 'duplicate_cluster_id')):
                raise ValueError(f'Drawing/embedding metadata row alignment mismatch at row {row}.')
            if not metadata.get('category') or not metadata.get('duplicate_cluster_id'):
                raise ValueError('Source records need categories and duplicate cluster identities.')
        if handle.readline():
            raise ValueError('Embedding metadata has extra rows.')
    raw = np.load(embedding_root / 'raw.npy', mmap_mode='r', allow_pickle=False)
    cosine = np.load(embedding_root / 'cosine.npy', mmap_mode='r', allow_pickle=False)
    if (raw.shape != (count, 512) or cosine.shape != raw.shape
            or raw.dtype != np.float32 or cosine.dtype != np.float32
            or feature_manifest.get('raw_normalization') != 'none'
            or feature_manifest.get('cosine_normalization') != 'l2'):
        raise ValueError('Expected aligned raw and cosine float32 [drawings,512] features.')
    for start in _progress(range(0, count, 8192), 'Validating normalized embeddings',
                           unit='batches', enabled=progress):
        values = raw[start:start + 8192]
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.all(np.isfinite(values)) or np.any(norms <= 1e-12):
            raise ValueError('Frozen features must be finite and nonzero.')
        if not np.allclose(cosine[start:start + 8192], values / norms, rtol=3e-6, atol=3e-7):
            raise ValueError('Cosine features do not normalize the recorded raw features.')
    return cache, feature_manifest, cosine


def _rank_distinct(index, vectors, rows, duplicate_ids, top_m, batch_size):
    """Exact flat-IP candidates with float64 reranking and stable ID ties.

    FAISS computes float32 dot products. A conservative 1e-4 boundary margin
    covers 512D accumulation roundoff for unit vectors; ambiguous boundaries
    expand until no omitted candidate can enter the exact float64 top M.
    """
    count = len(rows)
    initial = min(count, top_m + 17)
    # Global rows follow base-ID lexicographic order, which supplies tie breaks.
    for start in range(0, count, batch_size):
        queries = vectors[start:start + batch_size]
        distances, positions = index.search(queries, initial)
        result = np.empty((len(queries), top_m), np.int32)
        result_scores = np.empty((len(queries), top_m), np.float64)
        for offset, query in enumerate(queries):
            query_row = int(rows[start + offset])
            width = initial
            found, boundary = positions[offset], float(distances[offset, -1])
            while True:
                # Fixed 512-element row reductions give identical vectors exactly
                # equal scores independent of BLAS kernels and candidate count.
                scores = np.sum(np.asarray(vectors[found], np.float64) * np.asarray(query, np.float64), axis=1)
                order = np.lexsort((rows[found], -scores))
                used = {int(duplicate_ids[query_row])}
                selected, selected_scores = [], []
                for candidate in order:
                    row = int(rows[found[candidate]])
                    duplicate = int(duplicate_ids[row])
                    if duplicate not in used:
                        selected.append(row)
                        selected_scores.append(float(scores[candidate]))
                        used.add(duplicate)
                        if len(selected) == top_m:
                            break
                complete = len(selected) == top_m
                if complete and (width == count or selected_scores[-1] > boundary + 1e-4):
                    result[offset], result_scores[offset] = selected, selected_scores
                    break
                if width == count:
                    raise ValueError('Insufficient distinct duplicate groups for the requested neighbor table.')
                width = min(count, width * 2)
                expanded_distances, expanded_positions = index.search(query[None], width)
                found, boundary = expanded_positions[0], float(expanded_distances[0, -1])
        yield start, result, result_scores


def build_full_dataset(
    cache_root: str | Path, embedding_root: str | Path, output: str | Path, *,
    seed: int = 0, top_m: int = 32, max_steps: int = 128,
    train_fraction: float = .7, development_fraction: float = .15,
    reference_fraction: float = .15, batch_size: int = 256, threads: int = 4,
    progress: bool = True,
) -> Path:
    """Build every category's complete familiar-drawing task population once."""
    from .classifier_data import _progress, _status
    import faiss

    if (top_m < 1 or max_steps < 2 or batch_size < 1 or threads < 1
            or not 0 < train_fraction < 1 or not 0 < development_fraction < 1
            or train_fraction + development_fraction >= 1 or not 0 <= reference_fraction < 1):
        raise ValueError('Invalid full-dataset sizes, fractions, or thread count.')
    cache_root, embedding_root, root = Path(cache_root), Path(embedding_root), Path(output)
    if root.exists():
        raise FileExistsError(f'Full-dataset outputs are immutable: {root}')
    cache, feature_manifest, cosine = _aligned_sources(cache_root, embedding_root, progress)
    records = cache.pop('records')
    count = len(records)
    categories = tuple(sorted({record['category'] for record in records}))
    category_lookup = {name: row for row, name in enumerate(categories)}
    clusters = tuple(sorted({record['duplicate_cluster_id'] for record in records}))
    cluster_lookup = {name: row for row, name in enumerate(clusters)}
    category_ids = np.fromiter((category_lookup[record['category']] for record in records), np.int32, count)
    duplicate_ids = np.fromiter((cluster_lookup[record['duplicate_cluster_id']] for record in records), np.int32, count)
    base_ids = np.asarray([record['base_id'] for record in records])
    _status('Splitting whole duplicate groups and reserving reference drawings ...', progress)
    split_codes, roles, category_counts, exclusions = _assign_pools(
        category_ids, duplicate_ids, categories, clusters, seed=seed,
        train_fraction=train_fraction, development_fraction=development_fraction,
        reference_fraction=reference_fraction, top_m=top_m)
    del records, category_lookup, cluster_lookup, clusters
    _status('Loading ragged vector arrays and writing fixed-length policy tokens ...', progress)
    with np.load(cache_root / 'records.npz', allow_pickle=False) as archive:
        offsets = archive['offsets']
        absolute, incoming_pen = archive['absolute'], archive['incoming_pen']
    lengths = np.diff(offsets)
    if (offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(absolute)
            or absolute.shape != (offsets[-1], 2) or incoming_pen.shape != (offsets[-1],)
            or np.any(lengths < 1) or np.any(lengths >= max_steps)):
        raise ValueError('Ragged vectors are invalid or a drawing plus STOP exceeds max_steps; no truncation is allowed.')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'indexes').mkdir()
    tokens = np.lib.format.open_memmap(root / 'tokens.npy', mode='w+', dtype=np.float32,
                                      shape=(count, max_steps, 4))
    tokens[:] = 0
    for start in _progress(range(0, count, 4096), 'Writing policy tokens', unit='batches', enabled=progress):
        end = min(start + 4096, count)
        low, high = int(offsets[start]), int(offsets[end])
        xy, pen = absolute[low:high], incoming_pen[low:high]
        if (not np.all(np.isfinite(xy)) or not np.all((pen == 0) | (pen == 1))
                or np.any(incoming_pen[offsets[start:end]] != 0)):
            raise ValueError('Coordinates must be finite and incoming pen flags binary with first point pen up.')
        for row in range(start, end):
            length = int(lengths[row])
            tokens[row, :length, :2] = absolute[offsets[row]:offsets[row + 1]]
            tokens[row, :length, 2] = incoming_pen[offsets[row]:offsets[row + 1]]
            tokens[row, length, 3] = 1.
    tokens.flush()
    del tokens, absolute, incoming_pen, offsets
    arrays = {'lengths': lengths.astype(np.int32), 'category_ids': category_ids,
              'duplicate_ids': duplicate_ids, 'base_ids': base_ids,
              'split_codes': split_codes, 'roles': roles}
    for name, values in arrays.items():
        np.save(root / f'{name}.npy', values, allow_pickle=False)
    neighbors = np.lib.format.open_memmap(root / 'neighbors.npy', mode='w+', dtype=np.int32, shape=(count, top_m))
    scores = np.lib.format.open_memmap(root / 'neighbor_scores.npy', mode='w+', dtype=np.float64, shape=(count, top_m))
    neighbors[:] = -1
    scores[:] = 0
    for code, split in enumerate(SPLITS):
        for role, suffix in ((EXAMPLE, 'rows'), (REFERENCE, 'reference_rows')):
            np.save(root / f'{split}_{suffix}.npy', np.flatnonzero((split_codes == code) & (roles == role)).astype(np.int32))
    previous_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(threads)
    entries = []
    try:
        work = [(code, split, category, name) for code, split in enumerate(SPLITS)
                for category, name in enumerate(categories)]
        for code, split, category, name in _progress(work, 'Building complete split/category NN tables',
                                                    unit='indexes', enabled=progress):
            rows = np.flatnonzero((category_ids == category) & (split_codes == code) & (roles == EXAMPLE)).astype(np.int32)
            vectors = np.ascontiguousarray(cosine[rows])
            index = faiss.IndexFlatIP(512)
            index.add(vectors)
            prefix = f'indexes/{split}_{category:04d}'
            faiss.write_index(index, str(root / f'{prefix}.faiss'))
            np.save(root / f'{prefix}_rows.npy', rows)
            for start, result, result_scores in _rank_distinct(index, vectors, rows, duplicate_ids, top_m, batch_size):
                query_rows = rows[start:start + len(result)]
                neighbors[query_rows], scores[query_rows] = result, result_scores
            entries.append({'split': split, 'category_id': category, 'category': name,
                            'count': len(rows), 'index_file': f'{prefix}.faiss',
                            'rows_file': f'{prefix}_rows.npy'})
    finally:
        faiss.omp_set_num_threads(previous_threads)
    neighbors.flush()
    scores.flush()
    del neighbors, scores
    _status('Hashing immutable policy arrays and indexes ...', progress)
    files = sorted(path for path in root.rglob('*') if path.is_file())
    hashes = {str(path.relative_to(root)): file_sha256(path)
              for path in _progress(files, 'Hashing completed dataset', unit='files', enabled=progress)}
    manifest = {
        'version': VERSION, 'schema_version': 1, 'count': count, 'categories': list(categories),
        'category_count': len(categories), 'category_counts': category_counts, 'splits': list(SPLITS),
        'split_counts': {split: {
            'examples': int(np.count_nonzero((split_codes == code) & (roles == EXAMPLE))),
            'references': int(np.count_nonzero((split_codes == code) & (roles == REFERENCE)))}
            for code, split in enumerate(SPLITS)},
        'exclusions': exclusions, 'all_categories_in_every_split': True,
        'target_coverage': 'every_nonreference_nonexcluded_drawing_once_per_epoch_before_shuffling',
        'split_regime': 'familiar_drawings', 'seed': seed, 'train_fraction': train_fraction,
        'development_fraction': development_fraction, 'reference_fraction': reference_fraction,
        'split_rule': 'per_category_floor_fractions_of_sha256_seed_sorted_duplicate_groups',
        'reference_rule': 'separate_sha256_order_whole_groups_minimum_two_per_split_when_enabled',
        'duplicate_rule': cache.get('duplicate_rule'),
        'retrieval_exclusions': 'query_duplicate_group_and_duplicate_groups_of_already_selected_supports',
        'cross_category_duplicates': 'exclude_entire_group_and_report_counts',
        'coordinate_mode': 'absolute', 'pen_semantics': 'destination_incoming_segment',
        'token_channels': ['x', 'y', 'pen_down', 'STOP'], 'max_steps': max_steps,
        'padding': 'zero_after_one_STOP_at_point_count', 'top_m': top_m,
        'selection_modes': ['exact_top_k', 'sample_top_m'],
        'neighbor_metric': 'float64_inner_product_of_stored_l2_normalized_float32_embeddings',
        'tie_policy': 'fixed_float64_numpy_row_sum_score_descending_then_base_id_lexicographic',
        'index_type': 'IndexFlatIP', 'faiss_version': faiss.__version__, 'numpy_version': np.__version__,
        'build_batch_size': batch_size, 'build_threads': threads,
        'cache_identifier': cache['identifier'], 'embedding_identifier': feature_manifest['identifier'],
        'source_cache_root': str(cache_root.resolve()), 'embedding_root': str(embedding_root.resolve()),
        'source_cache_files': feature_manifest['source_cache_files'],
        'source_cache_provenance': cache.get('provenance', {}),
        'extractor_sha256': feature_manifest.get('extractor_sha256'),
        'checkpoint_provenance_sha256': feature_manifest.get('checkpoint_provenance_sha256'),
        'embedding_resource_role': 'frozen_offline_curation_and_evaluation_never_policy_input',
        'embedding_resource_drawings': count, 'embedding_resource_categories': len(categories),
        'allow_subset_fixture': bool(feature_manifest.get('allow_subset_fixture', False)),
        'indexes': entries, 'files': hashes,
    }
    manifest['identifier'] = _identifier(manifest)
    (root / 'manifest.json').write_bytes(_canonical(manifest) + b'\n')
    _status(f'Saved {count:,} drawings across {len(categories)} categories: {root}', progress)
    return root


class FullDataset:
    """Framework-free memory-mapped policy inputs and offline row metadata."""

    def __init__(self, root, manifest, arrays, split_rows, reference_rows):
        self.root, self.manifest = root, manifest
        self.identifier = manifest['identifier']
        self.max_steps, self.top_m = manifest['max_steps'], manifest['top_m']
        self.categories = tuple(manifest['categories'])
        for name, values in arrays.items():
            setattr(self, name, values)
        self._split_rows, self._reference_rows = split_rows, reference_rows

    def __len__(self):
        return self.manifest['count']

    @classmethod
    def open(cls, root: str | Path, *, verify_hashes: bool = True) -> 'FullDataset':
        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('version') != VERSION or manifest.get('identifier') != _identifier(manifest):
            raise ValueError('Full-dataset manifest version/hash mismatch.')
        expected = {f'{name}.npy' for name in ARRAY_NAMES}
        expected.update(f'{split}_{suffix}.npy' for split in SPLITS for suffix in ('rows', 'reference_rows'))
        for entry in manifest['indexes']:
            expected.update((entry['index_file'], entry['rows_file']))
        if set(manifest.get('files', {})) != expected:
            raise ValueError('Full-dataset manifest files are incomplete or unexpected.')
        for name, digest in manifest['files'].items():
            path = root / name
            if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file():
                raise ValueError(f'Invalid or missing dataset file: {name}')
            if verify_hashes and file_sha256(path) != digest:
                raise ValueError(f'Full-dataset file hash mismatch: {name}')
        arrays = {name: np.load(root / f'{name}.npy', mmap_mode='r', allow_pickle=False) for name in ARRAY_NAMES}
        split_rows = {split: np.load(root / f'{split}_rows.npy', mmap_mode='r', allow_pickle=False) for split in SPLITS}
        reference_rows = {split: np.load(root / f'{split}_reference_rows.npy', mmap_mode='r', allow_pickle=False) for split in SPLITS}
        result = cls(root, manifest, arrays, split_rows, reference_rows)
        result._validate()
        return result

    def _validate(self):
        count, steps, top_m = len(self), self.max_steps, self.top_m
        if (count < 1 or steps < 2 or top_m < 1 or tuple(sorted(set(self.categories))) != self.categories
                or self.manifest['category_count'] != len(self.categories)):
            raise ValueError('Invalid full-dataset dimensions or categories.')
        if (self.manifest.get('token_channels') != ['x', 'y', 'pen_down', 'STOP']
                or self.manifest.get('coordinate_mode') != 'absolute'
                or self.manifest.get('pen_semantics') != 'destination_incoming_segment'):
            raise ValueError('Full-dataset token representation differs from the policy contract.')
        specifications = {'tokens': ((count, steps, 4), np.float32),
                          'lengths': ((count,), np.int32), 'category_ids': ((count,), np.int32),
                          'duplicate_ids': ((count,), np.int32), 'split_codes': ((count,), np.int8),
                          'roles': ((count,), np.uint8), 'neighbors': ((count, top_m), np.int32),
                          'neighbor_scores': ((count, top_m), np.float64)}
        for name, (shape, dtype) in specifications.items():
            values = getattr(self, name)
            if values.shape != shape or values.dtype != dtype:
                raise ValueError(f'Invalid full-dataset array shape/dtype: {name}')
        if (self.base_ids.shape != (count,) or self.base_ids.dtype.kind != 'U'
                or np.any(self.base_ids[1:] <= self.base_ids[:-1])):
            raise ValueError('Full-dataset base IDs must be sorted unique Unicode rows.')
        if (np.any(self.lengths < 1) or np.any(self.lengths >= steps)
                or np.any(self.category_ids < 0) or np.any(self.category_ids >= len(self.categories))
                or np.any(self.duplicate_ids < 0) or np.any(self.roles > EXCLUDED)
                or np.any(self.split_codes[self.roles != EXCLUDED] < 0)
                or np.any(self.split_codes > 2) or np.any(self.split_codes[self.roles == EXCLUDED] != -1)):
            raise ValueError('Invalid lengths, category, duplicate, split, or role codes.')
        # One duplicate group never crosses a split or reference/example role.
        order = np.argsort(self.duplicate_ids, kind='stable')
        equal = self.duplicate_ids[order[1:]] == self.duplicate_ids[order[:-1]]
        if (np.any(self.split_codes[order[1:]][equal] != self.split_codes[order[:-1]][equal])
                or np.any(self.roles[order[1:]][equal] != self.roles[order[:-1]][equal])):
            raise ValueError('A duplicate group crosses dataset splits or reference/example roles.')
        cross_category = equal & (self.category_ids[order[1:]] != self.category_ids[order[:-1]])
        if np.any(self.roles[order[1:]][cross_category] != EXCLUDED):
            raise ValueError('Cross-category duplicate groups must remain excluded.')
        expected_entries = {(split, category) for split in SPLITS for category in range(len(self.categories))}
        entries = self.manifest['indexes']
        if (len(entries) != len(expected_entries)
                or {(entry['split'], entry['category_id']) for entry in entries} != expected_entries):
            raise ValueError('FAISS indexes must cover every split and category exactly once.')
        category_sizes = {}
        for code, split in enumerate(SPLITS):
            for role, row_table in ((EXAMPLE, self.rows(split)), (REFERENCE, self.reference_rows(split))):
                expected = np.flatnonzero((self.split_codes == code) & (self.roles == role))
                if row_table.dtype != np.int32 or not np.array_equal(row_table, expected):
                    raise ValueError(f'{split} compact row table differs from declared split/role membership.')
            if set(self.category_ids[self.rows(split)].tolist()) != set(range(len(self.categories))):
                raise ValueError(f'Every category must have examples in {split}.')
            actual = {'examples': len(self.rows(split)), 'references': len(self.reference_rows(split))}
            if self.manifest['split_counts'][split] != actual:
                raise ValueError(f'{split} realized counts differ from manifest.')
            for role, row_table in ((EXAMPLE, self.rows(split)), (REFERENCE, self.reference_rows(split))):
                category_sizes[split, role] = np.bincount(self.category_ids[row_table], minlength=len(self.categories))
            for category, name in enumerate(self.categories):
                declared = self.manifest['category_counts'][name][split]
                if (declared['examples'] != category_sizes[split, EXAMPLE][category]
                        or declared['references'] != category_sizes[split, REFERENCE][category]):
                    raise ValueError('Per-category realized counts differ from manifest.')
        for entry in entries:
            split, category = entry['split'], entry['category_id']
            rows = np.load(self.root / entry['rows_file'], mmap_mode='r', allow_pickle=False)
            if (rows.ndim != 1 or rows.dtype != np.int32
                    or entry['category'] != self.categories[category]
                    or len(rows) != entry['count'] or len(rows) != category_sizes[split, EXAMPLE][category]
                    or np.any(rows < 0) or np.any(rows >= count) or np.any(np.diff(rows) <= 0)):
                raise ValueError('FAISS row mapping dimensions/order/count differ from the candidate pool.')
            if (np.any(self.category_ids[rows] != category) or np.any(self.roles[rows] != EXAMPLE)
                    or np.any(self.split_codes[rows] != SPLITS.index(split))):
                raise ValueError('FAISS row mapping crosses split/category/reference boundaries.')
        for start in range(0, count, 4096):
            end = min(start + 4096, count)
            values, lengths = self.tokens[start:end], self.lengths[start:end]
            time = np.arange(steps)[None]
            point = time < lengths[:, None]
            stop = time == lengths[:, None]
            if (not np.all(np.isfinite(values)) or not np.array_equal(values[..., 3], stop.astype(np.float32))
                    or not np.all((values[..., 2] == 0) | (values[..., 2] == 1))
                    or np.any(values[:, 0, 2] != 0) or np.any(values[..., :3][~point] != 0)):
                raise ValueError('Policy token geometry, incoming pen, STOP, or padding is invalid.')
            eligible = self.roles[start:end] == EXAMPLE
            neighbors = self.neighbors[start:end][eligible]
            rows = np.arange(start, end)[eligible]
            if np.any(self.neighbors[start:end][~eligible] != -1):
                raise ValueError('Reserved/excluded rows cannot have retrieval neighbors.')
            if np.any(neighbors < 0) or np.any(neighbors >= count):
                raise ValueError('Example neighbor tables contain an invalid row.')
            if len(rows):
                groups = self.duplicate_ids[neighbors]
                if (np.any(self.roles[neighbors] != EXAMPLE)
                        or np.any(self.split_codes[neighbors] != self.split_codes[rows, None])
                        or np.any(self.category_ids[neighbors] != self.category_ids[rows, None])
                        or np.any(groups == self.duplicate_ids[rows, None])
                        or np.any(np.diff(np.sort(groups, axis=1), axis=1) == 0)):
                    raise ValueError('Neighbor rows violate split/category/duplicate/reference boundaries.')
                scores = self.neighbor_scores[rows]
                if not np.all(np.isfinite(scores)) or np.any(np.abs(scores) > 1.00001) or np.any(np.diff(scores, axis=1) > 0):
                    raise ValueError('Neighbor scores must be finite descending cosine values.')
                ties = scores[:, 1:] == scores[:, :-1]
                if np.any((neighbors[:, 1:] < neighbors[:, :-1]) & ties):
                    raise ValueError('Neighbor score ties must use stable base-ID order.')

    def rows(self, split: str):
        return self._split_rows[_split_name(split)]

    def reference_rows(self, split: str, category_id: int | None = None):
        rows = self._reference_rows[_split_name(split)]
        if category_id is None:
            return rows
        if not 0 <= category_id < len(self.categories):
            raise ValueError('Unknown reference category ID.')
        return rows[self.category_ids[rows] == category_id]

    def batch(self, target_rows, *, support_count: int = 4,
              selection_mode: str = 'exact_top_k', rng=None) -> dict:
        targets = np.asarray(target_rows)
        if targets.ndim != 1 or targets.dtype.kind not in 'iu' or not len(targets):
            raise ValueError('Target rows must be a nonempty one-dimensional integer array.')
        if np.any(targets < 0) or np.any(targets >= len(self)) or np.any(self.roles[targets] != EXAMPLE):
            raise ValueError('Target rows must be eligible examples, never reserved reference rows.')
        selection_mode = 'sample_top_m' if selection_mode == 'sampled_top_m' else selection_mode
        if not 0 <= support_count <= self.top_m or selection_mode not in ('exact_top_k', 'sample_top_m'):
            raise ValueError('Support count must fit top_m and selection must be exact_top_k or sample_top_m.')
        if selection_mode == 'sample_top_m':
            if not isinstance(rng, np.random.Generator):
                raise ValueError('sample_top_m requires an explicit NumPy Generator for reproducibility.')
            choices = np.stack([rng.choice(self.top_m, support_count, replace=False) for _ in targets])
            supports = np.take_along_axis(self.neighbors[targets], choices, axis=1)
        else:
            supports = np.asarray(self.neighbors[targets, :support_count])
        time = np.arange(self.max_steps)
        return {
            'support_tokens': np.asarray(self.tokens[supports]),
            'support_mask': time <= self.lengths[supports, None],
            'query_tokens': np.asarray(self.tokens[targets]),
            'query_mask': time <= self.lengths[targets, None],
            'query_point_mask': time < self.lengths[targets, None],
            'example_mask': np.ones(len(targets), bool),
            'metadata': {'target_rows': targets.astype(np.int32), 'support_rows': supports,
                         'category_ids': np.asarray(self.category_ids[targets])},
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Build complete split-local token and nearest-neighbor resources.')
    build.add_argument('--cache-root', required=True)
    build.add_argument('--embedding-root', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--seed', type=int, default=0)
    build.add_argument('--top-m', type=int, default=32)
    build.add_argument('--max-steps', type=int, default=128)
    build.add_argument('--train-fraction', type=float, default=.7)
    build.add_argument('--development-fraction', type=float, default=.15)
    build.add_argument('--reference-fraction', type=float, default=.15)
    build.add_argument('--batch-size', type=int, default=256)
    build.add_argument('--threads', type=int, default=4)
    build.add_argument('--no-progress', action='store_true')
    args = vars(parser.parse_args(argv))
    args.pop('command')
    args['progress'] = not args.pop('no_progress')
    root = build_full_dataset(**args)
    manifest = json.loads((root / 'manifest.json').read_text())
    print(json.dumps({'output': str(root), 'identifier': manifest['identifier'],
                      'categories': manifest['category_count'], 'split_counts': manifest['split_counts']}, indent=2))


if __name__ == '__main__':
    main()
