"""Offline frozen sketch features and split-local exact cosine indexes.

Run extraction/build-index in the separate metrics interpreter. Importing the
EmbeddingStore reader imports NumPy only, never Torch, FAISS, or JAX. Raw pooled
512D features retain the Sketch-FD convention; cosine vectors are separate.
Example: python -m icil_jax_rlbench.quickdraw.embeddings extract --help
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from .metrics import FEATURE_VERSION, RENDERER_CONFIG, file_sha256


EMBEDDING_VERSION = 'frozen_resnet18_raw512_and_cosine_v1'
INDEX_VERSION = 'split_category_flat_ip_v1'


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _identifier(value) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in value.items() if k != 'identifier'})).hexdigest()


class EmbeddingStore:
    """Validated immutable feature rows with a memory-mapped NumPy interface."""

    def __init__(self, root: Path, manifest: dict, records: tuple[dict, ...], raw, cosine):
        self.root, self.manifest, self.records = root, manifest, records
        self.raw, self.cosine = raw, cosine
        self.identifier = manifest['identifier']
        self.cache_id = manifest['cache_identifier']
        self._by_id = {record['base_id']: index for index, record in enumerate(records)}

    def row(self, base_id: str) -> int:
        return self._by_id[base_id]

    @classmethod
    def open(cls, root: str | Path, *, verify_hashes: bool = True) -> 'EmbeddingStore':
        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('embedding_version') != EMBEDDING_VERSION or manifest.get('identifier') != _identifier(manifest):
            raise ValueError('Embedding manifest version/hash mismatch.')
        expected_files = {'raw.npy', 'cosine.npy', 'records.jsonl'}
        if set(manifest.get('files', {})) != expected_files:
            raise ValueError('Embedding manifest has incomplete files.')
        if verify_hashes:
            for name, digest in manifest['files'].items():
                if file_sha256(root / name) != digest:
                    raise ValueError(f'Embedding file hash mismatch: {name}')
        with (root / 'records.jsonl').open() as handle:
            records = tuple(json.loads(line) for line in handle)
        if not records or len({row['base_id'] for row in records}) != len(records):
            raise ValueError('Embedding drawing IDs must be nonempty and unique.')
        if any(row.get('row_index') != index or not row.get('category') or not row.get('duplicate_cluster_id')
               for index, row in enumerate(records)):
            raise ValueError('Embedding record rows/categories/duplicate clusters are invalid.')
        raw = np.load(root / 'raw.npy', mmap_mode='r', allow_pickle=False)
        cosine = np.load(root / 'cosine.npy', mmap_mode='r', allow_pickle=False)
        shape = (len(records), 512)
        if (raw.shape != shape or cosine.shape != shape or raw.dtype != np.float32
                or cosine.dtype != np.float32 or manifest.get('actual_count') != len(records)):
            raise ValueError('Embedding arrays must be aligned float32 [drawings,512].')
        if manifest.get('raw_normalization') != 'none' or manifest.get('cosine_normalization') != 'l2':
            raise ValueError('Raw and retrieval feature conventions must remain separate.')
        # Bounded memory even for a multi-million drawing export.
        for start in range(0, len(records), 8192):
            values, normalized = raw[start:start + 8192], cosine[start:start + 8192]
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if not np.all(np.isfinite(values)) or np.any(norms <= 1e-12):
                raise ValueError('Cosine retrieval requires finite nonzero feature vectors.')
            if not np.allclose(normalized, values / norms, rtol=3e-6, atol=3e-7):
                raise ValueError('Cosine array does not normalize the recorded raw features.')
        return cls(root, manifest, records, raw, cosine)


def _checkpoint_provenance(checkpoint: Path, path: str | Path | None, *, allow_subset_fixture: bool) -> tuple[dict, str | None]:
    # Fresh checkpoints carry an adjacent attestation; accept legacy checkpoint
    # provenance as unknown, while never silently overlooking a known smoke run.
    provenance_path = Path(path) if path else checkpoint.with_name('provenance.json')
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get('extractor_sha256') != file_sha256(checkpoint):
            raise ValueError('Checkpoint provenance hash does not match the selected checkpoint.')
        smoke = provenance.get('smoke_only', False) or provenance.get('config', {}).get('smoke_only', False)
        subset = provenance.get('cache_partition_provenance', {}).get('allow_subset_fixture', False)
        if (smoke or subset) and not allow_subset_fixture:
            raise ValueError('A smoke/subset classifier cannot curate scientific tasks; use --allow-subset-fixture only for diagnostics.')
        if provenance.get('renderer_config', RENDERER_CONFIG) != RENDERER_CONFIG:
            raise ValueError('Classifier provenance uses a different rendering convention.')
        return provenance, file_sha256(provenance_path)
    if path:
        raise FileNotFoundError(f'Checkpoint provenance is missing: {provenance_path}')
    return {'training_ids': 'unknown', 'checkpoint_role': 'external_frozen_checkpoint_unknown_training'}, None


def extract_embeddings(
    cache_root: str | Path, donor_root: str | Path, checkpoint: str | Path,
    output: str | Path, *, checkpoint_provenance: str | Path | None = None,
    batch_size: int = 256, device: str = 'auto', threads: int = 4,
    allow_subset_fixture: bool = False, progress: bool = True,
) -> EmbeddingStore:
    """Render one batch at a time; preserve raw features and a normalized copy."""
    from .classifier_data import _load_renderer, _progress, _status
    from .data import SketchStore
    from .metrics_worker import load_source

    if batch_size < 1 or threads < 1 or device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Need positive batch size/threads and device auto, cpu, or cuda.')
    root, donor_root, checkpoint = Path(output), Path(donor_root), Path(checkpoint)
    if root.exists():
        raise FileExistsError(f'Embedding outputs are immutable: {root}')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Frozen classifier checkpoint is missing: {checkpoint}')
    provenance, provenance_hash = _checkpoint_provenance(checkpoint, checkpoint_provenance,
                                                        allow_subset_fixture=allow_subset_fixture)
    checkpoint_hash = file_sha256(checkpoint)
    _status(f'Loading and validating policy vector cache: {cache_root} ...', progress)
    store = SketchStore.open(cache_root)
    if not allow_subset_fixture and any(record.provenance.get('synthetic_fixture') for record in store.records):
        raise ValueError('Synthetic fixture data requires --allow-subset-fixture.')
    renderer_path, model_path = donor_root / 'dataset/rasterize.py', donor_root / 'metrics/resnet18.py'
    renderer = _load_renderer(renderer_path)
    original = load_source(model_path, '_quickdraw_frozen_curation_resnet18')
    import torch
    import torchvision
    import PIL

    torch.set_num_threads(threads)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable in the offline interpreter.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = original.resnet18(pretrained=False, num_classes=345)
    model.conv1 = original.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.fc = original.nn.Identity()
    model.eval().requires_grad_(False).to(device)
    config = renderer.RasterizerConfig(**RENDERER_CONFIG)
    root.mkdir(parents=True, exist_ok=False)
    raw = np.lib.format.open_memmap(root / 'raw.npy', mode='w+', dtype=np.float32, shape=(len(store), 512))
    cosine = np.lib.format.open_memmap(root / 'cosine.npy', mode='w+', dtype=np.float32, shape=(len(store), 512))
    with (root / 'records.jsonl').open('x') as records_file, torch.inference_mode():
        for start in _progress(range(0, len(store), batch_size), 'Rendering and extracting embeddings',
                               unit='batches', enabled=progress):
            records = store.records[start:start + batch_size]
            rasters = np.stack([renderer.rasterize_absolute_points(
                np.column_stack((record.absolute, record.incoming_pen)), config=config) for record in records])
            features = model(torch.from_numpy(rasters[:, None]).to(device=device, dtype=torch.float32))
            values = features.cpu().numpy()
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if values.shape != (len(records), 512) or not np.all(np.isfinite(values)) or np.any(norms <= 1e-12):
                raise ValueError('Frozen extractor must produce finite nonzero 512D vectors.')
            raw[start:start + len(records)] = values
            cosine[start:start + len(records)] = values / norms
            for index, record in enumerate(records, start):
                records_file.write(_canonical({'row_index': index, 'base_id': record.base_id,
                    'category': record.category, 'duplicate_cluster_id': record.duplicate_cluster_id}).decode() + '\n')
    raw.flush()
    cosine.flush()
    del raw, cosine
    _status('Hashing feature arrays and record mapping ...', progress)
    if file_sha256(checkpoint) != checkpoint_hash:
        raise ValueError('Classifier checkpoint changed during extraction; freeze it before building curation resources.')
    manifest = {
        'schema_version': 1, 'embedding_version': EMBEDDING_VERSION,
        'cache_identifier': store.identifier, 'actual_count': len(store),
        'category_counts': dict(sorted(Counter(record.category for record in store.records).items())),
        'feature_dimension': 512, 'feature_version': FEATURE_VERSION,
        'feature_definition': 'frozen_grayscale_345_class_resnet18_global_average_pool_before_fc',
        'raw_normalization': 'none', 'cosine_normalization': 'l2',
        'resource_role': 'fixed_offline_task_curation_and_evaluation_never_policy_input',
        'allow_subset_fixture': bool(allow_subset_fixture),
        'extractor_sha256': checkpoint_hash,
        'checkpoint_provenance_sha256': provenance_hash,
        'extractor_training_provenance': provenance,
        'renderer_config': dict(RENDERER_CONFIG),
        'source_hashes': {'rasterizer': file_sha256(renderer_path), 'resnet18': file_sha256(model_path),
                          'exporter': file_sha256(__file__)},
        'source_cache_files': {name: file_sha256(Path(cache_root) / name) for name in ('records.npz', 'index.json')},
        'files': {name: file_sha256(root / name) for name in ('raw.npy', 'cosine.npy', 'records.jsonl')},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                        'torchvision': torchvision.__version__, 'pillow': PIL.__version__, 'device': device,
                        'batch_size': batch_size, 'threads': threads, 'tf32': False},
    }
    manifest['identifier'] = _identifier(manifest)
    (root / 'manifest.json').write_bytes(_canonical(manifest) + b'\n')
    _status(f'Saved {len(store):,} raw and cosine feature rows: {root}', progress)
    return EmbeddingStore.open(root, verify_hashes=False)


def _candidate_pools(embeddings: EmbeddingStore, manifest: dict) -> dict:
    if manifest.get('identifier') != _identifier(manifest):
        raise ValueError('Task manifest hash mismatch.')
    if manifest.get('cache_identifier') != embeddings.cache_id:
        raise ValueError('Task manifest and embeddings refer to different vector caches.')
    feature_id = manifest.get('feature_identifier', manifest.get('embedding_identifier',
                              manifest.get('curation', {}).get('embedding_identifier')))
    if feature_id is not None and feature_id != embeddings.identifier:
        raise ValueError('Task manifest and index use different frozen features.')
    pools = manifest['a_ids']
    seen_ids, seen_clusters = {}, {}
    # Anchors/references also belong to a declared split even though only model
    # examples enter indexes. Check their boundaries before constructing files.
    for split, categories in manifest.get('split_pools', {}).items():
        for category, roles in categories.items():
            for ids in roles.values():
                for item in ids:
                    record = embeddings.records[embeddings.row(item)]
                    cluster = record['duplicate_cluster_id']
                    if (record['category'] != category or item in seen_ids
                            or cluster in seen_clusters):
                        raise ValueError('Split roles contain overlapping IDs/duplicate clusters or mismatched categories.')
                    seen_ids[item], seen_clusters[cluster] = split, split
    declared_ids = dict(seen_ids)
    seen_ids, seen_clusters = {}, {}
    for split, categories in pools.items():
        if split not in ('train', 'development', 'test'):
            raise ValueError(f'Unknown candidate split: {split}')
        for category, ids in categories.items():
            if not ids or len(ids) != len(set(ids)):
                raise ValueError('Index candidate pools must be nonempty with distinct IDs.')
            if 'split_pools' in manifest and set(ids) != set(manifest['split_pools'][split][category]['example_ids']):
                raise ValueError('Index candidates must equal model-example roles, excluding references and anchors.')
            for item in ids:
                record = embeddings.records[embeddings.row(item)]
                if record['category'] != category:
                    raise ValueError('Index candidate category mismatch.')
                cluster = record['duplicate_cluster_id']
                if declared_ids and declared_ids.get(item) != split:
                    raise ValueError('Index candidate differs from its declared split.')
                if item in seen_ids or (cluster in seen_clusters and seen_clusters[cluster] != split):
                    raise ValueError('Index candidates cross split ID/duplicate-cluster boundaries.')
                seen_ids[item], seen_clusters[cluster] = split, split
    return pools


def build_indexes(embedding_root: str | Path, task_manifest: str | Path | dict,
                  output: str | Path, *, progress: bool = True) -> Path:
    """Build an exact FAISS index per declared split/category candidate reservoir."""
    from .classifier_data import _progress
    import faiss

    root = Path(output)
    if root.exists():
        raise FileExistsError(f'Index outputs are immutable: {root}')
    embeddings = EmbeddingStore.open(embedding_root)
    manifest = task_manifest if isinstance(task_manifest, dict) else json.loads(Path(task_manifest).read_text())
    pools = _candidate_pools(embeddings, manifest)
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    work = [(split, category, ids) for split, categories in sorted(pools.items())
            for category, ids in sorted(categories.items())]
    for split, category, ids in _progress(work, 'Building split-local cosine indexes', unit='indexes', enabled=progress):
        ids = sorted(ids)
        rows = np.asarray([embeddings.row(item) for item in ids], np.int64)
        index = faiss.IndexFlatIP(512)
        index.add(np.ascontiguousarray(embeddings.cosine[rows]))
        prefix = f'{split}_{hashlib.sha256(category.encode()).hexdigest()[:16]}'
        index_file, mapping_file = prefix + '.faiss', prefix + '.json'
        faiss.write_index(index, str(root / index_file))
        mapping = [{'index_row': i, 'embedding_row': int(row), **embeddings.records[row]}
                   for i, row in enumerate(rows)]
        (root / mapping_file).write_bytes(_canonical(mapping) + b'\n')
        entries.append({'split': split, 'category': category, 'count': len(ids),
                        'allowed_query_ids': sorted(set(ids) | set(manifest.get('construction_anchors', {}).get(split, {}).get(category, []))),
                        'index_file': index_file, 'mapping_file': mapping_file,
                        'index_sha256': file_sha256(root / index_file),
                        'mapping_sha256': file_sha256(root / mapping_file)})
    result = {
        'schema_version': 1, 'index_version': INDEX_VERSION, 'index_type': 'IndexFlatIP',
        'metric': 'inner_product_of_l2_normalized_512D_embeddings',
        'cache_identifier': embeddings.cache_id, 'feature_identifier': embeddings.identifier,
        'task_manifest_identifier': manifest['identifier'],
        'task_manifest_sha256': (hashlib.sha256(_canonical(manifest)).hexdigest()
                                 if isinstance(task_manifest, dict) else file_sha256(task_manifest)),
        'task_manifest_hash_encoding': 'canonical_json' if isinstance(task_manifest, dict) else 'file_bytes',
        'extractor_sha256': embeddings.manifest['extractor_sha256'],
        'eligible_membership': 'task_manifest_a_ids_split_then_category',
        'exclusions': 'query_or_anchor_id_and_declared_duplicate_clusters_applied_at_retrieval',
        'tie_policy': 'cosine_descending_then_base_id_lexicographic_after_recomputing_float64_scores',
        'feature_curation_resource_count': len(embeddings.records),
        'indexed_candidate_count': sum(entry['count'] for entry in entries),
        'entries': entries, 'faiss_version': faiss.__version__, 'numpy_version': np.__version__,
        'source_sha256': file_sha256(__file__),
    }
    result['identifier'] = _identifier(result)
    (root / 'manifest.json').write_bytes(_canonical(result) + b'\n')
    return root


def search_index(index_root: str | Path, embeddings: EmbeddingStore, *, split: str,
                 category: str, query_id: str, k: int, selection_mode: str = 'exact_top_k',
                 top_m: int | None = None, seed: int = 0,
                 exclusion_mode: str = 'duplicate_cluster', exclude_ids=()) -> list[dict]:
    """Offline retrieval; scientific filtering is separate from named legacy mode.

    Search all exact-index entries before filtering so large duplicate groups
    cannot shorten the requested result. Canonical float64 rescoring resolves
    backend-dependent ties consistently with the numerical manifest builder.
    """
    import faiss

    if k < 1 or selection_mode not in ('exact_top_k', 'sample_top_m', 'sample_from_top_m'):
        raise ValueError('Invalid K or retrieval selection mode.')
    if exclusion_mode not in ('duplicate_cluster', 'legacy_exact_id_only'):
        raise ValueError('Invalid retrieval exclusion mode.')
    root = Path(index_root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('identifier') != _identifier(manifest) or manifest.get('index_version') != INDEX_VERSION:
        raise ValueError('Index manifest version/hash mismatch.')
    if manifest.get('feature_identifier') != embeddings.identifier:
        raise ValueError('Index feature fingerprint mismatch.')
    entry = next((entry for entry in manifest['entries'] if (entry['split'], entry['category']) == (split, category)), None)
    if entry is None:
        raise ValueError('No index exists for the declared split/category.')
    for name, digest in ((entry['index_file'], entry['index_sha256']), (entry['mapping_file'], entry['mapping_sha256'])):
        if file_sha256(root / name) != digest:
            raise ValueError(f'Index file hash mismatch: {name}')
    mapping = json.loads((root / entry['mapping_file']).read_text())
    query = embeddings.records[embeddings.row(query_id)]
    if query['category'] != category:
        raise ValueError('Query and index categories differ.')
    if query_id not in entry['allowed_query_ids']:
        raise ValueError('Query/anchor does not belong to the declared index split.')
    index = faiss.read_index(str(root / entry['index_file']))
    if index.d != 512 or index.ntotal != len(mapping) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError('FAISS index dimensions/count/metric mismatch.')
    rows = np.asarray([row['embedding_row'] for row in mapping], np.int64)
    for i, row in enumerate(mapping):
        original = embeddings.records[rows[i]]
        if row['index_row'] != i or any(row[key] != original[key] for key in ('base_id', 'category', 'duplicate_cluster_id')):
            raise ValueError('Index-to-drawing mapping mismatch.')
    # Flat indices retain exact vectors; verify the stored row mapping itself.
    if not np.array_equal(index.reconstruct_n(0, index.ntotal), embeddings.cosine[rows]):
        raise ValueError('Index vectors differ from their declared embedding rows.')
    _, found = index.search(np.ascontiguousarray(embeddings.cosine[embeddings.row(query_id)][None]), index.ntotal)
    vector = np.asarray(embeddings.cosine[embeddings.row(query_id)], np.float64)
    excluded = set(exclude_ids) | {query_id}
    blocked_clusters = {embeddings.records[embeddings.row(item)]['duplicate_cluster_id'] for item in excluded}
    candidates = []
    for i in found[0]:
        row = mapping[int(i)]
        if row['base_id'] in excluded or (exclusion_mode == 'duplicate_cluster' and row['duplicate_cluster_id'] in blocked_clusters):
            continue
        candidates.append({**row, 'cosine_similarity': float(np.asarray(embeddings.cosine[row['embedding_row']], np.float64) @ vector)})
    candidates.sort(key=lambda row: (-row['cosine_similarity'], row['base_id']))
    if exclusion_mode == 'duplicate_cluster':
        distinct, seen = [], set()
        for row in candidates:
            if row['duplicate_cluster_id'] not in seen:
                seen.add(row['duplicate_cluster_id'])
                distinct.append(row)
        candidates = distinct
    if len(candidates) < k:
        raise ValueError('Insufficient eligible same-category neighbors; no arbitrary replenishment.')
    if selection_mode in ('sample_top_m', 'sample_from_top_m'):
        if top_m is None or top_m < k:
            raise ValueError('Sample-from-top-M requires M >= K.')
        candidates = candidates[:top_m]
        chosen = np.random.default_rng(seed).choice(len(candidates), size=k, replace=False)
        return [candidates[int(i)] for i in chosen]
    return candidates[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    extract = sub.add_parser('extract', help='Render policy trajectories and freeze raw/cosine features.')
    extract.add_argument('--cache-root', required=True)
    extract.add_argument('--donor-root', required=True)
    extract.add_argument('--extractor-checkpoint', required=True)
    extract.add_argument('--checkpoint-provenance')
    extract.add_argument('--output', required=True)
    extract.add_argument('--batch-size', type=int, default=256)
    extract.add_argument('--threads', type=int, default=4)
    extract.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    extract.add_argument('--allow-subset-fixture', action='store_true')
    extract.add_argument('--no-progress', action='store_true')
    build = sub.add_parser('build-index', help='Build exact split/category indexes from a task manifest.')
    build.add_argument('--embedding-root', required=True)
    build.add_argument('--manifest', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    if args.command == 'extract':
        result = extract_embeddings(args.cache_root, args.donor_root, args.extractor_checkpoint,
            args.output, checkpoint_provenance=args.checkpoint_provenance, batch_size=args.batch_size,
            device=args.device, threads=args.threads, allow_subset_fixture=args.allow_subset_fixture,
            progress=not args.no_progress)
        print(json.dumps({'feature_identifier': result.identifier, 'actual_count': len(result.records), 'output': args.output}))
    else:
        result = build_indexes(args.embedding_root, args.manifest, args.output, progress=not args.no_progress)
        print(json.dumps({'output': str(result), 'identifier': json.loads((result / 'manifest.json').read_text())['identifier']}))


if __name__ == '__main__':
    main()
