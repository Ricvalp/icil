"""Class-balanced Sketch-FID for full-dataset supervised sketch policies.

This is Gaussian Frechet distance in the frozen sketch-trained ResNet18's raw
512-dimensional features, not ImageNet Inception FID. References are reserved
real drawings; independent held-out query drawings select the context neighbors.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .classifier_data import _progress, _status
from .full_data import FullDataset
from . import metrics


VERSION = 1


def _identifier(value):
    payload = {key: item for key, item in value.items() if key != 'identifier'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def _split(split, allow_test=False):
    split = 'development' if split == 'validation' else split
    if split not in ('development', 'test'):
        raise ValueError('Sketch-FID uses held-out development or test drawings.')
    if split == 'test' and not allow_test:
        raise ValueError('Test evaluation requires explicit --allow-test.')
    return split


def _balanced_rows(dataset, rows, samples_per_category, seed, domain):
    if samples_per_category < 1 or seed < 0:
        raise ValueError('samples_per_category must be positive and seed nonnegative.')
    selected = []
    for category in range(len(dataset.categories)):
        candidates = np.asarray(rows)[dataset.category_ids[rows] == category]
        # One representative per duplicate group, sampled without replacement.
        _, first = np.unique(dataset.duplicate_ids[candidates], return_index=True)
        candidates = candidates[np.sort(first)]
        if len(candidates) < samples_per_category:
            raise ValueError(f'{dataset.categories[category]} has only {len(candidates)} unique '
                             f'drawings for the requested {samples_per_category} samples.')
        rng = np.random.default_rng(np.random.SeedSequence([seed, domain, category]))
        selected.append(rng.permutation(candidates)[:samples_per_category])
    # Round-robin categories keep small prefixes representative for diagnostics.
    return np.stack(selected, axis=1).ravel().astype(np.int32)


def select_centroids(dataset, split='development', samples_per_category=100, seed=2029):
    """Choose distinct query duplicate groups, without looking at their tokens."""
    return _balanced_rows(dataset, dataset.rows(split), samples_per_category, seed, 0x51554552)


def _reference_sequences(dataset, rows):
    lengths = np.asarray(dataset.lengths[rows], np.int32)
    time_axis = np.arange(dataset.max_steps)
    return {'tokens': np.asarray(dataset.tokens[rows]), 'lengths': lengths,
            'event_mask': time_axis < lengths[:, None] + 1,
            'point_mask': time_axis < lengths[:, None], 'stopped': np.ones(len(rows), bool)}


def _trajectory_metadata(dataset, records, selection_id):
    return {'schema_version': 1, 'coordinate_mode': 'absolute', 'pen_semantics': 'incoming',
            'renderer': metrics.RENDERER_CONFIG, 'expected_count': len(records), 'records': records,
            'manifest_id': selection_id, 'evaluation_id': selection_id,
            'dataset_identifier': dataset.identifier,
            'reference_protocol': 'class_balanced_unique_reserved_drawings_v1'}


def _resources(python, donor_root, extractor_checkpoint, checkpoint_provenance=None):
    executable = shutil.which(str(python))
    if executable is None:
        raise FileNotFoundError(f'Metric Python interpreter is missing or not executable: {python}')
    _check_interpreter(str(Path(executable).absolute()))
    donor = Path(donor_root)
    sources = {'rasterizer': donor / 'dataset/rasterize.py', 'resnet18': donor / 'metrics/resnet18.py'}
    for path in (Path(extractor_checkpoint), *sources.values()):
        if not path.is_file():
            raise FileNotFoundError(f'Required frozen metric resource is missing: {path}')
    digest = metrics.file_sha256(extractor_checkpoint)
    provenance_hash = None
    if checkpoint_provenance is not None:
        provenance = json.loads(Path(checkpoint_provenance).read_text())
        if provenance.get('extractor_sha256') != digest:
            raise ValueError('Checkpoint provenance hash differs from the frozen evaluator.')
        provenance_hash = metrics.file_sha256(checkpoint_provenance)
    return {'extractor_sha256': digest,
            'source_hashes': {key: metrics.file_sha256(path) for key, path in sources.items()},
            'checkpoint_provenance_sha256': provenance_hash}


@lru_cache(maxsize=8)
def _check_interpreter(python):
    result = subprocess.run([python, '-c', 'import numpy, PIL, scipy, torch, torchvision'],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f'Metric interpreter lacks working NumPy/Pillow/SciPy/Torch/torchvision: '
                         f'{python}\n{result.stderr.strip()}')


@dataclass(frozen=True)
class Reference:
    root: Path
    metadata: dict
    target_rows: np.ndarray
    reference_rows: np.ndarray
    features: np.ndarray
    feature_metadata: dict

    @property
    def identifier(self):
        return self.metadata['identifier']

    @property
    def split(self):
        return self.metadata['split']

    @property
    def samples_per_category(self):
        return self.metadata['samples_per_category']


def _validate_selection(dataset, metadata, targets, references):
    count = metadata['samples_per_category']
    expected = count * len(dataset.categories)
    if metadata['dataset_identifier'] != dataset.identifier:
        raise ValueError('FID reference and full dataset identities differ.')
    if metadata['categories'] != list(dataset.categories):
        raise ValueError('FID reference category population differs from dataset.')
    for rows, allowed, role in ((targets, dataset.rows(metadata['split']), 'centroid'),
                                (references, dataset.reference_rows(metadata['split']), 'reference')):
        if rows.shape != (expected,) or rows.dtype.kind not in 'iu' or not np.all(np.isin(rows, allowed)):
            raise ValueError(f'FID {role} rows cross the declared split/role boundary.')
        if len(np.unique(dataset.duplicate_ids[rows])) != expected:
            raise ValueError(f'FID {role} population repeats duplicate groups.')
        if not np.array_equal(np.bincount(dataset.category_ids[rows], minlength=len(dataset.categories)),
                              np.full(len(dataset.categories), count)):
            raise ValueError(f'FID {role} population is not class balanced.')
    if np.intersect1d(dataset.duplicate_ids[targets], dataset.duplicate_ids[references]).size:
        raise ValueError('FID references overlap query centroid duplicate groups.')
    neighbors = np.asarray(dataset.neighbors[targets])
    if (np.any(neighbors < 0) or np.any(neighbors >= len(dataset))
            or not np.all(np.isin(neighbors, dataset.rows(metadata['split'])))
            or np.any(dataset.category_ids[neighbors] != dataset.category_ids[targets, None])
            or np.any(dataset.duplicate_ids[neighbors] == dataset.duplicate_ids[targets, None])
            or np.intersect1d(dataset.duplicate_ids[neighbors], dataset.duplicate_ids[references]).size):
        raise ValueError('FID context neighbors leak centroids/references or cross split/category boundaries.')


def prepare_reference(dataset_root, output, *, samples_per_category=100, split='development',
                      seed=2029, python, donor_root, extractor_checkpoint,
                      checkpoint_provenance=None, batch_size=64, allow_test=False, progress=True):
    """Render and freeze real references once; no embedding cache is required."""
    split = _split(split, allow_test)
    if batch_size < 1:
        raise ValueError('Feature batch size must be positive.')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'FID references are immutable; use a new output: {output}')
    resources = _resources(python, donor_root, extractor_checkpoint, checkpoint_provenance)
    _status('Verifying full dataset and selecting class-balanced FID populations ...', progress)
    dataset = FullDataset.open(dataset_root)
    targets = select_centroids(dataset, split, samples_per_category, seed)
    references = _balanced_rows(dataset, dataset.reference_rows(split), samples_per_category, seed, 0x52454653)
    selection = {'version': VERSION, 'dataset_identifier': dataset.identifier,
                 'split': split, 'samples_per_category': samples_per_category, 'seed': seed,
                 'categories': list(dataset.categories), 'target_rows': targets.tolist(),
                 'reference_rows': references.tolist()}
    _validate_selection(dataset, selection, targets, references)
    selection_id = _identifier(selection)
    records = [{'drawing_id': str(dataset.base_ids[row]), 'reference_row': int(row),
                'intended_category': dataset.categories[int(dataset.category_ids[row])]}
               for row in references]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as temporary:
        stage = Path(temporary)
        np.savez_compressed(stage / 'selection.npz', target_rows=targets, reference_rows=references)
        metrics.save_trajectories(stage / 'real', _reference_sequences(dataset, references),
                                  _trajectory_metadata(dataset, records, selection_id))
        _status(f'Extracting frozen real features for {len(references):,} drawings on CPU ...', progress)
        metrics.extract_features(stage / 'real', stage / 'features', python=python, donor_root=donor_root,
            extractor_checkpoint=extractor_checkpoint, checkpoint_provenance=checkpoint_provenance,
            batch_size=batch_size)
        features, feature_metadata = metrics.load_features(stage / 'features')
        if (len(features) != len(references) or feature_metadata['extractor_sha256'] != resources['extractor_sha256']
                or feature_metadata['source_hashes'] != resources['source_hashes']):
            raise ValueError('Extracted reference features differ from requested population/resources.')
        # The compact transferable artifact needs only rows, features, and provenance.
        shutil.rmtree(stage / 'real')
        (stage / 'features/rasters.npz').unlink(missing_ok=True)
        metadata = {key: value for key, value in selection.items() if key not in ('target_rows', 'reference_rows')}
        metadata.update(selection_id=selection_id, resources=resources,
            feature_version=metrics.FEATURE_VERSION, fd_version=metrics.FD_VERSION,
            renderer_config=metrics.RENDERER_CONFIG, feature_normalization='none',
            protocol='one generation per independent held-out centroid; reserved real references; uniform categories',
            files={name: metrics.file_sha256(stage / name) for name in
                   ('selection.npz', 'features/features.npz', 'features/metadata.json')})
        metadata['identifier'] = _identifier(metadata)
        _dump(stage / 'metadata.json', metadata)
        stage.rename(output)
    _status(f'Frozen FID reference saved: {output}', progress)
    return output


def load_reference(root, dataset, *, allow_test=False):
    root = Path(root)
    metadata = json.loads((root / 'metadata.json').read_text())
    if metadata.get('version') != VERSION or metadata.get('identifier') != _identifier(metadata):
        raise ValueError('FID reference metadata changed or has an unsupported version.')
    _split(metadata['split'], allow_test)
    expected = {'selection.npz', 'features/features.npz', 'features/metadata.json'}
    if set(metadata.get('files', {})) != expected:
        raise ValueError('FID reference file manifest is incomplete.')
    for name, digest in metadata['files'].items():
        if metrics.file_sha256(root / name) != digest:
            raise ValueError(f'FID reference file hash mismatch: {name}')
    with np.load(root / 'selection.npz', allow_pickle=False) as archive:
        targets, references = archive['target_rows'], archive['reference_rows']
    _validate_selection(dataset, metadata, targets, references)
    selection = {key: metadata[key] for key in ('version', 'dataset_identifier', 'split',
                                               'samples_per_category', 'seed', 'categories')}
    selection.update(target_rows=targets.tolist(), reference_rows=references.tolist())
    if _identifier(selection) != metadata['selection_id']:
        raise ValueError('FID reference selection identity mismatch.')
    features, feature_metadata = metrics.load_features(root / 'features')
    if (len(features) != len(references) or feature_metadata.get('evaluation_id') != metadata['selection_id']
            or feature_metadata.get('extractor_sha256') != metadata['resources']['extractor_sha256']
            or feature_metadata.get('source_hashes') != metadata['resources']['source_hashes']
            or feature_metadata.get('renderer_config') != metrics.RENDERER_CONFIG):
        raise ValueError('FID reference features and selection/evaluator provenance differ.')
    for row, record in zip(references, feature_metadata['records']):
        if (record.get('drawing_id') != str(dataset.base_ids[row])
                or record.get('intended_category') != dataset.categories[int(dataset.category_ids[row])]):
            raise ValueError('FID reference feature records differ from selected drawing identities.')
    return Reference(root, metadata, targets, references, features, feature_metadata)


def select_reference_categories(reference, dataset, category_ids):
    """Select matching centroid/real-feature populations without rendering again.

    The parent artifact stays immutable. A subset has its own identity and
    aligned feature records; full-category selection keeps the original identity.
    """
    ids = np.asarray(category_ids)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= len(dataset.categories))
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('FID category IDs must be distinct, nonempty, valid integers.')
    ids = np.sort(ids).astype(np.int32)
    if reference.metadata['dataset_identifier'] != dataset.identifier:
        raise ValueError('FID reference and full dataset identities differ.')
    available = np.asarray(reference.metadata.get('category_ids',
        np.arange(len(dataset.categories))), dtype=np.int32)
    if not np.all(np.isin(ids, available)):
        raise ValueError('Requested categories are absent from the FID reference population.')
    if np.array_equal(ids, np.sort(available)):
        return reference
    target_mask = np.isin(dataset.category_ids[reference.target_rows], ids)
    real_mask = np.isin(dataset.category_ids[reference.reference_rows], ids)
    targets, real_rows = reference.target_rows[target_mask], reference.reference_rows[real_mask]
    expected = len(ids) * reference.samples_per_category
    for rows in (targets, real_rows):
        counts = np.bincount(dataset.category_ids[rows], minlength=len(dataset.categories))
        if len(rows) != expected or not np.all(counts[ids] == reference.samples_per_category):
            raise ValueError('Selected FID population is not balanced over the requested categories.')
    selection = {
        'parent_reference_id': reference.identifier, 'dataset_identifier': dataset.identifier,
        'split': reference.split, 'samples_per_category': reference.samples_per_category,
        'category_ids': ids.tolist(), 'categories': [dataset.categories[int(i)] for i in ids],
        'target_rows': targets.tolist(), 'reference_rows': real_rows.tolist(),
    }
    selection_id = _identifier(selection)
    metadata = {key: value for key, value in reference.metadata.items()
                if key not in ('identifier', 'selection_id', 'files')}
    metadata.update({key: value for key, value in selection.items()
                     if key not in ('target_rows', 'reference_rows')})
    metadata.update(selection_id=selection_id,
        parent_selection_id=reference.metadata['selection_id'],
        parent_files=reference.metadata.get('files', reference.metadata.get('parent_files', {})),
        category_selection='in_memory_subset_of_frozen_centroids_and_reserved_real_features')
    metadata['identifier'] = _identifier(metadata)
    feature_metadata = {key: value for key, value in reference.feature_metadata.items()
                        if key != 'statistics'}
    feature_metadata.update(
        records=[record for record, selected in zip(reference.feature_metadata['records'], real_mask) if selected],
        actual_count=expected, expected_count=expected, manifest_id=selection_id,
        evaluation_id=selection_id, parent_reference_id=reference.identifier,
        category_ids=ids.tolist(), category_selection=metadata['category_selection'])
    return Reference(reference.root, metadata, targets, real_rows,
                     reference.features[real_mask], feature_metadata)


def preflight(reference, *, python, donor_root, extractor_checkpoint, checkpoint_provenance=None):
    """Fail before training/generation if reference and current evaluator differ."""
    current = _resources(python, donor_root, extractor_checkpoint, checkpoint_provenance)
    if current != reference.metadata['resources']:
        raise ValueError('Frozen FID evaluator/donor/provenance differs from the prepared reference.')


@lru_cache(maxsize=8)
def _generator(model_cfg):
    import jax
    from .policy_backend import generate

    def one(params, tokens, mask, key):
        result = generate(params, tokens[None], mask[None], model_cfg, key)
        return {name: value[0] for name, value in result.items()}

    # Independent one-example random streams also survive changes to batch size.
    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0)))


def generate_samples(params, model_cfg, dataset, target_rows, *, support_count,
                     selection_mode, condition_on_support=True, seed=2030,
                     batch_size=64, progress=True):
    """Generate from context only, with keys determined by global centroid row."""
    import jax
    import jax.numpy as jnp

    targets = np.asarray(target_rows)
    if (targets.ndim != 1 or targets.dtype.kind not in 'iu' or not len(targets)
            or batch_size < 1 or not 0 <= seed < 2 ** 32):
        raise ValueError('Need nonempty target rows, a positive batch size and an unsigned 32-bit seed.')
    targets = targets.astype(np.int64)
    if model_cfg.max_steps != dataset.max_steps:
        raise ValueError('Model and dataset trajectory lengths differ.')
    if not 1 <= support_count <= dataset.top_m:
        raise ValueError('Support count must be between one and the saved neighbor-table width.')
    selection_mode = 'sample_top_m' if selection_mode == 'sampled_top_m' else selection_mode
    if selection_mode not in ('exact_top_k', 'sample_top_m'):
        raise ValueError('Unknown context neighbor selection mode.')
    allowed = np.concatenate([dataset.rows(split) for split in ('train', 'development', 'test')])
    if not np.all(np.isin(targets, allowed)):
        raise ValueError('Generation centroids must be nonreference examples.')
    supports = []
    for row in targets:
        candidates = np.asarray(dataset.neighbors[row], np.int64)
        if selection_mode == 'sample_top_m':
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53555050, int(row)]))
            candidates = candidates[rng.choice(dataset.top_m, support_count, replace=False)]
        else:
            candidates = candidates[:support_count]
        supports.append(candidates)
    supports = np.asarray(supports, np.int64)
    if (np.any(supports < 0) or np.any(supports >= len(dataset))
            or np.any(dataset.duplicate_ids[supports] == dataset.duplicate_ids[targets, None])
            or np.any(dataset.category_ids[supports] != dataset.category_ids[targets, None])
            or np.any(np.diff(np.sort(dataset.duplicate_ids[supports], axis=1), axis=1) == 0)):
        raise ValueError('Context includes invalid, repeated, cross-category or query centroid duplicate groups.')
    for split in ('train', 'development', 'test'):
        same_split = np.isin(targets, dataset.rows(split))
        if not np.all(np.isin(supports[same_split], dataset.rows(split))):
            raise ValueError('Context crosses example/reference or dataset split boundaries.')
    generator = _generator(model_cfg)
    base_key = jax.random.fold_in(jax.random.PRNGKey(seed), 0x46494431)
    blocks, records = [], []
    for start in _progress(range(0, len(targets), batch_size), 'Generating held-out sketches',
                           unit='batches', enabled=progress):
        count = min(batch_size, len(targets) - start)
        indices = np.minimum(np.arange(start, start + batch_size), len(targets) - 1)
        shown = supports[indices]
        # No query tokens, IDs, categories, embeddings, or lengths enter the model.
        tokens = np.asarray(dataset.tokens[shown])
        mask = np.arange(dataset.max_steps) <= dataset.lengths[shown, None]
        if not condition_on_support:
            tokens, mask = np.zeros_like(tokens), np.zeros_like(mask)
        keys = jnp.stack([jax.random.fold_in(base_key, int(row)) for row in targets[indices]])
        generated = jax.tree_util.tree_map(lambda value: np.asarray(value)[:count],
                                          generator(params, tokens, mask, keys))
        blocks.append(generated)
        for local, index in enumerate(indices[:count]):
            row = targets[index]
            actual_supports = supports[index].tolist() if condition_on_support else []
            records.append({'target_row': int(row), 'target_id': str(dataset.base_ids[row]),
                'query_id': str(dataset.base_ids[row]), 'support_rows': actual_supports,
                'support_ids': [str(dataset.base_ids[item]) for item in actual_supports],
                'category_id': int(dataset.category_ids[row]),
                'intended_category': dataset.categories[int(dataset.category_ids[row])],
                'query_seed': np.asarray(keys[local]).astype(int).tolist()})
    arrays = {name: np.concatenate([block[name] for block in blocks]) for name in blocks[0]}
    arrays['lengths'] = arrays.pop('length').astype(np.int32)
    return arrays, records


def evaluate_params(params, model_cfg, dataset, reference, output, *, support_count,
                    selection_mode, condition_on_support=True, seed=2030, batch_size=64,
                    python, donor_root, extractor_checkpoint, checkpoint_provenance=None,
                    feature_batch_size=64, keep_artifacts=False, checkpoint_id=None,
                    optimizer_step=None, progress=True, class_split=None, category_scope='auto'):
    """Evaluate live parameters without saving a checkpoint or advancing train RNG."""
    import jax
    from .policy_backend import method_name, numerical_sources

    if feature_batch_size < 1:
        raise ValueError('Feature batch size must be positive.')
    if reference.metadata['dataset_identifier'] != dataset.identifier:
        raise ValueError('FID reference and full dataset identities differ.')
    if class_split is not None:
        from .class_split import evaluation_category_ids
        expected = evaluation_category_ids(dataset, class_split, category_scope)
        actual = sorted(int(value) for value in np.unique(dataset.category_ids[reference.target_rows]))
        if actual != sorted(expected):
            raise ValueError('FID reference categories differ from the checkpoint evaluation scope.')
    preflight(reference, python=python, donor_root=donor_root,
              extractor_checkpoint=extractor_checkpoint, checkpoint_provenance=checkpoint_provenance)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'reference_id': reference.identifier, 'generation_seed': seed,
                'support_count': support_count, 'selection_mode': selection_mode,
                'condition_on_support': condition_on_support, 'checkpoint_id': checkpoint_id,
                'optimizer_step': optimizer_step}
    if class_split is not None:
        identity.update(class_split_id=class_split['identifier'], category_scope=category_scope)
    previous_path = output / 'summary.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError('Existing FID output uses a different reference, seed, context, step, or checkpoint.')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='.fid-', dir=output) as temporary:
        work = Path(temporary)
        arrays, records = generate_samples(params, model_cfg, dataset, reference.target_rows,
            support_count=support_count, selection_mode=selection_mode,
            condition_on_support=condition_on_support, seed=seed, batch_size=batch_size, progress=progress)
        metadata = _trajectory_metadata(dataset, records, reference.metadata['selection_id'])
        generation_provenance = {
            'model': asdict(model_cfg) if model_cfg is not None else None,
            'method': method_name(model_cfg) if model_cfg is not None else None,
            'source_hashes': {name: metrics.file_sha256(Path(__file__).with_name(name)) for name in
                              ('supervised_fid.py', 'class_split.py', 'metrics.py', 'metrics_worker.py')},
            'environment': {name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'numpy')},
            'backend': jax.default_backend(),
            'device_kinds': [device.device_kind for device in jax.local_devices()],
        }
        if model_cfg is not None:
            generation_provenance['source_hashes'].update({
                name: metrics.file_sha256(path) for name, path in numerical_sources(model_cfg).items()})
        metadata.update(reference_id=reference.identifier, checkpoint_id=checkpoint_id,
                        optimizer_step=optimizer_step, generation_seed=seed,
                        generation_provenance=generation_provenance)
        if class_split is not None:
            metadata.update(class_split=class_split, category_scope=category_scope)
        metadata['evaluation_categories'] = reference.metadata['categories']
        metrics.save_trajectories(work / 'generated', arrays, metadata)
        _status(f'Extracting frozen features for {len(records):,} generated sketches on CPU ...', progress)
        metrics.extract_features(work / 'generated', work / 'features', python=python,
            donor_root=donor_root, extractor_checkpoint=extractor_checkpoint,
            checkpoint_provenance=checkpoint_provenance, batch_size=feature_batch_size)
        generated, feature_metadata = metrics.load_features(work / 'features')
        metrics._compatible(feature_metadata, reference.feature_metadata)
        if len(generated) != len(reference.features):
            raise ValueError('Generated and reference sample counts differ; failures must be retained.')
        score = metrics.sketch_fd(generated, reference.features)
        report = {'metric': 'sketch_fid', 'sketch_fid': score,
            'feature_version': metrics.FEATURE_VERSION, 'fd_version': metrics.FD_VERSION,
            'feature_dimension': 512, 'feature_normalization': 'none',
            'description': 'Pooled Frechet distance in frozen sketch-trained ResNet18 features; not Inception FID',
            'dataset_identifier': dataset.identifier, 'reference_id': reference.identifier,
            'selection_id': reference.metadata['selection_id'], 'split': reference.split,
            'category_weighting': 'uniform', 'category_count': len(reference.metadata['categories']),
            'categories': reference.metadata['categories'],
            'samples_per_category': reference.samples_per_category,
            'generated_count': len(generated), 'reference_count': len(reference.features),
            'generation_batch_size': batch_size, 'feature_batch_size': feature_batch_size,
            'generation_seed': seed, 'support_count': support_count, 'selection_mode': selection_mode,
            'condition_on_support': condition_on_support, 'checkpoint_id': checkpoint_id,
            'optimizer_step': optimizer_step, 'statistics': feature_metadata['statistics'],
            'extractor_sha256': feature_metadata['extractor_sha256'],
            'source_hashes': feature_metadata['source_hashes'],
            'renderer_config': feature_metadata['renderer_config'],
            'generation_provenance': generation_provenance,
            'metric_environment': feature_metadata.get('environment'),
            'generated_feature_sha256': metrics.file_sha256(work / 'features/features.npz'),
            'generated_trajectories_sha256': metrics.file_sha256(work / 'generated/trajectories.npz'),
            'elapsed_seconds': time.monotonic() - started,
            'uncertainty': 'Finite-sample, model-dependent biased estimate; compare fixed populations/sample counts.',
            'artifacts_retained': bool(keep_artifacts)}
        if 'parent_reference_id' in reference.metadata:
            report['parent_reference_id'] = reference.metadata['parent_reference_id']
        if class_split is not None:
            report.update(class_split=class_split, class_split_id=class_split['identifier'],
                          category_scope=category_scope)
        # Small enough for periodic provenance; full images/features are opt-in.
        _dump(work / 'generation.json', metadata)
        (work / 'generation.json').replace(output / 'generation.json')
        if keep_artifacts:
            destination = output / 'artifacts'
            if destination.exists():
                # Same fixed run identity was checked above; replay after a
                # checkpoint restart replaces only this generated artifact.
                if not previous_path.exists():
                    raise FileExistsError(f'Evaluation artifacts already exist without a summary: {destination}')
                shutil.rmtree(destination)
            work.rename(destination)
        elif previous_path.exists() and (output / 'artifacts').exists():
            shutil.rmtree(output / 'artifacts')
        _dump(output / 'summary.json.tmp', report)
        (output / 'summary.json.tmp').replace(output / 'summary.json')
    _status(f'Sketch-FID: {score:.6f}; {len(generated):,} generated / {len(reference.features):,} real', progress)
    return report


def evaluate_checkpoint(checkpoint, reference, output, *, dataset_root=None, allow_test=False,
                        category_scope='auto', **kwargs):
    import jax
    import jax.numpy as jnp
    from .supervised_train import load_run
    from .class_split import evaluation_category_ids, resolve_class_split

    payload, cfg, model_cfg, dataset = load_run(checkpoint, dataset_root=dataset_root)
    frozen = load_reference(reference, dataset, allow_test=allow_test)
    class_split = resolve_class_split(dataset, cfg)
    categories = evaluation_category_ids(dataset, cfg, category_scope)
    frozen = select_reference_categories(frozen, dataset, categories)
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])
    return evaluate_params(params, model_cfg, dataset, frozen, output,
        support_count=cfg['support_count'], selection_mode=cfg['selection_mode'],
        condition_on_support=cfg.get('condition_on_support', True),
        checkpoint_id=metrics.file_sha256(checkpoint), optimizer_step=int(payload['step']),
        class_split=class_split if class_split['heldout_category_count'] else None,
        category_scope=category_scope, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Freeze balanced held-out real references once.')
    prepare.add_argument('--dataset-root', required=True)
    prepare.add_argument('--samples-per-category', type=int, default=100)
    prepare.add_argument('--split', choices=('development', 'test'), default='development')
    prepare.add_argument('--seed', type=int, default=2029)
    prepare.add_argument('--batch-size', type=int, default=64)
    evaluate = commands.add_parser('evaluate', help='Score a trained policy checkpoint.')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--reference', required=True)
    evaluate.add_argument('--dataset-root')
    evaluate.add_argument('--category-scope', choices=('auto', 'heldout', 'seen', 'all'), default='auto')
    evaluate.add_argument('--seed', type=int, default=2030)
    evaluate.add_argument('--batch-size', type=int, default=64)
    evaluate.add_argument('--feature-batch-size', type=int, default=64)
    evaluate.add_argument('--keep-artifacts', action='store_true')
    for command in (prepare, evaluate):
        command.add_argument('--output', required=True)
        command.add_argument('--python', default='.venv-quickdraw-metrics/bin/python')
        command.add_argument('--donor-root', default='../quick-robot-draw')
        command.add_argument('--extractor-checkpoint', default='outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt')
        command.add_argument('--checkpoint-provenance',
                             default='outputs/quickdraw_evaluator/resnet18_v1/provenance.json')
        command.add_argument('--allow-test', action='store_true')
        command.add_argument('--no-progress', dest='progress', action='store_false')
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    result = prepare_reference(**args) if command == 'prepare' else evaluate_checkpoint(**args)
    print(result if command == 'prepare' else json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
