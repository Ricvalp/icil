"""FID protocol, deterministic context-only generation, and failure accounting."""
from __future__ import annotations

import json
import sys

import jax
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import full_data, metrics, supervised_fid as fid
from icil_jax_rlbench.quickdraw.supervised_models import SupervisedModelConfig, init_model


def _dataset(tmp_path, monkeypatch):
    count, steps = 64, 6
    categories = np.repeat(np.arange(2, dtype=np.int32), count // 2)
    tokens = np.zeros((count, steps, 4), np.float32)
    lengths = np.full(count, 4, np.int32)
    for row in range(count):
        tokens[row, :4, 0] = np.linspace(-.7, .7, 4)
        tokens[row, :4, 1] = np.sin(np.linspace(0, 3, 4) + row) * .6
        tokens[row, 1:4, 2] = 1
        tokens[row, 4, 3] = 1
    roles = np.zeros(count, np.uint8)
    rows, references = {}, {}
    neighbors = np.full((count, 3), -1, np.int32)
    for split, lo, hi, rlo, rhi in (('train', 0, 8, 26, 26),
                                  ('development', 8, 20, 26, 30), ('test', 20, 26, 30, 32)):
        rows[split] = np.concatenate([np.arange(lo, hi) + offset for offset in (0, 32)]).astype(np.int32)
        references[split] = np.concatenate([np.arange(rlo, rhi) + offset for offset in (0, 32)]).astype(np.int32)
        roles[references[split]] = full_data.REFERENCE
        for row in rows[split]:
            candidates = rows[split][(categories[rows[split]] == categories[row]) & (rows[split] != row)]
            neighbors[row] = candidates[:3]
    duplicates = np.arange(count, dtype=np.int32)
    duplicates[27] = duplicates[26]
    duplicates[59] = duplicates[58]
    dataset = full_data.FullDataset(tmp_path, {'identifier': 'synthetic-full-fid', 'max_steps': steps,
        'top_m': 3, 'count': count, 'categories': ['cat', 'dog']},
        dict(tokens=tokens, lengths=lengths, category_ids=categories, duplicate_ids=duplicates,
             base_ids=np.asarray([f'{categories[row]}/{row}' for row in range(count)]),
             roles=roles, neighbors=neighbors), rows, references)
    monkeypatch.setattr(full_data.FullDataset, 'open', lambda *args, **kwargs: dataset)
    return dataset


def _resources(tmp_path, monkeypatch):
    donor = tmp_path / 'donor'
    for file in ('dataset/rasterize.py', 'metrics/resnet18.py'):
        path = donor / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic evaluator fixture\n')
    checkpoint = tmp_path / 'classifier.pt'
    checkpoint.write_bytes(b'synthetic-classifier')
    monkeypatch.setattr(fid, '_check_interpreter', lambda python: None)
    monkeypatch.setattr(metrics, 'extract_features', _extract)
    return dict(python=sys.executable, donor_root=donor, extractor_checkpoint=checkpoint)


def _extract(artifact, output, *, donor_root, extractor_checkpoint, **kwargs):
    arrays, metadata = metrics.load_trajectories(artifact)
    tokens = np.nan_to_num(arrays['tokens'], nan=0.)
    features = np.zeros((len(tokens), 512), np.float32)
    features[:, :4] = np.mean(tokens, axis=1)
    features[:, 4:8] = np.std(tokens, axis=1)
    output = fid.Path(output)
    output.mkdir(parents=True)
    np.savez_compressed(output / 'features.npz', features=features)
    # Synthetic rasters model blank retention for the bookkeeping assertions.
    rasters = np.zeros((len(tokens), 2, 2), np.float32)
    finite = np.all(np.isfinite(arrays['tokens'][..., :2]), axis=(1, 2))
    rasters[np.asarray(arrays['lengths']) > 0] = 1.
    rasters[~finite] = 0.
    metadata.update(feature_version=metrics.FEATURE_VERSION, feature_normalization='none',
        feature_dimension=512, actual_count=len(tokens), extractor_sha256=metrics.file_sha256(extractor_checkpoint),
        source_hashes={'rasterizer': metrics.file_sha256(fid.Path(donor_root) / 'dataset/rasterize.py'),
                       'resnet18': metrics.file_sha256(fid.Path(donor_root) / 'metrics/resnet18.py')},
        renderer_config=metrics.RENDERER_CONFIG, statistics=metrics.trajectory_statistics(arrays, rasters))
    fid._dump(output / 'metadata.json', metadata)


def _reference(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    resources = _resources(tmp_path, monkeypatch)
    path = fid.prepare_reference(tmp_path, tmp_path / 'reference', samples_per_category=2,
                                  seed=9, progress=False, **resources)
    return dataset, resources, fid.load_reference(path, dataset)


def test_reference_balanced_unique_reserved_immutable_and_hash_verified(tmp_path, monkeypatch):
    dataset, resources, reference = _reference(tmp_path, monkeypatch)
    assert reference.samples_per_category == 2
    assert reference.split == 'development'
    assert list(dataset.category_ids[reference.target_rows]) == [0, 1, 0, 1]
    assert len(set(dataset.duplicate_ids[reference.reference_rows])) == 4
    assert not set(reference.reference_rows) & set(dataset.neighbors[reference.target_rows].ravel())
    assert not (reference.root / 'real').exists()
    fid.preflight(reference, **resources)
    with pytest.raises(FileExistsError, match='immutable'):
        fid.prepare_reference(tmp_path, reference.root, **resources)
    with pytest.raises(ValueError, match='unique drawings'):
        fid.prepare_reference(tmp_path, tmp_path / 'too-many', samples_per_category=4, **resources)
    with pytest.raises(ValueError, match='allow-test'):
        fid.prepare_reference(tmp_path, tmp_path / 'test', split='test', **resources)
    resources['extractor_checkpoint'].write_bytes(b'different-classifier')
    with pytest.raises(ValueError, match='differs'):
        fid.preflight(reference, **resources)
    metadata = json.loads((reference.root / 'metadata.json').read_text())
    metadata['seed'] += 1
    fid._dump(reference.root / 'metadata.json', metadata)
    with pytest.raises(ValueError, match='metadata changed'):
        fid.load_reference(reference.root, dataset)


def test_reference_refuses_neighbor_leaks_and_tampered_features(tmp_path, monkeypatch):
    dataset, _, reference = _reference(tmp_path, monkeypatch)
    row = reference.target_rows[0]
    original = dataset.neighbors[row, 0]
    dataset.neighbors[row, 0] = reference.reference_rows[0]
    with pytest.raises(ValueError, match='leak'):
        fid.load_reference(reference.root, dataset)
    dataset.neighbors[row, 0] = row
    with pytest.raises(ValueError, match='leak'):
        fid.load_reference(reference.root, dataset)
    dataset.neighbors[row, 0] = original
    with (reference.root / 'features/features.npz').open('ab') as handle:
        handle.write(b'tampering')
    with pytest.raises(ValueError, match='hash mismatch'):
        fid.load_reference(reference.root, dataset)


@pytest.mark.parametrize('architecture', ['autoregressive', 'diffusion'])
def test_real_generation_keys_and_neighbor_selection_survive_batch_size(tmp_path, monkeypatch, architecture):
    dataset = _dataset(tmp_path, monkeypatch)
    cfg = SupervisedModelConfig(architecture=architecture, hidden_dim=8, num_heads=2,
        context_layers=1, decoder_layers=1, mlp_ratio=2, max_steps=6,
        mixture_components=2, dropout=0., diffusion_steps=3, dtype='float32')
    params = init_model(jax.random.PRNGKey(2), cfg, support_count=2)
    targets = fid.select_centroids(dataset, samples_per_category=2)
    kwargs = dict(support_count=2, selection_mode='sample_top_m', seed=12, progress=False)
    first, first_records = fid.generate_samples(params, cfg, dataset, targets, batch_size=2, **kwargs)
    second, second_records = fid.generate_samples(params, cfg, dataset, targets, batch_size=3, **kwargs)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert first_records == second_records
    for row, record in zip(targets, first_records):
        expected_rng = np.random.default_rng(np.random.SeedSequence([12, 0x53555050, int(row)]))
        np.testing.assert_array_equal(record['support_rows'],
            dataset.neighbors[row][expected_rng.choice(dataset.top_m, 2, replace=False)])
        assert row not in record['support_rows']
    # Altering query-only tokens must not enter generation. Pick a target whose
    # row is absent from every chosen support set, and change only that row.
    support_rows = {row for record in first_records for row in record['support_rows']}
    unseen = next(row for row in targets if row not in support_rows)
    dataset.tokens[unseen, :, :2] += 100.
    repeated, _ = fid.generate_samples(params, cfg, dataset, targets, batch_size=2, **kwargs)
    for name in first:
        np.testing.assert_array_equal(first[name], repeated[name])
    _, unconditioned = fid.generate_samples(params, cfg, dataset, targets, batch_size=2,
                                             condition_on_support=False, **kwargs)
    assert all(not record['support_rows'] and not record['support_ids'] for record in unconditioned)


def test_evaluation_retains_empty_invalid_samples_and_safe_replay(tmp_path, monkeypatch):
    dataset, resources, reference = _reference(tmp_path, monkeypatch)
    arrays = fid._reference_sequences(dataset, reference.target_rows)
    arrays['tokens'][0] = 0.
    arrays['tokens'][0, 0, 3] = 1.
    arrays['lengths'][0] = 0
    arrays['point_mask'][0] = False
    arrays['event_mask'][0] = False
    arrays['event_mask'][0, 0] = True
    arrays['tokens'][1, 0, 0] = np.nan
    records = [{'intended_category': dataset.categories[dataset.category_ids[row]],
                'target_row': int(row), 'support_ids': []} for row in reference.target_rows]
    monkeypatch.setattr(fid, 'generate_samples', lambda *args, **kwargs: (arrays, records))
    kwargs = dict(support_count=2, selection_mode='exact_top_k', optimizer_step=10000,
                  progress=False, **resources)
    output = tmp_path / 'evaluated'
    report = fid.evaluate_params(None, None, dataset, reference, output, **kwargs)
    assert np.isfinite(report['sketch_fid'])
    assert report['generated_count'] == report['reference_count'] == 4
    assert report['statistics']['empty_count'] == 1
    assert report['statistics']['invalid_count'] == 1
    assert report['statistics']['blank_count'] == 2
    assert set(path.name for path in output.iterdir()) == {'generation.json', 'summary.json'}
    replay = fid.evaluate_params(None, None, dataset, reference, output, **kwargs)
    assert replay['sketch_fid'] == report['sketch_fid']
    with pytest.raises(ValueError, match='different reference, seed'):
        fid.evaluate_params(None, None, dataset, reference, output, seed=42, **kwargs)
    retained = tmp_path / 'retained'
    fid.evaluate_params(None, None, dataset, reference, retained, keep_artifacts=True, **kwargs)
    assert (retained / 'artifacts/generated/trajectories.npz').is_file()
    fid.evaluate_params(None, None, dataset, reference, retained, keep_artifacts=True, **kwargs)
