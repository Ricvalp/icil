from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from icil_jax_rlbench.quickdraw.data import SketchSampler, build_manifest, create_fixture_cache, save_manifest
from icil_jax_rlbench.quickdraw.metrics import file_sha256, load_trajectories
from icil_jax_rlbench.quickdraw.populations import export_copy_populations


def test_copy_populations_are_frozen_permitted_actual_A_supports(tmp_path):
    cache = tmp_path / 'cache'
    store = create_fixture_cache(cache, categories=16, drawings_per_category=20)
    manifest = build_manifest(store, family_count=2, unique_budget=8, max_points=15)
    manifest_path = tmp_path / 'manifest.json'
    save_manifest(manifest_path, manifest)
    sampler = SketchSampler(store, manifest, split='development', max_steps=16, support_count=2)
    batch = sampler.build_batch(2)
    wrong = sampler.build_batch(2)
    episodes = tmp_path / 'episodes'
    episodes.mkdir()
    arrays = {f'support.{key}': value for key, value in batch['support'].items()}
    arrays.update({f'wrong_support.{key}': value for key, value in wrong['support'].items()})
    np.savez(episodes / 'episodes.npz', **arrays)
    metadata = {'experiment': 'a', 'cache_id': store.identifier, 'manifest_id': manifest['identifier'],
                'split': 'development', 'max_steps': 16, 'records': batch['meta']['tasks'],
                'wrong_records': wrong['meta']['tasks'], 'data_sha256': file_sha256(episodes / 'episodes.npz')}
    metadata['identifier'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    (episodes / 'metadata.json').write_text(json.dumps(metadata))
    left, right = tmp_path / 'copy-left', tmp_path / 'copy-right'
    export_copy_populations(cache, manifest_path, episodes, left, training_count=3, seed=7)
    export_copy_populations(cache, manifest_path, episodes, right, training_count=3, seed=7)
    selection = json.loads((left / 'selection.json').read_text())
    assert (left / 'selection.json').read_text() == (right / 'selection.json').read_text()
    assert selection['selected_training_count'] == 3
    assert selection['permitted_training_count'] == 8
    expected_support = {item for records in (metadata['records'], metadata['wrong_records']) for record in records for item in record['support_ids']}
    support_arrays, support_metadata = load_trajectories(left / 'supports')
    assert {record['drawing_id'] for record in support_metadata['records']} == expected_support
    assert len(support_arrays['tokens']) == len(expected_support)
    train_arrays, train_metadata = load_trajectories(left / 'training_reservoir')
    allowed = {item for ids in manifest['a_ids']['train'].values() for item in ids}
    assert {record['drawing_id'] for record in train_metadata['records']} <= allowed
    assert len(train_arrays['tokens']) == 3
    with pytest.raises(ValueError, match='cannot exceed'):
        export_copy_populations(cache, manifest_path, episodes, tmp_path / 'too-big', training_count=9)
    metadata['experiment'] = 'b1'
    metadata.pop('identifier')
    metadata['identifier'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    (episodes / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='canonical A only'):
        export_copy_populations(cache, manifest_path, episodes, tmp_path / 'wrong-experiment')
