from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import jax
import numpy as np
import optax
import pytest

from icil_jax_rlbench.quickdraw.analysis import (
    _overlap_groups, actual_update_alignment, extract_checkpoint_updates, grouped_category_probe,
    participation_ratio,
)
from icil_jax_rlbench.quickdraw.data import create_fixture_cache, build_manifest, save_manifest
from icil_jax_rlbench.quickdraw.evaluate import prepare_evaluation
from icil_jax_rlbench.quickdraw.model import SketchModelConfig, init_sketch_params
from icil_jax_rlbench.quickdraw.train import resolve_config
from icil_jax_rlbench.train.checkpoints import save_checkpoint
from icil_jax_rlbench.train.ttt_step import create_ttt_train_state


def test_participation_ratio_known_rank_and_zero_conventions():
    rank_two = np.asarray([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
    result = participation_ratio(rank_two)
    assert result['effective_dimension'] == 2.
    assert result['maximum_identifiable_dimension'] == 3
    assert participation_ratio(np.zeros((5, 3)))['effective_dimension'] == 0.
    assert participation_ratio(np.full((7, 3), .1))['effective_dimension'] == 0.
    assert participation_ratio(np.zeros((0, 3)))['effective_dimension'] == 0.
    assert participation_ratio(np.asarray([[1., 3.]]))['effective_dimension'] == 0.
    np.testing.assert_allclose(participation_ratio(rank_two * .3 + .7)['effective_dimension'], 2.)
    with pytest.raises(ValueError, match='finite'):
        participation_ratio(np.asarray([[np.nan]]))


def test_actual_update_alignment_has_positive_sign_for_descent():
    gradient = np.asarray([2., -1.])
    descent = actual_update_alignment(gradient, -.2 * gradient)
    ascent = actual_update_alignment(gradient, .2 * gradient)
    assert descent['predicted_local_query_improvement'] == 1.
    np.testing.assert_allclose(descent['descent_alignment_cosine'], 1.)
    assert ascent['predicted_local_query_improvement'] == -1.
    assert actual_update_alignment(gradient, np.zeros(2))['descent_alignment_cosine'] == 0.
    with pytest.raises(ValueError, match='aligned'):
        actual_update_alignment(gradient, np.zeros(3))


def test_category_probe_groups_duplicates_and_scales_only_training_folds():
    labels = np.repeat(['cat', 'dog'], 8)
    groups = np.repeat(np.arange(8), 2)
    feature = np.column_stack([np.repeat([-1., 1.], 8), np.tile(np.arange(8), 2)])
    result = grouped_category_probe(feature, labels, groups, seed=3)
    assert result['status'] == 'ok' and result['accuracy'] == 1.
    folds = np.asarray(result['fold_ids'])
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1
    assert result == grouped_category_probe(feature, labels, groups, seed=3)
    assert grouped_category_probe(feature, labels, np.repeat([0, 1], 8))['status'] == 'skipped'
    assert grouped_category_probe(feature, ['cat'] * 16, groups)['status'] == 'skipped'


def test_overlap_groups_join_shared_queries_duplicate_aliases_and_repeated_programs():
    clusters = {'a': 'a', 'b': 'b', 'c': 'c', 'c_alias': 'c',
                'd': 'd', 'e': 'e', 'f': 'f'}
    store = SimpleNamespace(get=lambda item: SimpleNamespace(duplicate_cluster_id=clusters[item]))
    records = [
        {'support_ids': ['a'], 'query_ids': ['b']},
        {'support_ids': ['c'], 'query_ids': ['b']},
        {'support_ids': ['c_alias'], 'query_ids': ['d']},
        {'support_ids': ['e'], 'query_ids': ['f']},
        {'support_ids': ['e'], 'query_ids': ['e']},
    ]
    groups, _ = _overlap_groups(records, store)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] == groups[4] and groups[0] != groups[3]


def _assets(tmp_path, model_type='ttt_kvb_full'):
    cache = tmp_path / 'cache'
    store = create_fixture_cache(cache, categories=8, drawings_per_category=24, seed=5)
    manifest = build_manifest(store, development_fraction=.25, max_points=31)
    manifest_path = tmp_path / 'manifest.json'
    save_manifest(manifest_path, manifest)
    cfg = resolve_config({
        'cache_root': str(cache), 'manifest_path': str(manifest_path),
        'model_type': model_type, 'support_count': 1, 'query_count': 1,
        'model': {'hidden_dim': 4, 'fast_dim': 2, 'fast_hidden_dim': 3,
                  'mixture_components': 2, 'max_steps': 32, 'segment_size': 16},
    })
    model_cfg = SketchModelConfig(**cfg['model'])
    params = init_sketch_params(jax.random.key(4), model_cfg)
    state = create_ttt_train_state(params, optax.adam(1e-3), jax.random.PRNGKey(3))
    checkpoint = tmp_path / 'last.pkl'
    save_checkpoint(checkpoint, state=state, step=0, config=cfg, replicated=False, extra={
        'checkpoint_type': f'quickdraw_a_{model_type}', 'cache_id': store.identifier,
        'manifest_id': manifest['identifier'], 'model_config': asdict(model_cfg),
    })
    episodes = prepare_evaluation(cache, manifest_path, tmp_path / 'episodes', tasks=4,
                                  support_count=1, query_count=1, max_steps=32, seed=13)
    return checkpoint, episodes


def _read_arrays(path):
    with np.load(Path(path) / 'features.npz', allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _alter_query_labels(episodes, destination):
    shutil.copytree(episodes, destination)
    archive_path = destination / 'episodes.npz'
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays['query.tokens'][..., :2] += .3 * arrays['query.point_mask'][..., None]
    np.savez_compressed(archive_path, **arrays)
    metadata_path = destination / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.pop('identifier')
    metadata['data_sha256'] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    metadata['identifier'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    return destination


def test_tiny_extraction_is_deterministic_and_common_probes_exclude_query_labels(tmp_path):
    checkpoint, episodes = _assets(tmp_path)
    first = extract_checkpoint_updates(checkpoint, episodes, tmp_path / 'first')
    second = extract_checkpoint_updates(checkpoint, episodes, tmp_path / 'second')
    left, right = _read_arrays(first), _read_arrays(second)
    for name in left:
        np.testing.assert_array_equal(left[name], right[name])
    assert left['final_fast_delta'].shape[0] == 4
    assert left['public_probe_previous_tokens'].shape == (8, 4)
    assert not left['public_probe_previous_tokens'].any()
    assert not left['public_probe_states'].any()
    np.testing.assert_array_equal(left['public_probe_frame'], [0., 0., 0., 1.])
    np.testing.assert_allclose(left['first_query_improvement'],
                               left['query_loss_before'] - left['query_loss_after_first_update'])
    np.testing.assert_allclose(left['local_improvement_prediction'],
                               -np.sum(left['oracle_query_gradient'] * left['actual_first_update'], axis=1), atol=1e-7)
    assert (first / 'records.json').exists() and (first / 'summary.json').exists()
    changed = _alter_query_labels(episodes, tmp_path / 'altered-episodes')
    third = extract_checkpoint_updates(checkpoint, changed, tmp_path / 'third')
    altered = _read_arrays(third)
    for name in ('raw_first_write_gradient', 'actual_first_update', 'final_fast_delta',
                 'functional_delta', 'public_read_delta'):
        np.testing.assert_array_equal(left[name], altered[name])
    assert not np.allclose(left['oracle_query_gradient'], altered['oracle_query_gradient'])
    with pytest.raises(FileExistsError):
        extract_checkpoint_updates(checkpoint, episodes, first)


def test_extraction_rejects_non_ttt_checkpoints(tmp_path):
    checkpoint, episodes = _assets(tmp_path, model_type='no_support')
    with pytest.raises(ValueError, match='TTT'):
        extract_checkpoint_updates(checkpoint, episodes, tmp_path / 'output')
