from copy import deepcopy
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw.data import SketchSampler, create_fixture_cache, save_manifest
from icil_jax_rlbench.quickdraw.neighborhoods import build_neighborhood_manifest
from icil_jax_rlbench.quickdraw.pilot import _category_ablation_manifest
from icil_jax_rlbench.quickdraw.train import numeric_batch, train
from icil_jax_rlbench.train.checkpoints import load_checkpoint


@pytest.fixture
def neighborhood_assets(tmp_path):
    store = create_fixture_cache(tmp_path/'cache', categories=4, drawings_per_category=128)
    rng = np.random.default_rng(91)
    raw = rng.normal(size=(len(store), 8)).astype(np.float32)
    features = SimpleNamespace(
        records=[{'base_id':record.base_id, 'category':record.category,
                  'duplicate_cluster_id':record.duplicate_cluster_id} for record in store.records],
        cosine=raw/np.linalg.norm(raw, axis=1, keepdims=True),
        cache_id=store.identifier, identifier='explicit_synthetic_fixture',
        manifest={'synthetic_fixture':True},
    )
    return store, features


def _manifest(assets, protocol):
    return build_neighborhood_manifest(
        *assets, a_protocol=protocol, family_count=2, neighborhood_count=4,
        evaluation_neighborhood_count=4, top_m=8, neighborhood_size=6,
        reference_per_neighborhood=2, reference_per_category=2, max_points=15,
    )


def test_matched_category_ablation_preserves_NN_targets_and_references(neighborhood_assets):
    store, features = neighborhood_assets
    nearest = _manifest(neighborhood_assets, 'a_nn')
    category = _category_ablation_manifest(store, nearest, features)
    assert category['a_ids'] == nearest['a_ids']
    for split in ('train', 'development', 'test'):
        tasks = sorted(nearest['a_tasks'][split])
        batches = [SketchSampler(store, manifest, split=split, support_count=2, max_steps=16,
                                 seed=29).build_batch(len(tasks), task_ids=tasks)
                   for manifest in (nearest, category)]
        for field in batches[0]['query']:
            np.testing.assert_array_equal(batches[0]['query'][field], batches[1]['query'][field])
        for task in tasks:
            assert category['a_tasks'][split][task]['reference_ids'] == nearest['a_tasks'][split][task]['reference_ids']
        for record in batches[1]['meta']['tasks']:
            assert set(record['support_ids']).isdisjoint(record['query_ids'])


def test_policy_batch_rejects_offline_features_but_ignores_private_metadata(neighborhood_assets):
    store, _ = neighborhood_assets
    batch = SketchSampler(store, _manifest(neighborhood_assets, 'a_nn'), support_count=2, max_steps=16).build_batch(1)
    numeric = numeric_batch(batch)
    changed = deepcopy(batch)
    changed['meta']['tasks'][0].update(anchor_id='another-anchor', query_embedding=[99.0]*512,
                                      intended_neighborhood_id='another-neighborhood')
    other = numeric_batch(changed)
    for role in numeric:
        for field in numeric[role]:
            np.testing.assert_array_equal(numeric[role][field], other[role][field])
    changed['query']['embedding'] = np.zeros((1,512), np.float32)
    with pytest.raises(ValueError, match='non-trajectory'):
        numeric_batch(changed)


def test_local_TTT_resume_preserves_neighborhood_exposure_and_pairs(tmp_path, neighborhood_assets):
    manifest = _manifest(neighborhood_assets, 'a_local')
    path = tmp_path/'local.json'
    save_manifest(path, manifest)
    cfg = dict(cache_root=str(tmp_path/'cache'), manifest_path=str(path), a_protocol='a_local',
               batch_size=1, support_count=2, query_count=1, seed=11,
               checkpoint_every=1, log_every=1,
               model=dict(hidden_dim=4, fast_dim=2, fast_hidden_dim=3,
                          mixture_components=2, segment_size=16, max_steps=16))
    full = train({**cfg, 'num_steps':2, 'output_dir':str(tmp_path/'full')})
    first = train({**cfg, 'num_steps':1, 'output_dir':str(tmp_path/'first')})
    resumed = train({**cfg, 'num_steps':2, 'output_dir':str(tmp_path/'resumed'),
                     'resume_path':str(first/'last.pkl')})
    expected, actual = (load_checkpoint(directory/'last.pkl') for directory in (full,resumed))
    for field in ('params','opt_state','rng'):
        for left, right in zip(jax.tree_util.tree_leaves(expected[field]), jax.tree_util.tree_leaves(actual[field])):
            np.testing.assert_array_equal(left, right)
    for field in ('exposure','observed_ids','reuse_counts','sampler_state'):
        assert expected['extra'][field] == actual['extra'][field]
    assert actual['config']['a_protocol'] == 'a_local'
    reuse = actual['extra']['reuse_counts']
    assert sum(reuse['tasks'].values()) == 2
    assert sum(reuse['support_drawings'].values()) == 4
    assert sum(reuse['query_drawings'].values()) == 2
    assert sum(reuse['support_target_pairs'].values()) == 4
    assert actual['extra']['exposure']['observed_unique_neighborhoods'] >= 1
    assert not actual['extra']['transient_fast_state_saved']
    jax.clear_caches()
