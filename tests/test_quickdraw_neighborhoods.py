from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from icil_jax_rlbench.quickdraw.data import (
    SketchSampler, _digest, build_manifest, create_fixture_cache, validate_manifest,
)
from icil_jax_rlbench.quickdraw.neighborhoods import (
    build_neighborhood_manifest, cosine_neighbors,
)


@pytest.fixture
def fixture(tmp_path):
    store = create_fixture_cache(tmp_path / 'vectors', categories=4, drawings_per_category=120)
    rng = np.random.default_rng(83)
    raw = np.zeros((len(store), 8), np.float32)
    for index, record in enumerate(store.records):
        # Six well-separated shape cells in every familiar category.
        raw[index, int(record.base_id.rsplit('/', 1)[1]) % 6] = 1
    raw += rng.normal(0, .01, raw.shape).astype(np.float32)
    cosine = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    features = SimpleNamespace(
        records=tuple({'base_id': record.base_id, 'category': record.category,
                       'duplicate_cluster_id': record.duplicate_cluster_id} for record in store.records),
        raw=raw, cosine=cosine, identifier='fixture-features', cache_id=store.identifier,
        manifest={'feature_definition': 'synthetic_fixture_only', 'checkpoint_hash': 'fixture'},
    )
    return store, features


def _manifest(fixture, **kwargs):
    store, features = fixture
    defaults = dict(family_count=2, neighborhood_count=4, evaluation_neighborhood_count=4,
                    neighborhood_size=4, top_m=6, reference_per_neighborhood=2,
                    reference_per_category=2, region_count=6)
    defaults.update(kwargs)
    return build_neighborhood_manifest(store, features, **defaults)


def test_exact_neighbors_match_normalized_squared_l2_and_stable_ties():
    vectors = np.asarray([[1, 0], [.8, .6], [.8, -.6], [0, 1], [1, 0]], np.float32)
    ids = ['query', 'z', 'a', 'far', 'copy']
    clusters = dict(zip(ids, ['q', 'z', 'a', 'far', 'q']))
    scientific, scores = cosine_neighbors(vectors[0], ids, vectors, exclude_ids=['query'],
                                          duplicate_clusters=clusters, exclude_clusters=['q'], count=2)
    assert scientific == ['a', 'z'] and scores[0] == scores[1]
    legacy, _ = cosine_neighbors(vectors[0], ids, vectors, exclude_ids=['query'], count=2)
    assert legacy == ['copy', 'a']
    eligible = [1, 2, 3]
    expected = sorted(eligible, key=lambda i: (float(np.sum((vectors[i].astype(np.float64) - vectors[0]) ** 2)), ids[i]))
    assert scientific == [ids[index] for index in expected[:2]]
    with pytest.raises(ValueError, match='normalized'):
        cosine_neighbors(vectors[0] * 2, ids, vectors)


@pytest.mark.parametrize('regime', ['familiar_drawings', 'heldout_regions', 'heldout_categories'])
def test_split_local_members_references_and_duplicate_boundaries(fixture, regime):
    store, _ = fixture
    # Held-out category splits have one category each, so use a divisible cap.
    manifest = _manifest(fixture, split_regime=regime)
    validate_manifest(store, manifest)
    seen = set()
    for split, pools in manifest['split_pools'].items():
        for pool in pools.values():
            ids = pool['example_ids'] + pool['reference_ids'] + pool['anchor_ids']
            clusters = {store.get(item).duplicate_cluster_id for item in ids}
            assert len(clusters) == len(ids) and not clusters & seen
            seen.update(clusters)
        sampler = SketchSampler(store, manifest, split=split, support_count=3, seed=23)
        batch = sampler.build_batch(4)
        for task in batch['meta']['tasks']:
            specification = manifest['a_tasks'][split][task['task_id']]
            assert task['query_ids'] == [specification['target_id']]
            assert task['support_ids'] == specification['neighbor_ids'][:3]
            assert set(task['support_ids']).isdisjoint(task['query_ids'])
            assert task['a_protocol'] == 'a_nn' and task['split'] == split
        assert not ({'category', 'task_id', 'anchor_id', 'embedding', 'neighbor_scores'} & set(batch['query']))
    if regime == 'heldout_regions':
        for category, regions in manifest['region_construction'].items():
            for split in ('train', 'development', 'test'):
                union = {item for region in regions['region_splits'][split] for item in regions['members'][region]}
                assert union == set(manifest['policy_split_ids'][split][category])


def test_local_N_sweep_is_nested_balanced_fixed_reservoir_and_fixed_evaluation(fixture):
    store, _ = fixture
    small = _manifest(fixture, a_protocol='a_local', neighborhood_count=4)
    large = _manifest(fixture, a_protocol='a_local', neighborhood_count=8)
    assert small['a_ids'] == large['a_ids']
    assert set(small['a_tasks']['train']) < set(large['a_tasks']['train'])
    for task_id, task in small['a_tasks']['train'].items():
        assert task == large['a_tasks']['train'][task_id]
    assert small['a_tasks']['test'] == large['a_tasks']['test']
    assert small['a_tasks']['development'] == large['a_tasks']['development']
    assert small['budget']['empirical_task_category_probability'] == large['budget']['empirical_task_category_probability']
    assert small['categories']['train'] == large['categories']['train']
    sampler = SketchSampler(store, large, support_count=2, query_count=2, seed=11)
    snapshot = sampler.state_dict()
    batch = sampler.build_batch(5)
    for task in batch['meta']['tasks']:
        assert task['anchor_id'] not in task['support_ids'] + task['query_ids']
        assert len(set(task['support_ids'] + task['query_ids'])) == 4
        references = large['a_tasks']['train'][task['task_id']]['reference_ids']
        assert not set(task['support_ids'] + task['query_ids']) & set(references['real_a'] + references['real_b'])
    sampler.load_state_dict(snapshot)
    replay = sampler.build_batch(5)
    assert replay['meta'] == batch['meta']
    np.testing.assert_array_equal(replay['support']['tokens'], batch['support']['tokens'])
    halves = [set(item for task in large['a_tasks']['train'].values() for item in task['reference_ids'][half])
              for half in ('real_a', 'real_b')]
    assert not halves[0] & halves[1]


def test_sample_top_m_uses_distinct_neighbors_and_replays(fixture):
    store, _ = fixture
    manifest = _manifest(fixture, selection_mode='sample_top_m')
    sampler = SketchSampler(store, manifest, support_count=3, seed=91)
    task_id = sampler.task_ids[0]
    allowed = set(sampler.tasks[task_id]['neighbor_ids'])
    sets = []
    for _ in range(12):
        task = sampler.build_batch(1, task_ids=[task_id])['meta']['tasks'][0]
        assert len(set(task['support_ids'])) == 3 and set(task['support_ids']) <= allowed
        sets.append(tuple(task['support_ids']))
    assert len(set(sets)) > 1
    with pytest.raises(ValueError, match='query_count'):
        SketchSampler(store, manifest, query_count=2)


def test_subset_seed_changes_only_training_neighborhood_choice_and_metadata_is_unprivileged(fixture):
    store, _ = fixture
    first = _manifest(fixture, a_protocol='a_local', family_count=1, neighborhood_count=2,
                      evaluation_neighborhood_count=2, unique_budget=30, subset_seed=3)
    second = _manifest(fixture, a_protocol='a_local', family_count=1, neighborhood_count=2,
                       evaluation_neighborhood_count=2, unique_budget=30, subset_seed=8)
    assert first['categories'] == second['categories']
    assert first['a_ids'] == second['a_ids']
    assert first['a_tasks']['test'] == second['a_tasks']['test']
    assert set(first['a_tasks']['train']) != set(second['a_tasks']['train'])
    changed = deepcopy(first)
    task_id, task = next(iter(changed['a_tasks']['train'].items()))
    anchors = changed['split_pools']['train'][task['category']]['anchor_ids']
    task['anchor_id'] = next(item for item in anchors if item != task['anchor_id'])
    changed['identifier'] = _digest({key: value for key, value in changed.items() if key != 'identifier'})
    left = SketchSampler(store, first, support_count=2, seed=63).build_batch(1, task_ids=[task_id])
    right = SketchSampler(store, changed, support_count=2, seed=63).build_batch(1, task_ids=[task_id])
    assert left['meta']['tasks'][0]['anchor_id'] != right['meta']['tasks'][0]['anchor_id']
    for section in ('support', 'query'):
        for key in left[section]:
            np.testing.assert_array_equal(left[section][key], right[section][key])


def test_tampered_split_membership_and_unsorted_neighbors_are_rejected(fixture):
    store, _ = fixture
    original = _manifest(fixture)
    bad = deepcopy(original)
    task = next(iter(bad['a_tasks']['train'].values()))
    other = next(iter(bad['a_tasks']['test'].values()))['target_id']
    task['member_ids'][0] = other
    bad['identifier'] = _digest({key: value for key, value in bad.items() if key != 'identifier'})
    with pytest.raises(ValueError, match='split-local'):
        validate_manifest(store, bad)
    bad = deepcopy(original)
    task = next(iter(bad['a_tasks']['train'].values()))
    task['neighbor_ids'].reverse()
    task['neighbor_scores'].reverse()
    bad['identifier'] = _digest({key: value for key, value in bad.items() if key != 'identifier'})
    with pytest.raises(ValueError, match='tie policy'):
        validate_manifest(store, bad)


def test_category_ablation_and_B_preserve_original_contracts(fixture):
    store, _ = fixture
    category = _manifest(fixture, a_protocol='a_category', neighborhood_count=2,
                         evaluation_neighborhood_count=2)
    nearest = _manifest(fixture)
    assert category['a_ids'] == nearest['a_ids']
    assert category['curation']['policy_inputs_include_classifier_features'] is False
    sampler = SketchSampler(store, category, support_count=2)
    assert sampler.build_batch(1)['meta']['tasks'][0]['a_protocol'] == 'a_category'
    legacy = build_manifest(store, family_count=2, reference_per_category=2)
    assert category['b_ids'] == legacy['b_ids']
    familiar = SketchSampler(store, category, experiment='b1', split='test')
    heldout = SketchSampler(store, category, experiment='b1', split='test', b_partition='heldout_category')
    assert set(familiar.categories).isdisjoint(heldout.categories)


def test_rejects_infeasible_or_incoherent_neighborhood_without_fallback(fixture):
    with pytest.raises(ValueError, match='only|No eligible'):
        _manifest(fixture, neighborhood_size=1000, a_protocol='a_local')
    with pytest.raises(ValueError, match='only|No eligible'):
        _manifest(fixture, min_neighbor_cosine=1)
    with pytest.raises(ValueError, match='multiple'):
        _manifest(fixture, neighborhood_count=3)


def test_per_neighborhood_allowance_grows_member_union_without_changing_tasks(fixture):
    store, _ = fixture
    small = _manifest(fixture, a_protocol='a_local', budget_regime='per_neighborhood', neighborhood_count=4)
    large = _manifest(fixture, a_protocol='a_local', budget_regime='per_neighborhood', neighborhood_count=8)
    fixed = _manifest(fixture, a_protocol='a_local', neighborhood_count=4)
    assert small['categories'] == large['categories']
    assert small['a_tasks'] == fixed['a_tasks']
    assert small['a_tasks']['test'] == large['a_tasks']['test']
    assert small['a_tasks']['development'] == large['a_tasks']['development']
    assert small['construction_candidate_ids'] == large['construction_candidate_ids']
    assert set(small['a_tasks']['train']) < set(large['a_tasks']['train'])
    for task_id, task in small['a_tasks']['train'].items():
        assert task == large['a_tasks']['train'][task_id]
    unions = []
    for manifest in (small, large):
        validate_manifest(store, manifest)
        members = {item for task in manifest['a_tasks']['train'].values() for item in task['member_ids']}
        available = {item for ids in manifest['a_ids']['train'].values() for item in ids}
        assert available == members
        assert manifest['budget']['unique_a_train_drawings'] == len(members)
        assert manifest['budget']['regime'] == 'diversity_plus_data'
        assert manifest['budget']['train_neighborhood_member_incidences'] == manifest['budget']['neighborhood_count'] * 4
        assert all(len(task['member_ids']) == 4 for task in manifest['a_tasks']['train'].values())
        reserved = {item for pool in manifest['split_pools']['train'].values()
                    for role in ('anchor_ids', 'reference_ids') for item in pool[role]}
        assert not reserved & available
        assert len(available) <= manifest['budget']['construction_example_candidate_count']
        sampler = SketchSampler(store, manifest, support_count=2, seed=17)
        episode = sampler.build_batch(1)['meta']['tasks'][0]
        assert set(episode['support_ids'] + episode['query_ids']) <= available
        unions.append(available)
    assert unions[0] < unions[1]
    with pytest.raises(ValueError, match='requires A-local'):
        _manifest(fixture, budget_regime='per_neighborhood')
    with pytest.raises(ValueError, match='no global/category'):
        _manifest(fixture, a_protocol='a_local', budget_regime='per_neighborhood', unique_budget=40)
