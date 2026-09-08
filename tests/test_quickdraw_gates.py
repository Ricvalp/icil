from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw.data import (
    SketchSampler, build_manifest, create_fixture_cache, save_manifest,
)
from icil_jax_rlbench.quickdraw.gates import ordinary_fast_adaptation, run_gates
from icil_jax_rlbench.quickdraw.model import SketchModelConfig, init_sketch_params
from icil_jax_rlbench.quickdraw.train import numeric_batch


def test_ordinary_support_updates_ignore_query_targets_and_leave_slow_params_fixed(tmp_path):
    store = create_fixture_cache(tmp_path / 'cache', categories=8, drawings_per_category=24)
    manifest = build_manifest(store, max_points=31)
    sampler = SketchSampler(store, manifest, support_count=1, query_count=1, max_steps=32, seed=9)
    batch = numeric_batch(sampler.build_batch(1))
    cfg = SketchModelConfig(hidden_dim=4, fast_dim=2, fast_hidden_dim=3,
                            max_steps=32, mixture_components=2, segment_size=16)
    params = init_sketch_params(jax.random.key(0), cfg)
    before = jax.tree_util.tree_map(lambda value: np.array(value), params)
    normal = ordinary_fast_adaptation(params, batch, cfg, steps=3, learning_rate=.1)
    changed = {**batch, 'query': {**batch['query'],
               'tokens': batch['query']['tokens'].at[..., :2].add(.3)}}
    altered = ordinary_fast_adaptation(params, changed, cfg, steps=3, learning_rate=.1)
    for key in ('support_loss_before', 'support_loss_after', 'support_loss_before_step',
                'gradient_norm', 'update_norm', 'fast_delta_norm'):
        np.testing.assert_array_equal(normal[key], altered[key])
    assert not np.array_equal(normal['query_loss_after'], altered['query_loss_after'])
    for left, right in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(params)):
        np.testing.assert_array_equal(left, right)
    assert np.any(normal['functional_delta_norm'] > 0)
    with pytest.raises(ValueError, match='positive'):
        ordinary_fast_adaptation(params, batch, cfg, steps=0)


def test_bounded_fixed_batch_gate_writes_honest_fixture_artifacts(tmp_path):
    cache = tmp_path / 'cache'
    store = create_fixture_cache(cache, categories=8, drawings_per_category=24, seed=4)
    manifest = build_manifest(store, max_points=31)
    manifest_path = tmp_path / 'manifest.json'
    save_manifest(manifest_path, manifest)
    output = run_gates(cache, manifest_path, tmp_path / 'gate', steps=4,
                       tasks=1, support_count=1, query_count=1, adaptation_steps=2,
                       model_config={'hidden_dim': 4, 'fast_dim': 2, 'fast_hidden_dim': 3},
                       train_no_support_reference=False)
    report = json.loads((output / 'report.json').read_text())
    assert report['synthetic_fixture']
    assert report['fixed_meta_batch']['final_query_loss'] < report['fixed_meta_batch']['initial_query_loss']
    assert report['fixed_meta_batch']['passed']
    assert report['ordinary_fast_adaptation']['slow_parameters_unchanged']
    assert not report['ordinary_fast_adaptation']['query_used_for_updates_or_selection']
    assert (output / 'fixed_batch.npz').exists()
    with np.load(output / 'ordinary_fast_adaptation.npz', allow_pickle=False) as archive:
        np.testing.assert_allclose(archive['query_loss_improvement'],
                                   archive['query_loss_before'] - archive['query_loss_after'])
        expected = bool(archive['query_loss_improvement'].mean() > 1e-8)
        assert report['ordinary_fast_adaptation']['independent_query_improved'] == expected
