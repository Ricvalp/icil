"""Heldout-class figures and matched real/generated FID populations."""
from __future__ import annotations

import json

import jax
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import (
    policy_backend, supervised_evaluate as controls, supervised_fid as fid,
    supervised_plots as plots, supervised_train as training,
    supervised_visualize as figures,
)
from icil_jax_rlbench.quickdraw.class_split import resolve_class_split
from icil_jax_rlbench.quickdraw.metrics import load_trajectories
from test_quickdraw_supervised_evaluate import _prepare
from test_quickdraw_supervised_fid import _reference


def _policy(dataset, method='icil', architecture='autoregressive'):
    cfg = dict(architecture=architecture, hidden_dim=8, num_heads=2,
        context_layers=1, decoder_layers=1, mlp_ratio=2, max_steps=dataset.max_steps,
        mixture_components=2, dropout=0., diffusion_steps=3, dtype='float32')
    if method == 'kvb':
        cfg.update(fast_dim=4, fast_hidden_dim=6, inner_steps=3)
    model = policy_backend.model_config(cfg, method)
    params = policy_backend.init_model(jax.random.PRNGKey(42), model, support_count=2)
    run = {'method': method, 'support_count': 2, 'selection_mode': 'exact_top_k',
           'condition_on_support': True, 'heldout_category_count': 1, 'heldout_category_seed': 37}
    return run, model, params


def _checkpoint(tmp_path, monkeypatch, dataset, run, model, params):
    path = tmp_path / 'checkpoint.pkl'
    path.write_bytes(b'synthetic-holdout-checkpoint')
    monkeypatch.setattr(training, 'load_run', lambda *args, **kwargs:
        ({'params': params, 'step': 10000}, run, model, dataset))
    return path


def test_fid_subset_keeps_features_and_records_aligned_and_parent_immutable(tmp_path, monkeypatch):
    dataset, _, parent = _reference(tmp_path, monkeypatch)
    original_metadata = json.dumps(parent.metadata, sort_keys=True)
    assert fid.select_reference_categories(parent, dataset, [1, 0]) is parent
    selected = fid.select_reference_categories(parent, dataset, [1])
    assert selected.identifier != parent.identifier
    assert selected.metadata['parent_reference_id'] == parent.identifier
    assert selected.metadata['categories'] == ['dog']
    assert selected.metadata['category_ids'] == [1]
    assert len(selected.target_rows) == len(selected.reference_rows) == len(selected.features) == 2
    expected = dataset.category_ids[parent.reference_rows] == 1
    np.testing.assert_array_equal(selected.features, parent.features[expected])
    np.testing.assert_array_equal(selected.reference_rows, parent.reference_rows[expected])
    np.testing.assert_array_equal(selected.target_rows,
        parent.target_rows[dataset.category_ids[parent.target_rows] == 1])
    assert selected.feature_metadata['actual_count'] == selected.feature_metadata['expected_count'] == 2
    assert selected.feature_metadata['evaluation_id'] == selected.metadata['selection_id']
    for record, row in zip(selected.feature_metadata['records'], selected.reference_rows):
        assert record['drawing_id'] == str(dataset.base_ids[row])
        assert record['intended_category'] == 'dog'
    assert json.dumps(parent.metadata, sort_keys=True) == original_metadata
    assert len(parent.feature_metadata['records']) == 4
    assert fid.select_reference_categories(parent, dataset, [1]).identifier == selected.identifier
    assert fid.select_reference_categories(selected, dataset, [1]) is selected
    for ids in ([], [0, 0], [-1], [2], [0.5], [True]):
        with pytest.raises(ValueError, match='category IDs'):
            fid.select_reference_categories(parent, dataset, ids)
    with pytest.raises(ValueError, match='absent'):
        fid.select_reference_categories(selected, dataset, [0])


def test_training_plots_and_test_galleries_use_same_heldout_classes_different_drawings(tmp_path, monkeypatch):
    dataset, _, _ = _reference(tmp_path, monkeypatch)
    split = resolve_class_split(dataset, {'heldout_category_count': 1, 'heldout_category_seed': 37})
    ids = split['heldout_category_ids']
    kwargs = dict(count=4, support_count=2, selection_mode='exact_top_k', seed=2027)
    default = plots.prepare_plot_batch(dataset, **kwargs)
    explicit_none = plots.prepare_plot_batch(dataset, category_ids=None, **kwargs)
    assert default['metadata'] == explicit_none['metadata']
    assert 'evaluation_category_ids' not in default['metadata']
    selected = plots.prepare_plot_batch(dataset, category_ids=ids, **kwargs)
    targets = np.asarray(selected['metadata']['target_rows'])
    support = np.asarray(selected['metadata']['support_rows'])
    assert set(dataset.category_ids[targets]) == set(ids)
    assert set(dataset.category_ids[support].ravel()) == set(ids)
    assert set(targets) <= set(dataset.rows('development'))
    assert not set(targets) & set(dataset.rows('test'))
    assert np.all(targets[:, None] != support)
    test = figures.select_gallery_targets(dataset, count=4, split='test', seed=2027, category_ids=ids)
    assert set(dataset.category_ids[test]) == set(ids)
    assert not set(test) & set(targets)


@pytest.mark.parametrize('method,architecture', [
    ('icil', 'autoregressive'), ('icil', 'diffusion'), ('kvb', 'autoregressive')])
def test_checkpoint_figures_and_fid_default_to_heldout_categories(tmp_path, monkeypatch, method, architecture):
    dataset, resources, reference = _reference(tmp_path, monkeypatch)
    run, model, params = _policy(dataset, method, architecture)
    checkpoint = _checkpoint(tmp_path, monkeypatch, dataset, run, model, params)
    split = resolve_class_split(dataset, run)
    ids = split['heldout_category_ids']
    output = figures.visualize_checkpoint(checkpoint, tmp_path / 'figures',
        context_examples=2, grid_rows=2, grid_columns=2, batch_size=2,
        formats=('png',), dpi=50, progress=False)
    metadata = json.loads((output / 'samples.json').read_text())
    assert metadata['class_split'] == split
    assert metadata['category_scope'] == 'auto'
    assert metadata['evaluation_category_ids'] == ids
    assert {record['category_id'] for record in metadata['records']} == set(ids)
    assert all(np.all(dataset.category_ids[record['support_rows']] == record['category_id'])
               for record in metadata['records'])
    report = fid.evaluate_checkpoint(checkpoint, reference.root, tmp_path / 'fid',
        batch_size=2, progress=False, **resources)
    assert np.isfinite(report['sketch_fid'])
    assert report['category_count'] == 1
    assert report['categories'] == split['heldout_categories']
    assert report['generated_count'] == report['reference_count'] == 2
    assert report['class_split_id'] == split['identifier']
    assert report['parent_reference_id'] == reference.identifier
    generated = json.loads((tmp_path / 'fid/generation.json').read_text())
    assert {record['category_id'] for record in generated['records']} == set(ids)
    assert all(record['target_row'] not in record['support_rows'] for record in generated['records'])
    with pytest.raises(ValueError, match='reference categories'):
        fid.evaluate_params(params, model, dataset, reference, tmp_path / 'wrong-population',
            support_count=2, selection_mode='exact_top_k', class_split=split, progress=False, **resources)
    if method == 'icil' and architecture == 'autoregressive':
        seen = fid.evaluate_checkpoint(checkpoint, reference.root, tmp_path / 'fid-seen',
            category_scope='seen', batch_size=2, progress=False, **resources)
        assert seen['categories'] == split['training_categories']
        assert seen['reference_id'] != report['reference_id']
        test_reference = fid.prepare_reference(tmp_path, tmp_path / 'test-reference',
            samples_per_category=2, split='test', allow_test=True, progress=False, **resources)
        with pytest.raises(ValueError, match='allow-test'):
            fid.evaluate_checkpoint(checkpoint, test_reference, tmp_path / 'denied-test',
                batch_size=2, progress=False, **resources)
        test_report = fid.evaluate_checkpoint(checkpoint, test_reference, tmp_path / 'fid-test',
            allow_test=True, batch_size=2, progress=False, **resources)
        assert test_report['categories'] == report['categories']
        test_metadata = json.loads((tmp_path / 'fid-test/generation.json').read_text())
        assert {record['target_row'] for record in test_metadata['records']} <= set(dataset.rows('test'))


def test_legacy_controls_reject_mixed_scope_and_copy_baseline_excludes_heldout_training_rows(tmp_path, monkeypatch):
    dataset, episodes = _prepare(tmp_path, monkeypatch)
    run, model, params = _policy(dataset)
    checkpoint = _checkpoint(tmp_path, monkeypatch, dataset, run, model, params)
    with pytest.raises(ValueError, match='manifest includes categories outside'):
        controls.generate_run(checkpoint, episodes, tmp_path / 'denied', progress=False)
    split = resolve_class_split(dataset, run)
    output = controls.generate_run(checkpoint, episodes, tmp_path / 'controls',
        conditions=('nearest_training_example',), category_scope='all',
        samples_per_task=1, batch_size=2, progress=False)
    _, metadata = load_trajectories(output / 'nearest_training_example')
    indices = {str(name): row for row, name in enumerate(dataset.base_ids)}
    copied = [indices[record['copied_training_id']] for record in metadata['records']]
    assert set(dataset.category_ids[copied]) <= set(split['training_category_ids'])
    report = controls.geometry_copying(tmp_path, output / 'nearest_training_example',
        tmp_path / 'geometry.json', progress=False)
    expected = np.count_nonzero(np.isin(dataset.category_ids[dataset.rows('train')], split['training_category_ids']))
    assert report['training_population_count'] == expected
    assert set(dataset.category_ids[[indices[row['nearest_training_id']] for row in report['per_generated']]]) <= set(split['training_category_ids'])
