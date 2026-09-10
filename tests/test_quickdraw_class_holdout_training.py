"""Class-disjoint policy training keeps validation/test drawings separate."""

from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import class_split
from icil_jax_rlbench.quickdraw import supervised_train as training
from icil_jax_rlbench.quickdraw.policy_backend import model_config
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from test_quickdraw_supervised_training import _config, dataset, wandb_calls


def test_class_split_is_category_stable_and_independent_of_training_seed():
    names = tuple(f'category-{index:03d}' for index in range(345))
    data = SimpleNamespace(identifier='same-vector-dataset', categories=names)
    cfg = {'heldout_category_count': 35, 'heldout_category_seed': 37, 'seed': 0}
    first = class_split.resolve_class_split(data, cfg)
    repeated = class_split.resolve_class_split(data, {**cfg, 'seed': 123})
    assert first == repeated
    assert len(first['training_category_ids']) == 310
    assert len(first['heldout_category_ids']) == 35
    assert set(first['training_category_ids']).isdisjoint(first['heldout_category_ids'])
    assert sorted(first['training_category_ids'] + first['heldout_category_ids']) == list(range(345))
    assert first['evaluation_category_ids'] == first['heldout_category_ids']
    assert first['training_categories'] == [names[index] for index in first['training_category_ids']]
    assert first['heldout_categories'] == [names[index] for index in first['heldout_category_ids']]
    reordered = class_split.resolve_class_split(
        SimpleNamespace(identifier=data.identifier, categories=tuple(reversed(names))), cfg)
    assert set(reordered['heldout_categories']) == set(first['heldout_categories'])
    different = class_split.resolve_class_split(data, {**cfg, 'heldout_category_seed': 38})
    assert different['identifier'] != first['identifier']
    assert different['heldout_categories'] != first['heldout_categories']


@pytest.mark.parametrize('count', [-1, 345, 346, 1.5, True])
def test_invalid_class_holdout_count_rejected(count):
    data = SimpleNamespace(identifier='fixture', categories=tuple(map(str, range(345))))
    with pytest.raises(ValueError, match='heldout_category_count'):
        class_split.resolve_class_split(data, {
            'heldout_category_count': count, 'heldout_category_seed': 37})


@pytest.mark.parametrize('seed', [-1, 2 ** 32, 1.5, True])
def test_invalid_class_holdout_seed_rejected(seed):
    with pytest.raises(ValueError, match='heldout_category_seed'):
        training.resolve_config({'heldout_category_count': 1, 'heldout_category_seed': seed})


def test_split_rows_preserve_drawings_and_neighbor_boundaries(dataset):
    cfg = {'heldout_category_count': 1, 'heldout_category_seed': 37}
    manifest = class_split.resolve_class_split(dataset, cfg)
    seen, heldout = manifest['training_category_ids'], manifest['heldout_category_ids']
    selected = {}
    for split in ('train', 'development', 'test'):
        expected_categories = seen if split == 'train' else heldout
        rows = class_split.split_rows(dataset, cfg, split)
        selected[split] = set(rows.tolist())
        original = dataset.rows(split)
        np.testing.assert_array_equal(rows, original[np.isin(dataset.category_ids[original], expected_categories)])
        assert set(dataset.category_ids[rows]) == set(expected_categories)
        assert set(dataset.category_ids[dataset.neighbors[rows]].ravel()) == set(expected_categories)
        assert np.all(dataset.split_codes[dataset.neighbors[rows]] == dataset.split_codes[rows, None])
        refs = class_split.split_rows(dataset, cfg, split, reference=True)
        assert set(dataset.category_ids[refs]) == set(expected_categories)
        assert selected[split].isdisjoint(refs)
    assert selected['train'].isdisjoint(selected['development'] | selected['test'])
    assert selected['development'].isdisjoint(selected['test'])
    for scope, expected in (('auto', heldout), ('heldout', heldout), ('seen', seen), ('all', [0, 1])):
        assert list(class_split.evaluation_category_ids(dataset, cfg, scope)) == expected
    np.testing.assert_array_equal(class_split.split_rows(dataset, cfg, 'validation'),
                                  class_split.split_rows(dataset, cfg, 'development'))
    disabled = {'heldout_category_count': 0, 'heldout_category_seed': 37}
    for split in ('train', 'development', 'test'):
        np.testing.assert_array_equal(class_split.split_rows(dataset, disabled, split), dataset.rows(split))
    assert list(class_split.evaluation_category_ids(dataset, disabled)) == [0, 1]


def test_validation_cap_applies_only_to_heldout_classes(dataset, tmp_path):
    cfg = {**_config(dataset, tmp_path / 'unused'), 'heldout_category_count': 1,
           'heldout_category_seed': 37, 'validation_examples_per_category': 1}
    expected = class_split.resolve_class_split(dataset, cfg)['heldout_category_ids']
    rows = training.validation_rows(dataset, cfg)
    assert len(rows) == 1
    assert list(dataset.category_ids[rows]) == expected
    assert set(rows).issubset(dataset.rows('development'))
    np.testing.assert_array_equal(training.validation_rows(dataset, cfg), rows)


def test_saved_class_split_is_required_and_matches_restored_config(dataset):
    cfg = {'heldout_category_count': 1, 'heldout_category_seed': 37}
    manifest = class_split.resolve_class_split(dataset, cfg)
    assert class_split.validate_saved_class_split(manifest, dataset, cfg) == manifest
    with pytest.raises(ValueError, match='class split'):
        class_split.validate_saved_class_split(None, dataset, cfg)
    with pytest.raises(ValueError, match='class split'):
        class_split.validate_saved_class_split(manifest, dataset, {**cfg, 'heldout_category_seed': 38})
    disabled = {**cfg, 'heldout_category_count': 0}
    assert class_split.validate_saved_class_split(None, dataset, disabled)['heldout_category_count'] == 0


@pytest.mark.parametrize('policy', ['autoregressive', 'diffusion', 'kvb', 'support_bc'])
def test_class_disjoint_epoch_resume_and_validation(dataset, tmp_path, policy, wandb_calls):
    architecture = 'autoregressive' if policy in ('kvb', 'support_bc') else policy
    cfg = {**_config(dataset, tmp_path / 'continuous', architecture),
           'heldout_category_count': 1, 'heldout_category_seed': 37,
           'support_count': 2, 'plot_every': 0}
    if policy in ('kvb', 'support_bc'):
        cfg.update(method=policy)
        cfg['model'] = asdict(model_config({**cfg['model'], 'fast_dim': 3,
                                            'fast_hidden_dim': 5, 'inner_steps': 3}, policy))
    manifest = class_split.resolve_class_split(dataset, cfg)
    train_rows = class_split.split_rows(dataset, cfg, 'train')
    validation_rows = class_split.split_rows(dataset, cfg, 'development')
    epoch_steps = math.ceil(len(train_rows) / cfg['batch_size'])
    assert epoch_steps >= 2 and len(train_rows) % cfg['batch_size']
    continuous = load_checkpoint(training.train(cfg))
    resumed_cfg = {**cfg, 'output_dir': str(tmp_path / 'resumed'),
                   'plot_every': 2, 'plot_examples': 1}
    interrupted_path = training.train({**resumed_cfg, 'max_steps': 1})
    resumed_path = training.train({**resumed_cfg, 'resume_path': str(interrupted_path)})
    resumed = load_checkpoint(resumed_path)
    for name in ('params', 'opt_state', 'rng'):
        for left, right in zip(jax.tree.leaves(continuous[name]), jax.tree.leaves(resumed[name])):
            np.testing.assert_array_equal(left, right)
    assert continuous['step'] == resumed['step'] == epoch_steps
    extra = resumed['extra']
    assert extra['next_epoch'] == 1 and extra['next_batch'] == 0
    assert extra['class_split'] == manifest
    output = Path(resumed_cfg['output_dir'])
    assert json.loads((output / 'class_split.json').read_text()) == manifest
    exposure = extra['exposure']
    np.testing.assert_array_equal(exposure['targets'][train_rows], 1)
    assert exposure['targets'].sum() == len(train_rows)
    assert exposure['supports'].sum() == cfg['support_count'] * len(train_rows)
    excluded = np.flatnonzero(np.isin(dataset.category_ids, manifest['heldout_category_ids']))
    for name in ('targets', 'supports', 'pairs'):
        assert not np.any(exposure[name][excluded])
        np.testing.assert_array_equal(exposure[name], continuous['extra']['exposure'][name])
    for name in ('targets', 'supports'):
        assert not np.any(exposure[name][dataset.rows('development')])
        assert not np.any(exposure[name][dataset.rows('test')])
    validation = [row for row in extra['history'] if 'validation/loss' in row]
    assert len(validation) == 1
    assert validation[0]['validation/sample_count'] == len(validation_rows)
    assert np.isfinite(validation[0]['validation/loss'])
    selection = json.loads((output / 'plots/selection.json').read_text())
    for field in ('target_rows', 'support_rows'):
        rows = np.asarray(selection[field], dtype=np.int32).ravel()
        assert set(rows).issubset(validation_rows)
        assert set(dataset.category_ids[rows]) == set(manifest['heldout_category_ids'])
    images = [row for run in wandb_calls for row in run.rows
              if 'samples/context_and_generated' in row]
    assert len(images) == epoch_steps // resumed_cfg['plot_every']
    assert all(Path(row['samples/context_and_generated'].path).is_file() for row in images)
    _, loaded_cfg, _, _ = training.load_run(resumed_path)
    assert loaded_cfg['heldout_category_count'] == 1
    assert loaded_cfg['heldout_category_seed'] == 37
    for changed in ({'heldout_category_count': 0}, {'heldout_category_seed': 38}):
        with pytest.raises(ValueError, match='identical|[Cc]lass|heldout_category'):
            training.train({**resumed_cfg, 'epochs': 2, 'resume_path': str(resumed_path), **changed})
