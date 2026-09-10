"""Preserve pre-holdout experiments and reject changing their class population."""

from dataclasses import asdict
import json
from pathlib import Path
import pickle

import jax
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import supervised_train as training
from icil_jax_rlbench.quickdraw import supervised_fid as fid
from icil_jax_rlbench.quickdraw.class_split import resolve_class_split
from icil_jax_rlbench.quickdraw.kvb_models import KVBModelConfig
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from test_quickdraw_supervised_training import _config, dataset


@pytest.mark.parametrize('method', ['icil', 'kvb'])
def test_pre_holdout_checkpoint_resumes_exactly_but_cannot_change_classes(dataset, tmp_path, method):
    cfg = {**_config(dataset, tmp_path / 'continuous'), 'method': method,
           'wandb_project': None, 'plot_every': 0, 'max_steps': 3}
    if method == 'kvb':
        cfg['model'] = asdict(KVBModelConfig(**cfg['model'], fast_dim=4, fast_hidden_dim=6))
    continuous = load_checkpoint(training.train(cfg))
    resumed_cfg = {**cfg, 'output_dir': str(tmp_path / 'resumed')}
    path = training.train({**resumed_cfg, 'max_steps': 1})
    legacy = load_checkpoint(path)
    legacy['config'].pop('heldout_category_count')
    legacy['config'].pop('heldout_category_seed')
    legacy['extra'].pop('class_split')
    sources = legacy['extra']['execution']['source_hashes']
    sources.pop('class_split.py')
    sources['supervised_train.py'] = training.PRE_CLASS_HOLDOUT_TRAINER_SHA256
    # Reproduce the actual old checkpoint schema; numerical state is unchanged.
    with path.open('wb') as handle:
        pickle.dump(legacy, handle)
    (path.parent / 'class_split.json').unlink()
    _, restored_cfg, _, _ = training.load_run(path)
    assert restored_cfg['heldout_category_count'] == 0
    with pytest.raises(ValueError, match='identical model, data sampling'):
        training.train({**resumed_cfg, 'resume_path': str(path), 'heldout_category_count': 1})
    resumed = load_checkpoint(training.train({**resumed_cfg, 'resume_path': str(path)}))
    for name in ('params', 'opt_state', 'rng'):
        for left, right in zip(jax.tree.leaves(continuous[name]), jax.tree.leaves(resumed[name])):
            np.testing.assert_array_equal(left, right)
    assert not resumed['extra']['class_split']['heldout_categories']
    assert (Path(resumed_cfg['output_dir']) / 'class_split.json').is_file()
    assert (Path(resumed_cfg['output_dir']) / 'evaluation_upgrade.json').is_file()


def test_pre_holdout_source_upgrade_only_accepts_the_exact_predecessor():
    model = KVBModelConfig(hidden_dim=8, num_heads=2)
    current = training._execution_signature(model)
    previous = {**current, 'source_hashes': {
        name: (training.PRE_CLASS_HOLDOUT_TRAINER_SHA256 if name == 'supervised_train.py' else digest)
        for name, digest in current['source_hashes'].items() if name != 'class_split.py'}}
    assert training._compatible_execution(previous, current)
    for name in ('kvb_models.py', 'fast_weight_ttt.py', 'policy_backend.py', 'full_data.py'):
        changed = {**previous, 'source_hashes': {**previous['source_hashes'], name: 'modified'}}
        assert not training._compatible_execution(changed, current)
    assert not training._compatible_execution({**previous, 'backend': 'changed'}, current)


def test_periodic_fid_filters_real_references_and_freezes_the_class_population(dataset, tmp_path, monkeypatch):
    count = len(dataset.categories)
    targets = np.asarray([dataset.rows('development')[
        dataset.category_ids[dataset.rows('development')] == category][0] for category in range(count)])
    reals = np.asarray([dataset.reference_rows('development', category)[0] for category in range(count)])
    metadata = {'identifier': 'all-categories-reference', 'dataset_identifier': dataset.identifier,
                'categories': list(dataset.categories), 'split': 'development',
                'samples_per_category': 1, 'selection_id': 'all-categories-selection'}
    records = [{'drawing_id': str(dataset.base_ids[row]),
                'intended_category': dataset.categories[int(dataset.category_ids[row])]} for row in reals]
    reference = fid.Reference(tmp_path, metadata, targets, reals,
                              np.arange(count * 512).reshape(count, 512), {'records': records})
    monkeypatch.setattr(fid, 'load_reference', lambda *args: reference)
    observed = []
    monkeypatch.setattr(fid, 'preflight', lambda subset, **kwargs: observed.append(subset.identifier))
    cfg = {**training.default_config(), 'heldout_category_count': 1, 'fid_enabled': True}
    specification = resolve_class_split(dataset, cfg)
    subset, selection = training._prepare_training_fid(cfg, dataset, tmp_path)
    assert selection['class_split_id'] == specification['identifier']
    assert observed == [selection['reference_id']] and selection['reference_id'] != reference.identifier
    assert subset.metadata['parent_reference_id'] == reference.identifier
    assert subset.metadata['categories'] == specification['heldout_categories']
    assert len(subset.target_rows) == len(subset.features) == 1
    np.testing.assert_array_equal(subset.features, reference.features[specification['heldout_category_ids']])
    (tmp_path / 'fid').mkdir()
    (tmp_path / 'fid/selection.json').write_text(json.dumps(selection))
    _, repeated = training._prepare_training_fid(cfg, dataset, tmp_path)
    assert repeated == selection
    with pytest.raises(ValueError, match='reference and seed fixed'):
        training._prepare_training_fid({**cfg, 'heldout_category_seed': 41}, dataset, tmp_path)
