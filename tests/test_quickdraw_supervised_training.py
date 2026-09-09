from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from icil_jax_rlbench.quickdraw import supervised_train as training
from icil_jax_rlbench.quickdraw.supervised_models import SupervisedModelConfig, init_model
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from test_quickdraw_full_data import _build, _sources


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp('supervised-training-data')
    _sources(root)
    return _build(root)


def _config(dataset, output, architecture='autoregressive'):
    model = SupervisedModelConfig(architecture=architecture, hidden_dim=8, num_heads=2,
        context_layers=1, decoder_layers=1, mlp_ratio=2, max_steps=5,
        mixture_components=2, dropout=.1, diffusion_steps=3, dtype='float32')
    return {**training.default_config(architecture), 'model': asdict(model),
        'dataset_root': str(dataset.root), 'output_dir': str(output), 'support_count': 1,
        'selection_mode': 'sample_top_m', 'epochs': 1, 'batch_size': 10,
        'micro_batch_size': 2, 'warmup_steps': 0, 'schedule_steps': 100,
        'log_every': 2, 'checkpoint_every': 2, 'wandb_project': 'test-project'}


@pytest.fixture
def wandb_calls(monkeypatch):
    calls = []
    module = ModuleType('wandb')

    class Run:
        def __init__(self, kwargs):
            self.id, self.entity, self.step = kwargs['id'], kwargs['entity'], 0
            previous = [run for run in calls if run.id == self.id]
            if kwargs['resume'] == 'allow' and previous:
                self.step = previous[-1].step
            self.kwargs, self.rows, self.metrics = kwargs, [], []
            self.exit_code = None

        def define_metric(self, *args, **kwargs):
            self.metrics.append((args, kwargs))

        def log(self, row, step, commit):
            assert step >= self.step
            assert commit
            self.rows.append(dict(row))
            self.step = step + 1

        def finish(self, exit_code):
            self.exit_code = exit_code

    def init(**kwargs):
        run = Run(kwargs)
        calls.append(run)
        return run

    module.init = init
    class Image:
        def __init__(self, path, caption):
            assert Path(path).read_bytes().startswith(b'\x89PNG')
            self.path, self.caption = path, caption
    module.Image = Image
    monkeypatch.setitem(sys.modules, 'wandb', module)
    return calls


@pytest.mark.parametrize('architecture', ['autoregressive', 'diffusion'])
def test_full_epoch_mid_epoch_resume_and_direct_wandb(dataset, tmp_path, architecture, wandb_calls):
    continuous_cfg = _config(dataset, tmp_path / 'continuous', architecture)
    continuous_cfg['plot_every'] = 0
    continuous = load_checkpoint(training.train(continuous_cfg))
    cfg = {**_config(dataset, tmp_path / 'resumed', architecture),
           'plot_every': 2, 'plot_examples': 2}
    interrupted_path = training.train({**cfg, 'max_steps': 3})
    interrupted = load_checkpoint(interrupted_path)
    assert interrupted['extra']['next_batch'] == 3
    assert interrupted['extra']['epoch_count'] == 3 * cfg['batch_size']
    initial_provenance = (Path(cfg['output_dir']) / 'provenance.json').read_bytes()
    selection = (Path(cfg['output_dir']) / 'plots/selection.json').read_bytes()
    resumed_path = training.train({**cfg, 'resume_path': str(interrupted_path)})
    resumed = load_checkpoint(resumed_path)
    for name in ('params', 'opt_state', 'rng'):
        for a, b in zip(jax.tree.leaves(continuous[name]), jax.tree.leaves(resumed[name])):
            np.testing.assert_array_equal(a, b)
    assert continuous['step'] == resumed['step']
    assert resumed['extra']['next_epoch'] == 1 and resumed['extra']['next_batch'] == 0
    for name, value in continuous['extra']['exposure'].items():
        np.testing.assert_array_equal(value, resumed['extra']['exposure'][name])
    exposure = resumed['extra']['exposure']
    train_rows = dataset.rows('train')
    np.testing.assert_array_equal(exposure['targets'][train_rows], 1)
    assert exposure['targets'].sum() == len(train_rows)
    assert exposure['supports'].sum() == len(train_rows) * cfg['support_count']
    assert exposure['pairs'].sum() == exposure['supports'].sum()
    assert len(train_rows) % cfg['batch_size']  # Last batch actually exercises padding.
    assert (Path(cfg['output_dir']) / 'provenance.json').read_bytes() == initial_provenance
    assert (Path(cfg['output_dir']) / 'best.pkl').is_file()
    loaded, restored_cfg, model_cfg, restored_data = training.load_run(resumed_path)
    assert loaded['step'] == resumed['step'] and model_cfg.architecture == architecture
    assert restored_data.identifier == dataset.identifier
    assert restored_cfg['schedule_steps'] == 100
    history = resumed['extra']['history']
    assert any(row.get('train/epoch_sample_count') == len(train_rows) for row in history)
    validation = [row for row in history if 'validation/loss' in row]
    assert len(validation) == 1
    assert validation[0]['validation/sample_count'] == len(dataset.rows('development'))
    assert np.isfinite(validation[0]['validation/loss'])
    assert all(run.exit_code == 0 for run in wandb_calls)
    assert wandb_calls[-1].kwargs['resume'] == 'allow'
    assert wandb_calls[-1].id == wandb_calls[-2].id
    assert any('train/loss' in row for row in wandb_calls[-1].rows)
    assert any('validation/loss' in row for row in wandb_calls[-1].rows)
    stored_rows = [json.loads(line) for line in (Path(cfg['output_dir']) / 'metrics.jsonl').read_text().splitlines()]
    assert stored_rows == history
    assert (Path(cfg['output_dir']) / 'plots/selection.json').read_bytes() == selection
    assert not (Path(continuous_cfg['output_dir']) / 'plots').exists()
    images = [row for run in wandb_calls[1:] for row in run.rows
              if 'samples/context_and_generated' in row]
    assert [row['optimizer_step'] for row in images] == list(range(2, resumed['step'] + 1, 2))
    for row in images:
        assert row['samples/example_count'] == 2
        assert Path(row['samples/context_and_generated'].path).is_file()
    assert len(list((Path(cfg['output_dir']) / 'plots').glob('*.png'))) == len(images)
    with pytest.raises(ValueError, match='identical model'):
        training.train({**cfg, 'epochs': 2, 'resume_path': str(resumed_path), 'support_count': 2})
    with pytest.raises(ValueError, match='original output directory'):
        training.train({**cfg, 'epochs': 2, 'resume_path': str(resumed_path), 'output_dir': str(tmp_path / 'elsewhere')})
    with pytest.raises(ValueError, match='Keep plot_examples and plot_seed fixed'):
        training.train({**cfg, 'epochs': 2, 'resume_path': str(resumed_path), 'plot_examples': 1})


def test_microbatch_accumulation_matches_full_weighted_update(dataset, tmp_path):
    cfg = _config(dataset, tmp_path / 'unused')
    model_cfg = SupervisedModelConfig(**{**cfg['model'], 'dropout': 0.})
    batch = training.numeric_batch(dataset.batch(dataset.rows('train')[:4], support_count=1))
    batch['example_mask'] = jnp.asarray([True, True, True, False])
    params = init_model(jax.random.PRNGKey(4), model_cfg, support_count=1)
    optimizer = optax.sgd(.001)
    state = training.TrainState(jnp.asarray(0), params, optimizer.init(params), jax.random.PRNGKey(3))
    accumulated, metrics = training.create_train_step(optimizer, model_cfg, 2)(state, batch)
    complete, full_metrics = training.create_train_step(optimizer, model_cfg, 4)(state, batch)
    for a, b in zip(jax.tree.leaves(accumulated), jax.tree.leaves(complete)):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-5)
    for name in metrics:
        np.testing.assert_allclose(metrics[name], full_metrics[name], atol=1e-6, rtol=1e-5)


def test_validation_uses_bounded_batches_and_numeric_boundary(dataset, tmp_path, monkeypatch):
    cfg = _config(dataset, tmp_path / 'unused')
    batch = dataset.batch(dataset.rows('development')[:2], support_count=1)
    numeric = training.numeric_batch(batch)
    assert set(numeric) == training.BATCH_FIELDS
    changed_metadata = {**batch, 'metadata': {'query_action': 'must never enter model'}}
    for name, value in training.numeric_batch(changed_metadata).items():
        np.testing.assert_array_equal(value, numeric[name])
    unconditional = training.numeric_batch(batch, condition_on_support=False)
    assert not np.any(unconditional['support_mask']) and not np.any(unconditional['support_tokens'])
    with pytest.raises(ValueError, match='missing=.*query_mask'):
        training.numeric_batch({name: value for name, value in batch.items() if name != 'query_mask'})
    with pytest.raises(ValueError, match='unexpected=.*query_id'):
        training.numeric_batch({**batch, 'query_id': np.zeros(2)})
    sizes = []

    def evaluate(params, batch, key):
        sizes.append(len(batch['example_mask']))
        return {'loss': jnp.asarray(2.)}

    monkeypatch.setattr(training, 'create_validation_step', lambda cfg: evaluate)
    values, count = training.validate({}, dataset, cfg, SupervisedModelConfig(**cfg['model']))
    assert set(sizes) == {cfg['micro_batch_size']}
    assert count == len(dataset.rows('development')) and values['loss'] == 2.


def test_plotting_upgrade_is_limited_to_known_predecessor_and_keeps_science_checks():
    current = training._execution_signature()
    previous = {**current, 'source_hashes': {
        **current['source_hashes'], 'supervised_train.py': training.PRE_PLOT_TRAINER_SHA256}}
    assert training._compatible_execution(previous, current)
    for name in ('supervised_models.py', 'full_data.py'):
        changed = {**previous, 'source_hashes': {**previous['source_hashes'], name: 'changed'}}
        assert not training._compatible_execution(changed, current)
    changed = {**previous, 'source_hashes': {**previous['source_hashes'], 'supervised_train.py': 'unknown'}}
    assert not training._compatible_execution(changed, current)
    assert not training._compatible_execution({**previous, 'backend': 'changed'}, current)
    old_cfg = {key: value for key, value in training.default_config().items() if not key.startswith('plot_')}
    assert training._scientific_config(training.resolve_config(old_cfg)) == training._scientific_config(old_cfg)
    for invalid in ({'plot_every': -1}, {'plot_seed': -1}, {'plot_seed': 2 ** 32},
                    {'plot_examples': 0}, {'plot_examples': 9}):
        with pytest.raises(ValueError, match='plot_'):
            training.resolve_config(invalid)


def test_plot_files_are_kept_without_wandb_and_small_runs_do_not_force_uploads(dataset, tmp_path):
    cfg = {**_config(dataset, tmp_path / 'local'), 'wandb_project': None,
           'plot_every': 2, 'plot_examples': 1, 'max_steps': 3}
    payload = load_checkpoint(training.train(cfg))
    rows = [row for row in payload['extra']['history'] if 'samples/context_and_generated' in row]
    assert [row['optimizer_step'] for row in rows] == [2]
    assert (Path(cfg['output_dir']) / rows[0]['samples/context_and_generated']).is_file()


@pytest.mark.parametrize('architecture', ['autoregressive', 'diffusion'])
def test_periodic_fid_is_optional_replayable_and_preserves_training(dataset, tmp_path,
                                                                  architecture, wandb_calls, monkeypatch):
    from icil_jax_rlbench.quickdraw import supervised_fid as fid
    reference = SimpleNamespace(identifier='fixed-reference', split='development')
    monkeypatch.setattr(fid, 'load_reference', lambda *args: reference)
    monkeypatch.setattr(fid, 'preflight', lambda *args, **kwargs: None)
    evaluations = []

    def evaluate(params, model_cfg, data, real, output, **kwargs):
        assert real is reference and data.identifier == dataset.identifier
        assert kwargs['seed'] == 2030
        assert kwargs['selection_mode'] == 'sample_top_m'
        assert model_cfg.architecture == architecture
        # This evaluator has its own random stream and synchronizes live params.
        noise = jax.random.normal(jax.random.PRNGKey(kwargs['seed']), (4,))
        assert np.isfinite(float(noise.sum()) + float(jax.tree.leaves(params)[0].sum()))
        evaluations.append(kwargs['optimizer_step'])
        return {'sketch_fid': 3.25, 'generated_count': 20, 'reference_count': 20,
                'elapsed_seconds': .1, 'statistics': {'empty_count': 1, 'invalid_fraction': 0.}}

    monkeypatch.setattr(fid, 'evaluate_params', evaluate)
    base = {**_config(dataset, tmp_path / 'fid-disabled', architecture),
            'plot_every': 0, 'max_steps': 5}
    baseline = load_checkpoint(training.train(base))
    assert evaluations == []
    cfg = {**base, 'output_dir': str(tmp_path / 'fid-enabled'),
           'fid_enabled': True, 'fid_every': 2}
    interrupted = training.train({**cfg, 'max_steps': 3})
    resumed = load_checkpoint(training.train({**cfg, 'resume_path': str(interrupted)}))
    assert evaluations == [2, 4]
    for name in ('params', 'opt_state', 'rng'):
        for left, right in zip(jax.tree.leaves(baseline[name]), jax.tree.leaves(resumed[name])):
            np.testing.assert_array_equal(left, right)
    rows = [row for row in resumed['extra']['history'] if 'validation/sketch_fid' in row]
    assert [row['optimizer_step'] for row in rows] == [2, 4]
    assert all(row['validation/sketch_fid'] == 3.25 for row in rows)
    assert all(row['validation/fid_empty_count'] == 1 for row in rows)
    uploaded = [row for run in wandb_calls[1:] for row in run.rows if 'validation/sketch_fid' in row]
    assert uploaded == rows  # Resume does not re-upload the historical metric.
    with pytest.raises(ValueError, match='reference and seed fixed'):
        training.train({**cfg, 'max_steps': 6, 'resume_path': str(interrupted), 'fid_seed': 99})
    # Checkpoint-only restoration must retain the fixed metric identity too.
    (Path(cfg['output_dir']) / 'fid/selection.json').unlink()
    with pytest.raises(ValueError, match='reference and seed fixed'):
        training.train({**cfg, 'max_steps': 6, 'resume_path': str(interrupted), 'fid_seed': 99})


def test_fid_resume_migration_and_config_validation():
    current = training._execution_signature()
    previous = {**current, 'source_hashes': {
        **current['source_hashes'], 'supervised_train.py': training.PRE_FID_TRAINER_SHA256}}
    assert training._compatible_execution(previous, current)
    old_cfg = {key: value for key, value in training.default_config().items() if not key.startswith('fid_')}
    assert training._scientific_config(training.resolve_config(old_cfg)) == training._scientific_config(old_cfg)
    assert training.default_config()['fid_every'] == 10000
    assert training.default_config()['fid_enabled'] is False
    for invalid in ({'fid_enabled': 'false'}, {'fid_every': 0}, {'fid_batch_size': 0},
                    {'fid_feature_batch_size': -1}, {'fid_seed': -1}, {'fid_seed': 2 ** 32},
                    {'fid_keep_artifacts': 1}):
        with pytest.raises(ValueError, match='fid_'):
            training.resolve_config(invalid)


def test_training_fid_refuses_test_references(dataset, tmp_path, monkeypatch):
    from icil_jax_rlbench.quickdraw import supervised_fid as fid
    monkeypatch.setattr(fid, 'load_reference', lambda *args: SimpleNamespace(split='test'))
    with pytest.raises(ValueError, match='development references'):
        training._prepare_training_fid({**training.default_config(), 'fid_enabled': True}, dataset, tmp_path)
