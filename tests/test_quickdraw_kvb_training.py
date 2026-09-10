from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import supervised_fid as fid
from icil_jax_rlbench.quickdraw import supervised_train as training
from icil_jax_rlbench.quickdraw.kvb_models import KVBModelConfig
from icil_jax_rlbench.quickdraw.policy_backend import method_name, model_config
from icil_jax_rlbench.train.checkpoints import load_checkpoint
from test_quickdraw_supervised_training import _config, dataset, wandb_calls


@pytest.mark.parametrize('method', ['kvb', 'support_bc'])
def test_fast_weight_full_epoch_resume_figures_fid_and_checkpoint_state(dataset, tmp_path,
                                                                       wandb_calls, monkeypatch, method):
    base = _config(dataset, tmp_path / 'continuous')
    base.update(method=method, support_count=2, plot_every=0)
    base['model'] = asdict(model_config({**base['model'], 'fast_dim': 4,
                                        'fast_hidden_dim': 8, 'inner_steps': 3}, method))
    continuous = load_checkpoint(training.train(base))
    reference = SimpleNamespace(identifier='fixed-reference', split='development')
    monkeypatch.setattr(fid, 'load_reference', lambda *args: reference)
    monkeypatch.setattr(fid, 'preflight', lambda *args, **kwargs: None)
    evaluations = []

    def evaluate(params, model_cfg, data, real, output, **kwargs):
        assert real is reference and data.identifier == dataset.identifier
        assert isinstance(model_cfg, KVBModelConfig) and model_cfg.inner_steps == 3
        assert method_name(model_cfg) == method
        assert kwargs['batch_size'] == 64
        assert set(params) == {'backbone', 'adapter'}
        evaluations.append(kwargs['optimizer_step'])
        return {'sketch_fid': 3.25, 'generated_count': 20, 'reference_count': 20,
                'elapsed_seconds': .1, 'statistics': {'empty_count': 1}}

    monkeypatch.setattr(fid, 'evaluate_params', evaluate)
    cfg = {**base, 'output_dir': str(tmp_path / 'resumed'),
           'plot_every': 2, 'plot_examples': 1, 'fid_enabled': True, 'fid_every': 2}
    interrupted = training.train({**cfg, 'max_steps': 3})
    resumed = load_checkpoint(training.train({**cfg, 'resume_path': str(interrupted)}))
    for name in ('params', 'opt_state', 'rng'):
        for left, right in zip(jax.tree.leaves(continuous[name]), jax.tree.leaves(resumed[name])):
            np.testing.assert_array_equal(left, right)
    extra = resumed['extra']
    assert extra['checkpoint_type'] == training.CHECKPOINT_TYPES[method]
    assert extra['next_epoch'] == 1 and extra['next_batch'] == 0
    np.testing.assert_array_equal(extra['exposure']['targets'][dataset.rows('train')], 1)
    assert extra['exposure']['supports'].sum() == 2 * len(dataset.rows('train'))
    # Task-adapted copies have a batch axis; only the shared W0 belongs here.
    assert set(resumed['params']) == {'backbone', 'adapter'}
    assert resumed['params']['adapter']['fast_init']['fc1']['kernel'].shape == (4, 8)
    assert not any('adapted' in key or 'fast_state' in key for key in resumed)
    _, restored_cfg, model_cfg, _ = training.load_run(interrupted)
    assert restored_cfg['method'] == method and isinstance(model_cfg, KVBModelConfig)
    assert method_name(model_cfg) == method
    provenance = json.loads((Path(cfg['output_dir']) / 'provenance.json').read_text())
    assert provenance['meta_gradient'] == 'full_second_order'
    assert provenance['inner_steps_per_task'] == 3
    assert provenance['conditioning'] == 'support_only_through_adapted_fast_state_delta_read'
    assert provenance['outer_objective'] == 'query_likelihood_only'
    assert provenance['fast_parameter_count'] == 76
    if method == 'support_bc':
        assert provenance['write_objective'] == 'support_action_bc'
        assert provenance['write_representation'] == 'shared_causal_query_decoder_cached_per_demo'
        assert not any(name.startswith('context') for name in resumed['params']['backbone'])
        assert not {'key_projection', 'value_projection'} & set(resumed['params']['adapter'])
    history = extra['history']
    assert evaluations == [2, 4, 6]
    assert (Path(cfg['output_dir']) / 'best.pkl').exists()
    assert [row['exposure/write_steps'] for row in history if 'exposure/write_steps' in row][-1] == (
        3 * len(dataset.rows('train')))
    assert any('train/write_loss' in row for row in history)
    assert any('validation/write_loss' in row for row in history)
    assert any('validation/sketch_fid' in row for row in history)
    images = [row['samples/context_and_generated'] for run in wandb_calls[1:]
              for row in run.rows if 'samples/context_and_generated' in row]
    assert len(images) == 3 and all(Path(item.path).is_file() for item in images)
    for run in wandb_calls:
        assert run.exit_code == 0
        assert run.kwargs['job_type'] == method.replace('_', '-') + '-full-second-order'


def test_config_and_legacy_resume_are_separate_from_kvb():
    legacy = training.default_config()
    legacy.pop('method')
    assert training.resolve_config(legacy)['method'] == 'icil'
    assert training._scientific_config(legacy) == training._scientific_config(training.resolve_config(legacy))
    cfg = training.resolve_config({'method': 'kvb'})
    model = model_config(cfg['model'], cfg['method'])
    assert model.dtype == 'float32' and model.inner_steps == 3
    assert (model.fast_dim, model.fast_hidden_dim) == (64, 128)
    with pytest.raises(ValueError, match='method'):
        training.resolve_config({'method': 'unknown'})
    current = training._execution_signature()
    previous = {**current, 'source_hashes': {
        name: (training.PRE_KVB_TRAINER_SHA256 if name == 'supervised_train.py' else digest)
        for name, digest in current['source_hashes'].items() if name != 'policy_backend.py'}}
    assert training._compatible_execution(previous, current)
    kvb = training._execution_signature(model)
    assert training._compatible_execution(kvb, kvb)
    assert {'kvb_models.py', 'fast_weight_ttt.py', 'policy_backend.py'} <= set(kvb['source_hashes'])
    for name in kvb['source_hashes']:
        changed = {**kvb, 'source_hashes': {**kvb['source_hashes'], name: 'modified'}}
        assert not training._compatible_execution(changed, kvb)
    assert not training._compatible_execution(previous, kvb)
    first_order = model_config({**cfg['model'], 'first_order': True}, 'kvb')
    assert method_name(first_order) == 'kvb_first_order'
