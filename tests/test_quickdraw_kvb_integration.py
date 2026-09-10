"""KVB checkpoint generation, evaluation provenance and HPC routing."""
from __future__ import annotations

import json
import subprocess

import jax
import numpy as np

from icil_jax_rlbench.quickdraw import (
    policy_backend, supervised_evaluate as evaluation, supervised_fid as fid,
    supervised_plots as plots, supervised_train as training,
    supervised_visualize as figures,
)
from icil_jax_rlbench.quickdraw.kvb_models import KVBModelConfig
from icil_jax_rlbench.quickdraw.metrics import load_trajectories
from test_quickdraw_hpc import cluster  # noqa: F401 - shared fake Slurm fixture.
from test_quickdraw_supervised_evaluate import _prepare
from test_quickdraw_supervised_fid import _reference


def _model(steps):
    cfg = KVBModelConfig(hidden_dim=8, num_heads=2, context_layers=1,
        decoder_layers=1, mlp_ratio=2, max_steps=steps, mixture_components=2,
        dropout=0., fast_dim=4, fast_hidden_dim=6, inner_steps=3)
    params = policy_backend.init_model(jax.random.PRNGKey(12), cfg, support_count=2)
    return cfg, params


def _checkpoint(tmp_path, monkeypatch, dataset, cfg, params):
    checkpoint = tmp_path / 'best.pkl'
    checkpoint.write_bytes(b'synthetic-kvb-checkpoint-with-in-memory-parameters')
    monkeypatch.setattr(training, 'load_run', lambda *args, **kwargs: (
        {'params': params, 'step': 10000},
        {'method': 'kvb', 'support_count': 2, 'selection_mode': 'exact_top_k',
         'condition_on_support': True}, cfg, dataset))
    return checkpoint


def test_kvb_figures_and_fid_use_adapted_generation_and_record_numerical_sources(tmp_path, monkeypatch):
    dataset, resources, reference = _reference(tmp_path, monkeypatch)
    cfg, params = _model(dataset.max_steps)
    checkpoint = _checkpoint(tmp_path, monkeypatch, dataset, cfg, params)
    plot_batch = plots.prepare_plot_batch(dataset, count=2, support_count=2,
        selection_mode='exact_top_k', seed=2027)
    plot = plots.generate_plot(params, cfg, plot_batch, tmp_path / 'training.png', step=10000)
    assert plot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    output = figures.visualize_checkpoint(checkpoint, tmp_path / 'figures',
        context_examples=2, grid_rows=2, grid_columns=2, batch_size=2,
        formats=('png',), dpi=60, progress=False)
    metadata = json.loads((output / 'samples.json').read_text())
    assert metadata['method'] == 'kvb'
    assert metadata['architecture'] == 'autoregressive'
    assert metadata['model_config']['inner_steps'] == 3
    assert metadata['generated_count'] == len(metadata['records']) == 4
    assert set(policy_backend.numerical_sources(cfg)) <= set(metadata['execution']['source_hashes'])
    for record in metadata['records']:
        assert record['target_row'] not in record['support_rows']
        assert isinstance(record['generation_status'], list)

    report = fid.evaluate_checkpoint(checkpoint, reference.root, tmp_path / 'fid',
        batch_size=2, progress=False, **resources)
    assert np.isfinite(report['sketch_fid'])
    assert report['generated_count'] == report['reference_count'] == 4
    provenance = report['generation_provenance']
    assert provenance['method'] == 'kvb'
    assert provenance['model']['inner_steps'] == 3
    assert {'kvb_models.py', 'fast_weight_ttt.py', 'policy_backend.py'} <= set(provenance['source_hashes'])
    assert not (tmp_path / 'fid/artifacts').exists()


def test_kvb_control_evaluator_runs_loss_and_generation_with_empty_write_control(tmp_path, monkeypatch):
    dataset, episodes = _prepare(tmp_path, monkeypatch)
    cfg, params = _model(dataset.max_steps)
    checkpoint = _checkpoint(tmp_path, monkeypatch, dataset, cfg, params)
    output = evaluation.generate_run(checkpoint, episodes, tmp_path / 'controls',
        conditions=('correct_support', 'no_context'), samples_per_task=1,
        batch_size=2, seed=9, progress=False)
    _, correct = load_trajectories(output / 'correct_support')
    _, empty = load_trajectories(output / 'no_context')
    assert correct['method'] == empty['method'] == 'kvb'
    assert all(row['support_ids'] for row in correct['records'])
    assert all(not row['support_ids'] for row in empty['records'])
    assert [row['query_seed'] for row in correct['records']] == [row['query_seed'] for row in empty['records']]
    summary = json.loads((output / 'summary.json').read_text())
    for condition in ('correct_support', 'no_context'):
        assert np.isfinite(summary['conditions'][condition]['conditional_target_loss']['mean'])


def test_hpc_routes_kvb_without_submitting_a_real_job(cluster):
    root, env = cluster
    forwarded = ['--resume', '--set', 'fid_enabled=true']
    submitted = subprocess.run(['bash', str(root / 'hpc/submit_quickdraw_h200.sh'), 'kvb', *forwarded],
        env={**env, 'CLUSTER_TEST_RESOURCES': 'gpu:H200:4|gpu_node'}, capture_output=True, text=True)
    assert submitted.returncode == 0, submitted.stderr
    assert json.loads(submitted.stdout.splitlines()[-1]) == [
        '--gres=gpu:H200:1', '--job-name=quickdraw_kvb', 'hpc/quickdraw_h200.sbatch', 'kvb', *forwarded]
    python = root / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.touch(mode=0o700)
    config = root / 'icil_jax_rlbench/configs/quickdraw_kvb_transformer.py'
    config.parent.mkdir(parents=True)
    config.touch()
    manifest = root.parent / 'data/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    executed = subprocess.run(['bash', str(root / 'hpc/quickdraw_h200.sbatch'), 'kvb', *forwarded],
        env={**env, 'SLURM_SUBMIT_DIR': str(root), 'QUICKDRAW_DATASET_ROOT': str(manifest.parent),
             'WANDB_MODE': 'offline'}, capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    calls = [json.loads(line) for line in executed.stdout.splitlines()]
    assert calls[1] == ['--ntasks=1', '.venv/bin/python', '-u', '-m',
        'icil_jax_rlbench.quickdraw.supervised_train', '--config',
        'icil_jax_rlbench/configs/quickdraw_kvb_transformer.py', '--set',
        f'dataset_root="{manifest.parent}"', '--set', 'wandb_mode="offline"', *forwarded]
