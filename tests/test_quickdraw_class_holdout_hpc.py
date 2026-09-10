"""Held-out-class Slurm routing and shared configuration without real jobs."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_quickdraw_hpc import cluster  # noqa: F401 - shared fake Slurm fixture.


@pytest.fixture
def heldout_cluster(cluster):
    root, env = cluster
    source = Path(__file__).resolve().parents[1] / 'hpc'
    for name in ('quickdraw_icil_heldout_h200.sbatch', 'quickdraw_kvb_heldout_h200.sbatch'):
        shutil.copyfile(source / name, root / 'hpc' / name)
    return root, env


@pytest.mark.parametrize('mode, script', [
    ('ar-heldout', 'quickdraw_icil_heldout_h200.sbatch'),
    ('diffusion-heldout', 'quickdraw_icil_heldout_h200.sbatch'),
    ('kvb-heldout', 'quickdraw_kvb_heldout_h200.sbatch'),
])
def test_heldout_submission_routes_to_dedicated_single_h200_job(heldout_cluster, mode, script):
    root, env = heldout_cluster
    forwarded = ['--resume', '--set', 'fid_enabled=true', '--set', 'heldout_category_seed=41']
    result = subprocess.run(['bash', str(root / 'hpc/submit_quickdraw_h200.sh'), mode, *forwarded],
        cwd=root.parent, env={**env, 'CLUSTER_TEST_RESOURCES': 'gpu:nvidia_h200:8|gpu_node'},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [
        '--gres=gpu:nvidia_h200:1', f'--job-name=quickdraw_{mode}',
        f'hpc/{script}', mode, *forwarded]


@pytest.mark.parametrize('script, arguments, architecture', [
    ('quickdraw_icil_heldout_h200.sbatch', [], 'ar'),
    ('quickdraw_icil_heldout_h200.sbatch', ['ar-heldout'], 'ar'),
    ('quickdraw_icil_heldout_h200.sbatch', ['diffusion'], 'diffusion'),
    ('quickdraw_icil_heldout_h200.sbatch', ['diffusion-heldout'], 'diffusion'),
    ('quickdraw_kvb_heldout_h200.sbatch', [], 'kvb'),
    ('quickdraw_kvb_heldout_h200.sbatch', ['kvb-heldout'], 'kvb'),
])
def test_heldout_batch_wrappers_share_training_setup_even_from_slurm_spool(
        heldout_cluster, script, arguments, architecture):
    root, env = heldout_cluster
    source = root / 'hpc' / script
    directives = {line for line in source.read_text().splitlines() if line.startswith('#SBATCH ')}
    assert {'#SBATCH --time=12:00:00', '#SBATCH --gres=gpu:h200:1', '#SBATCH --nodes=1',
            '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=16', '#SBATCH --mem=64G'} <= directives
    python = root / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.touch(mode=0o700)
    config_name = f'icil_jax_rlbench/configs/quickdraw_{architecture}_transformer_heldout.py'
    config = root / config_name
    config.parent.mkdir(parents=True)
    config.touch()
    manifest = root.parent / 'external data/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    # Slurm executes a spooled copy, so locating the shared script through
    # BASH_SOURCE would fail. It must be resolved from SLURM_SUBMIT_DIR.
    spool_copy = root.parent / 'slurm_script'
    shutil.copyfile(source, spool_copy)
    forwarded = ['--resume', '--set', 'heldout_category_count=20']
    result = subprocess.run(['bash', str(spool_copy), *arguments, *forwarded],
        cwd=root.parent, env={**env, 'SLURM_SUBMIT_DIR': str(root),
             'QUICKDRAW_DATASET_ROOT': str(manifest.parent), 'WANDB_MODE': 'offline'},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(calls) == 2
    assert calls[0] == ['--ntasks=1', '.venv/bin/python', '-u', '-']
    assert calls[1] == ['--ntasks=1', '.venv/bin/python', '-u', '-m',
        'icil_jax_rlbench.quickdraw.supervised_train', '--config', config_name,
        '--set', f'dataset_root="{manifest.parent}"', '--set', 'wandb_mode="offline"', *forwarded]


def test_heldout_configs_preserve_architectures_and_use_the_same_class_split():
    from icil_jax_rlbench.configs.quickdraw_ar_transformer_heldout import get_config as ar
    from icil_jax_rlbench.configs.quickdraw_diffusion_transformer_heldout import get_config as diffusion
    from icil_jax_rlbench.configs.quickdraw_kvb_transformer_heldout import get_config as kvb

    configs = [ar(), diffusion(), kvb()]
    assert {cfg['heldout_category_count'] for cfg in configs} == {35}
    assert {cfg['heldout_category_seed'] for cfg in configs} == {37}
    assert len({cfg['output_dir'] for cfg in configs}) == 3
    assert all(cfg['output_dir'].endswith('_heldout35_v1') for cfg in configs)
    assert [cfg['method'] for cfg in configs] == ['icil', 'icil', 'kvb']
    assert [cfg['model']['architecture'] for cfg in configs] == [
        'autoregressive', 'diffusion', 'autoregressive']
    assert configs[2]['model']['inner_steps'] == 3
    assert configs[2]['model']['first_order'] is False
