"""Exercise the twelve capacity jobs against fake Slurm, including GPU flags."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_quickdraw_hpc import cluster  # noqa: F401 - shared fake Slurm fixture.


EXPERIMENTS = tuple(
    (f'{method}-{size}-f{capacity}',
     f'quickdraw_{"support_bc" if method == "bc" else method}_transformer_'
     f'{size}_fast{capacity}_heldout')
    for method in ('bc', 'kvb')
    for size in ('base', 'large')
    for capacity in ('500k', '1m', '3m'))


def _script(mode):
    return f'quickdraw_{mode.replace("-", "_")}_h200.sbatch'


@pytest.fixture
def capacity_cluster(cluster):
    root, env = cluster
    scripts = Path(__file__).resolve().parents[1] / 'hpc'
    for mode, _ in EXPERIMENTS:
        shutil.copyfile(scripts / _script(mode), root / 'hpc' / _script(mode))
    env = {key: value for key, value in env.items()
           if key not in ('QUICKDRAW_CUDA_GRAPHS', 'XLA_FLAGS')}
    fake_srun = root.parent / 'bin/srun'
    fake_srun.write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        'print(json.dumps({"args": sys.argv[1:], "xla_flags": os.environ.get("XLA_FLAGS")}))\n')
    return root, env


def _prepare_job(root, mode):
    config_name = dict(EXPERIMENTS)[mode]
    python = root / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.touch(mode=0o700)
    relative_config = f'icil_jax_rlbench/configs/{config_name}.py'
    config = root / relative_config
    config.parent.mkdir(parents=True)
    config.touch()
    manifest = root.parent / 'external data/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    spool = root.parent / 'slurm_script'
    shutil.copyfile(root / 'hpc' / _script(mode), spool)
    return spool, relative_config, manifest.parent


@pytest.mark.parametrize('mode', [item[0] for item in EXPERIMENTS])
def test_capacity_submission_routes_each_mode_to_its_own_job(capacity_cluster, mode):
    root, env = capacity_cluster
    forwarded = ['--resume', '--set', 'fid_enabled=true', '--set',
                 'output_dir="outputs/experiment with spaces"']
    result = subprocess.run(
        ['bash', str(root / 'hpc/submit_quickdraw_h200.sh'), mode, *forwarded],
        cwd=root.parent, env={**env, 'CLUSTER_TEST_RESOURCES': 'gpu:H200:4|gpu_node'},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [
        '--gres=gpu:H200:1', f'--job-name=quickdraw_{mode}',
        f'hpc/{_script(mode)}', mode, *forwarded]


@pytest.mark.parametrize('explicit_mode', [False, True])
@pytest.mark.parametrize('mode', [item[0] for item in EXPERIMENTS])
def test_spooled_capacity_jobs_keep_resources_flags_and_arguments(
        capacity_cluster, mode, explicit_mode):
    root, env = capacity_cluster
    script = root / 'hpc' / _script(mode)
    directives = {line for line in script.read_text().splitlines() if line.startswith('#SBATCH ')}
    assert {'#SBATCH --time=12:00:00', '#SBATCH --gres=gpu:h200:1', '#SBATCH --nodes=1',
            '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=16', '#SBATCH --mem=64G'} <= directives
    spool, config, dataset = _prepare_job(root, mode)
    forwarded = ['--resume', '--set', 'plot_every=20000', '--set',
                 'output_dir="outputs/experiment with spaces"']
    result = subprocess.run(
        ['bash', str(spool), *([mode] if explicit_mode else []), *forwarded],
        cwd=root.parent, env={**env, 'SLURM_SUBMIT_DIR': str(root),
                             'QUICKDRAW_DATASET_ROOT': str(dataset), 'WANDB_MODE': 'offline'},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == (
        'QUICKDRAW_CUDA_GRAPHS=0 XLA_FLAGS=--xla_gpu_enable_command_buffer=')
    calls = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert len(calls) == 2
    assert all(call['xla_flags'] == '--xla_gpu_enable_command_buffer=' for call in calls)
    assert calls[0]['args'] == ['--ntasks=1', '.venv/bin/python', '-u', '-']
    assert calls[1]['args'] == ['--ntasks=1', '.venv/bin/python', '-u', '-m',
        'icil_jax_rlbench.quickdraw.supervised_train', '--config', config,
        '--set', f'dataset_root="{dataset}"', '--set', 'wandb_mode="offline"', *forwarded]


@pytest.mark.parametrize('flags_env, expected', [
    ({'XLA_FLAGS': '--xla_gpu_autotune_level=2'},
     '--xla_gpu_autotune_level=2 --xla_gpu_enable_command_buffer='),
    ({'QUICKDRAW_CUDA_GRAPHS': '0', 'XLA_FLAGS': ''}, '--xla_gpu_enable_command_buffer='),
    ({'QUICKDRAW_CUDA_GRAPHS': '1'}, None),
    ({'QUICKDRAW_CUDA_GRAPHS': '1', 'XLA_FLAGS': '--xla_gpu_enable_command_buffer=FUSION'},
     '--xla_gpu_enable_command_buffer=FUSION'),
])
def test_capacity_cuda_graph_setting_preserves_inherited_flags(capacity_cluster, flags_env, expected):
    root, env = capacity_cluster
    spool, _, dataset = _prepare_job(root, 'kvb-large-f3m')
    result = subprocess.run(['bash', str(spool)], cwd=root.parent,
        env={**env, **flags_env, 'SLURM_SUBMIT_DIR': str(root),
             'QUICKDRAW_DATASET_ROOT': str(dataset)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert len(calls) == 2
    assert all(call['xla_flags'] == expected for call in calls)
    assert f'XLA_FLAGS={expected or ""}' in result.stdout.splitlines()[0]


@pytest.mark.parametrize('mode', [item[0] for item in EXPERIMENTS])
def test_capacity_job_rejects_invalid_cuda_graph_option_before_launch(capacity_cluster, mode):
    root, env = capacity_cluster
    result = subprocess.run(['bash', str(root / 'hpc' / _script(mode))],
        env={**env, 'QUICKDRAW_CUDA_GRAPHS': 'maybe'}, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'QUICKDRAW_CUDA_GRAPHS must be 0 or 1' in result.stderr
    assert not result.stdout


def test_capacity_job_rejects_wrong_experiment_selector(capacity_cluster):
    root, env = capacity_cluster
    result = subprocess.run(['bash', str(root / 'hpc' / _script('bc-base-f500k')), 'kvb-large-f3m'],
        env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'after bc-base-f500k' in result.stderr
    assert not result.stdout
