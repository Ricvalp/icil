"""Check Slurm resource selection and argument forwarding without submitting jobs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def cluster(tmp_path):
    root = tmp_path / 'project'
    (root / 'hpc').mkdir(parents=True)
    scripts = Path(__file__).resolve().parents[1] / 'hpc'
    for name in ('submit_quickdraw_h200.sh', 'quickdraw_h200.sbatch',
                 'quickdraw_evaluate_h200.sbatch'):
        shutil.copyfile(scripts / name, root / 'hpc' / name)
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    programs = {
        'sinfo': "import os; print(os.environ['CLUSTER_TEST_RESOURCES'])",
        'sbatch': "import json, sys; print(json.dumps(sys.argv[1:]))",
        'srun': "import json, sys; print(json.dumps(sys.argv[1:]))",
    }
    for name, code in programs.items():
        path = fake_bin / name
        path.write_text(f'#!{sys.executable}\n{code}\n')
        path.chmod(0o700)
    env = {**os.environ, 'PATH': str(fake_bin) + os.pathsep + os.environ['PATH']}
    return root, env


@pytest.mark.parametrize('resources, expected', [
    ('gpu:nvidia_h200:8(S:0-1)|gpu_node\ngpu:a100:4|gpu_node', ['--gres=gpu:nvidia_h200:1']),
    ('gpu:H200:4|h200\ngpu:H200:4|h200', ['--gres=gpu:H200:1']),
    ('gpu:8|h200,nvlink', ['--gres=gpu:1', '--constraint=h200']),
])
def test_submission_selects_one_h200_and_forwards_trainer_arguments(cluster, resources, expected):
    root, env = cluster
    result = subprocess.run(['bash', str(root / 'hpc/submit_quickdraw_h200.sh'),
                             'diffusion', '--resume', '--set', 'plot_every=20000'],
        cwd=root.parent, env={**env, 'CLUSTER_TEST_RESOURCES': resources}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout.splitlines()[-1])
    assert args[:len(expected)] == expected
    assert args[len(expected):] == ['--job-name=quickdraw_diffusion', 'hpc/quickdraw_h200.sbatch',
                                   'diffusion', '--resume', '--set', 'plot_every=20000']
    assert (root / 'hpc/logs').is_dir()


@pytest.mark.parametrize('resources', ['gpu:a100:8|gpu_node',
                                       'gpu:h200:8|gpu_node\ngpu:nvidia_h200:8|gpu_node'])
def test_submission_refuses_missing_or_ambiguous_h200(cluster, resources):
    root, env = cluster
    result = subprocess.run(['bash', str(root / 'hpc/submit_quickdraw_h200.sh'), 'ar'],
        env={**env, 'CLUSTER_TEST_RESOURCES': resources}, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Cannot identify one unambiguous H200' in result.stderr
    assert not result.stdout


@pytest.mark.parametrize('mode', ['visualize', 'fid'])
def test_submission_routes_checkpoint_evaluation_arguments(cluster, mode):
    root, env = cluster
    forwarded = ['--checkpoint', 'outputs/completed run/best.pkl', '--output', 'outputs/paper figures']
    if mode == 'fid':
        forwarded += ['--reference', '/data/frozen reference']
    result = subprocess.run(['bash', str(root / 'hpc/submit_quickdraw_h200.sh'), mode, *forwarded],
        cwd=root.parent, env={**env, 'CLUSTER_TEST_RESOURCES': 'gpu:H200:4|gpu_node'},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout.splitlines()[-1])
    assert args == ['--gres=gpu:H200:1', f'--job-name=quickdraw_{mode}',
                    'hpc/quickdraw_evaluate_h200.sbatch', mode, *forwarded]


def test_submission_without_arguments_still_defaults_to_ar(cluster):
    root, env = cluster
    result = subprocess.run(['bash', str(root / 'hpc/submit_quickdraw_h200.sh')],
        env={**env, 'CLUSTER_TEST_RESOURCES': 'gpu:H200:4|gpu_node'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout.splitlines()[-1])
    assert args == ['--gres=gpu:H200:1', '--job-name=quickdraw_ar', 'hpc/quickdraw_h200.sbatch']


def test_batch_job_requests_twelve_hours_and_forwards_offline_mode_without_installing(cluster):
    root, env = cluster
    script = root / 'hpc/quickdraw_h200.sbatch'
    directives = [line for line in script.read_text().splitlines() if line.startswith('#SBATCH ')]
    assert '#SBATCH --time=12:00:00' in directives
    assert '#SBATCH --gres=gpu:h200:1' in directives
    assert '#SBATCH --nodes=1' in directives and '#SBATCH --ntasks=1' in directives
    assert '/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1' in script.read_text()
    python = root / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.touch(mode=0o700)
    config = root / 'icil_jax_rlbench/configs/quickdraw_ar_transformer.py'
    config.parent.mkdir(parents=True)
    config.touch()
    manifest = root.parent / 'external data/quickdraw_full_nn_v1/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    result = subprocess.run(['bash', str(script), 'ar', '--resume'],
        env={**env, 'SLURM_SUBMIT_DIR': str(root), 'WANDB_MODE': 'offline',
             'QUICKDRAW_DATASET_ROOT': str(manifest.parent)},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(calls) == 2  # GPU preflight followed by the actual training process.
    assert calls[1] == ['--ntasks=1', '.venv/bin/python', '-u', '-m',
        'icil_jax_rlbench.quickdraw.supervised_train', '--config',
        'icil_jax_rlbench/configs/quickdraw_ar_transformer.py', '--set',
        f'dataset_root="{manifest.parent}"', '--set',
        'wandb_mode="offline"', '--resume']


@pytest.mark.parametrize('mode, module, subcommand', [
    ('visualize', 'supervised_visualize', []),
    ('fid', 'supervised_fid', ['evaluate']),
])
def test_evaluation_job_resources_and_argument_forwarding(cluster, mode, module, subcommand):
    root, env = cluster
    script = root / 'hpc/quickdraw_evaluate_h200.sbatch'
    directives = {line for line in script.read_text().splitlines() if line.startswith('#SBATCH ')}
    assert {'#SBATCH --time=12:00:00', '#SBATCH --gres=gpu:h200:1', '#SBATCH --nodes=1',
            '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=16', '#SBATCH --mem=64G'} <= directives
    assert '/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1' in script.read_text()
    python = root / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.touch(mode=0o700)
    source = root / f'icil_jax_rlbench/quickdraw/{module}.py'
    source.parent.mkdir(parents=True)
    source.touch()
    manifest = root.parent / 'external data/quickdraw_full_nn_v1/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    forwarded = ['--checkpoint', 'outputs/completed run/best.pkl', '--output', 'outputs/evaluation']
    if mode == 'fid':
        forwarded += ['--reference', '/data/frozen reference', '--batch-size', '8']
    result = subprocess.run(['bash', str(script), mode, *forwarded],
        env={**env, 'SLURM_SUBMIT_DIR': str(root), 'QUICKDRAW_DATASET_ROOT': str(manifest.parent)},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(calls) == 2
    assert calls[0] == ['--ntasks=1', '.venv/bin/python', '-u', '-']
    assert calls[1] == ['--ntasks=1', '.venv/bin/python', '-u', '-m',
        f'icil_jax_rlbench.quickdraw.{module}', *subcommand,
        '--dataset-root', str(manifest.parent), *forwarded]
