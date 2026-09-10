"""Support-BC experiment definitions and Slurm routing without real jobs."""

from importlib import import_module
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_quickdraw_hpc import cluster  # noqa: F401 - shared fake Slurm fixture.


EXPERIMENTS = (
    ('bc-heldout', 'quickdraw_support_bc_transformer_heldout', 'support_bc', 3, 64, 128, 256, 6, 4, 4),
    ('bc1-heldout', 'quickdraw_support_bc_transformer_s1_heldout', 'support_bc', 1, 64, 128, 256, 6, 4, 4),
    ('bc5-heldout', 'quickdraw_support_bc_transformer_s5_heldout', 'support_bc', 5, 64, 128, 256, 6, 4, 4),
    ('bc-fast128-heldout', 'quickdraw_support_bc_transformer_fast128_heldout',
     'support_bc', 3, 128, 256, 256, 6, 4, 4),
    ('bc-large-heldout', 'quickdraw_support_bc_transformer_large_heldout',
     'support_bc', 3, 64, 128, 384, 8, 6, 2),
    ('kvb-large-heldout', 'quickdraw_kvb_transformer_large_heldout', 'kvb', 3, 64, 128, 384, 8, 6, 2),
)


def _script(mode):
    return f'quickdraw_{mode.replace("-", "_")}_h200.sbatch'


@pytest.fixture
def support_bc_cluster(cluster):
    root, env = cluster
    scripts = Path(__file__).resolve().parents[1] / 'hpc'
    for mode, *_ in EXPERIMENTS:
        shutil.copyfile(scripts / _script(mode), root / 'hpc' / _script(mode))
    return root, env


@pytest.mark.parametrize('mode', [item[0] for item in EXPERIMENTS])
def test_support_bc_submission_routes_each_experiment_to_its_own_h200_job(support_bc_cluster, mode):
    root, env = support_bc_cluster
    forwarded = ['--resume', '--set', 'fid_enabled=true']
    result = subprocess.run(['bash', str(root / 'hpc/submit_quickdraw_h200.sh'), mode, *forwarded],
        cwd=root.parent, env={**env, 'CLUSTER_TEST_RESOURCES': 'gpu:H200:4|gpu_node'},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [
        '--gres=gpu:H200:1', f'--job-name=quickdraw_{mode}',
        f'hpc/{_script(mode)}', mode, *forwarded]


@pytest.mark.parametrize('explicit_mode', [False, True])
@pytest.mark.parametrize('mode, config_name', [(item[0], item[1]) for item in EXPERIMENTS])
def test_support_bc_spooled_batch_files_preserve_allocation_and_training_arguments(
        support_bc_cluster, mode, config_name, explicit_mode):
    root, env = support_bc_cluster
    source = root / 'hpc' / _script(mode)
    directives = {line for line in source.read_text().splitlines() if line.startswith('#SBATCH ')}
    assert {'#SBATCH --time=12:00:00', '#SBATCH --gres=gpu:h200:1', '#SBATCH --nodes=1',
            '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=16', '#SBATCH --mem=64G'} <= directives
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
    spool_copy = root.parent / 'slurm_script'
    shutil.copyfile(source, spool_copy)
    forwarded = ['--resume', '--set', 'plot_every=20000', '--set',
                 'output_dir="outputs/experiment with spaces"']
    result = subprocess.run(['bash', str(spool_copy), *([mode] if explicit_mode else []), *forwarded],
        cwd=root.parent, env={**env, 'SLURM_SUBMIT_DIR': str(root),
             'QUICKDRAW_DATASET_ROOT': str(manifest.parent), 'WANDB_MODE': 'offline'},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(calls) == 2
    assert calls[0] == ['--ntasks=1', '.venv/bin/python', '-u', '-']
    assert calls[1] == ['--ntasks=1', '.venv/bin/python', '-u', '-m',
        'icil_jax_rlbench.quickdraw.supervised_train', '--config', relative_config,
        '--set', f'dataset_root="{manifest.parent}"', '--set', 'wandb_mode="offline"', *forwarded]


def test_support_bc_experiments_share_task_split_training_budget_and_logging():
    configs = []
    for _, module, method, steps, fast_dim, fast_hidden, hidden, decoder, context, micro in EXPERIMENTS:
        cfg = import_module(f'icil_jax_rlbench.configs.{module}').get_config()
        configs.append(cfg)
        assert cfg['method'] == method
        assert cfg['seed'] == 0
        assert cfg['heldout_category_count'] == 35
        assert cfg['heldout_category_seed'] == 37
        assert cfg['support_count'] == 4 and cfg['condition_on_support'] is True
        assert cfg['batch_size'] == 64 and cfg['micro_batch_size'] == micro
        assert cfg['epochs'] == 20 and cfg['selection_mode'] == 'exact_top_k'
        assert cfg['wandb_project'] == 'icil-quickdraw' and cfg['wandb_mode'] == 'online'
        assert cfg['plot_every'] == cfg['fid_every'] == 10000
        assert cfg['fid_enabled'] is False
        model = cfg['model']
        assert model['architecture'] == 'autoregressive'
        assert model['inner_steps'] == steps and model['first_order'] is False
        assert model['fast_dim'] == fast_dim and model['fast_hidden_dim'] == fast_hidden
        assert model['hidden_dim'] == hidden and model['num_heads'] == 8
        assert model['decoder_layers'] == decoder and model['context_layers'] == context
        assert model['dtype'] == 'float32'
    assert len({cfg['output_dir'] for cfg in configs}) == len(EXPERIMENTS)
    assert all(cfg['output_dir'].endswith('_heldout35_v1') for cfg in configs)
    # The large BC and KVB comparison has the same declared Transformer and
    # fast-state dimensions, differing only in method and output directory.
    large_bc, large_kvb = configs[-2:]
    assert {key: value for key, value in large_bc.items() if key not in ('method', 'output_dir')} == {
        key: value for key, value in large_kvb.items() if key not in ('method', 'output_dir')}
