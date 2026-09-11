"""Verify actual model capacities and matched settings for the twelve jobs."""

from importlib import import_module
from itertools import product
from math import prod

import jax
import jax.numpy as jnp
import pytest

from icil_jax_rlbench.quickdraw import policy_backend
from icil_jax_rlbench.quickdraw.supervised_train import resolve_config


METHODS = ('support_bc', 'kvb')
SIZES = ('base', 'large')
BUDGETS = {'500k': (976, 500944), '1m': (1952, 1001632), '3m': (5840, 2996176)}
EXPERIMENTS = tuple(product(METHODS, SIZES, BUDGETS))
# Existing model totals include their 16,576-parameter learned initialization.
BASELINE_TOTALS = {
    ('support_bc', 'base'): 4830008,
    ('kvb', 'base'): 8052408,
    ('support_bc', 'large'): 14321464,
    ('kvb', 'large'): 25060152,
}


def _module(method, size, budget=None):
    suffix = f'_{size}_fast{budget}' if budget else ('' if size == 'base' else '_large')
    return import_module(f'icil_jax_rlbench.configs.quickdraw_{method}_transformer{suffix}_heldout')


@pytest.mark.parametrize('method,size,budget', EXPERIMENTS)
def test_capacity_configs_preserve_the_matched_training_experiment(method, size, budget):
    baseline = _module(method, size).get_config()
    cfg = _module(method, size, budget).get_config()
    assert resolve_config(cfg) == cfg
    assert cfg['method'] == method
    assert cfg['output_dir'] == (
        f'outputs/quickdraw_icil/{method}_transformer_{size}_fast{budget}_heldout35_v1')
    assert cfg['seed'] == 0
    assert (cfg['heldout_category_count'], cfg['heldout_category_seed']) == (35, 37)
    assert cfg['support_count'] == 4 and cfg['condition_on_support'] is True
    assert cfg['selection_mode'] == 'exact_top_k'
    assert cfg['batch_size'] == 64
    assert cfg['micro_batch_size'] == (4 if size == 'base' else 2)
    assert cfg['epochs'] == 20
    assert cfg['plot_every'] == cfg['fid_every'] == 10000
    assert cfg['fid_enabled'] is False
    assert cfg['wandb_project'] == 'icil-quickdraw' and cfg['wandb_mode'] == 'online'
    model = cfg['model']
    assert model['architecture'] == 'autoregressive' and model['dtype'] == 'float32'
    assert model['inner_steps'] == 3 and model['first_order'] is False
    assert model['fast_dim'] == 256 and model['fast_hidden_dim'] == BUDGETS[budget][0]
    assert model['hidden_dim'] == (256 if size == 'base' else 384)
    assert model['decoder_layers'] == (6 if size == 'base' else 8)
    assert model['context_layers'] == (4 if size == 'base' else 6)
    assert model['num_heads'] == 8
    assert model['pen_loss_weight'] == model['stop_loss_weight'] == .25
    # Capacity and output location are the only changes; keep the optimizer,
    # data selection, random seeds, generation and checkpoint settings matched.
    assert {key: value for key, value in cfg.items() if key not in ('model', 'output_dir')} == {
        key: value for key, value in baseline.items() if key not in ('model', 'output_dir')}
    assert {key: value for key, value in model.items() if key not in ('fast_dim', 'fast_hidden_dim')} == {
        key: value for key, value in baseline['model'].items() if key not in ('fast_dim', 'fast_hidden_dim')}


@pytest.mark.parametrize('method,size,budget', EXPERIMENTS)
def test_actual_parameter_trees_meet_the_fast_capacity_and_total_budgets(method, size, budget):
    cfg = _module(method, size, budget).get_config()
    model_cfg = policy_backend.model_config(cfg['model'], cfg['method'])
    params = jax.eval_shape(lambda key: policy_backend.init_model(
        key, model_cfg, support_count=cfg['support_count']), jax.random.key(0))

    def count(tree):
        return sum(prod(leaf.shape) for leaf in jax.tree.leaves(tree))

    fast_count = count(params['adapter']['fast_init'])
    assert fast_count == BUDGETS[budget][1]
    nominal = {'500k': 500000, '1m': 1000000, '3m': 3000000}[budget]
    assert abs(fast_count - nominal) / nominal < .002
    # Increase the known total by W0's growth and the wider slow projections.
    # BC has query/read projections; KVB also has independent key/value ones.
    hidden = model_cfg.hidden_dim
    projection_growth = ((2 * hidden + 1) if method == 'support_bc' else (4 * hidden + 3)) * (256 - 64)
    expected_total = BASELINE_TOTALS[method, size] + (fast_count - 16576) + projection_growth
    assert count(params) == expected_total
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
    assert count(params['adapter']['inner_lr_raw']) == 4
    if method == 'support_bc':
        assert not any(name.startswith('context') for name in params['backbone'])
        assert 'key_projection' not in params['adapter']
        assert 'value_projection' not in params['adapter']
    else:
        assert 'key_projection' in params['adapter'] and 'value_projection' in params['adapter']


def test_all_twelve_runs_have_independent_configuration_trees_and_output_paths():
    configurations = [_module(*experiment).get_config() for experiment in EXPERIMENTS]
    assert len(configurations) == len({cfg['output_dir'] for cfg in configurations}) == 12
    for experiment, cfg in zip(EXPERIMENTS, configurations):
        cfg['model']['fast_dim'] = 7
        cfg['heldout_category_count'] = 0
        pristine = _module(*experiment).get_config()
        assert pristine['model']['fast_dim'] == 256
        assert pristine['heldout_category_count'] == 35
    for method, size in product(METHODS, SIZES):
        original = _module(method, size).get_config()
        assert original['model']['fast_dim'] == 64
        assert original['model']['fast_hidden_dim'] == 128
        assert original['heldout_category_count'] == 35
