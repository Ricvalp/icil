"""Larger fast MLP with three support-BC steps and 35 held-out classes."""

from icil_jax_rlbench.configs.quickdraw_support_bc_transformer_heldout import get_config as base_config


def get_config():
    cfg = base_config()
    cfg['model'].update({'fast_dim': 128, 'fast_hidden_dim': 256})
    cfg['output_dir'] = 'outputs/quickdraw_icil/support_bc_transformer_fast128_heldout35_v1'
    return cfg
