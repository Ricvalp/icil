"""Base support-BC Transformer with ~1m fast parameters and 35 held-out classes."""

from icil_jax_rlbench.configs.quickdraw_support_bc_transformer_heldout import get_config as base_config


def get_config():
    cfg = base_config()
    cfg['model'].update({'fast_dim': 256, 'fast_hidden_dim': 1952})
    cfg['output_dir'] = 'outputs/quickdraw_icil/support_bc_transformer_base_fast1m_heldout35_v1'
    return cfg
