"""Full second-order support-BC WRITE with three steps and 35 held-out classes."""

from icil_jax_rlbench.configs.quickdraw_kvb_transformer_heldout import get_config as base_config


def get_config():
    cfg = base_config()
    cfg['method'] = 'support_bc'
    cfg['output_dir'] = 'outputs/quickdraw_icil/support_bc_transformer_heldout35_v1'
    return cfg
