"""One full second-order support-BC step with 35 held-out classes."""

from icil_jax_rlbench.configs.quickdraw_support_bc_transformer_heldout import get_config as base_config


def get_config():
    cfg = base_config()
    cfg['model']['inner_steps'] = 1
    cfg['output_dir'] = 'outputs/quickdraw_icil/support_bc_transformer_s1_heldout35_v1'
    return cfg
