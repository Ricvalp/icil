"""Ordinary diffusion ICIL with 35 classes held out of policy training."""

from icil_jax_rlbench.configs.quickdraw_diffusion_transformer import get_config as base_config


def get_config():
    cfg = base_config()
    cfg.update({
        'heldout_category_count': 35,
        'heldout_category_seed': 37,
        'output_dir': 'outputs/quickdraw_icil/diffusion_transformer_heldout35_v1',
    })
    return cfg
