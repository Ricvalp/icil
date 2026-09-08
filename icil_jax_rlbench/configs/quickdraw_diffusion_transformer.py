"""All-category supervised diffusion ICIL with context cross-attention."""

from icil_jax_rlbench.quickdraw.supervised_train import default_config


def get_config():
    cfg = default_config('diffusion')
    cfg['output_dir'] = 'outputs/quickdraw_icil/diffusion_transformer_v1'
    cfg['wandb_project'] = 'icil-quickdraw'
    return cfg
