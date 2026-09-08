"""All-category, full-training-pool supervised autoregressive ICIL."""

from icil_jax_rlbench.quickdraw.supervised_train import default_config


def get_config():
    cfg = default_config('autoregressive')
    cfg['output_dir'] = 'outputs/quickdraw_icil/ar_transformer_v1'
    cfg['wandb_project'] = 'icil-quickdraw'
    return cfg
