"""Full-dataset AR Transformer with three full-second-order pooled KVB updates."""

from icil_jax_rlbench.quickdraw.supervised_train import default_config


def get_config():
    cfg = default_config('autoregressive', method='kvb')
    cfg.update({
        'output_dir': 'outputs/quickdraw_icil/kvb_transformer_v1',
        'micro_batch_size': 4,
        'wandb_project': 'icil-quickdraw',
    })
    return cfg
