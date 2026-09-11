"""Large KVB Transformer with ~500k fast parameters and 35 held-out classes."""

from icil_jax_rlbench.configs.quickdraw_kvb_transformer_large_heldout import get_config as base_config


def get_config():
    cfg = base_config()
    cfg['model'].update({'fast_dim': 256, 'fast_hidden_dim': 976})
    cfg['output_dir'] = 'outputs/quickdraw_icil/kvb_transformer_large_fast500k_heldout35_v1'
    return cfg
