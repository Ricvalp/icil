"""Larger query Transformer with full KVB WRITE and 35 held-out classes."""

from icil_jax_rlbench.configs.quickdraw_kvb_transformer_heldout import get_config as base_config


def get_config():
    cfg = base_config()
    cfg['model'].update({'hidden_dim': 384, 'decoder_layers': 8,
                         'num_heads': 8, 'context_layers': 6})
    cfg['micro_batch_size'] = 2
    cfg['output_dir'] = 'outputs/quickdraw_icil/kvb_transformer_large_heldout35_v1'
    return cfg
