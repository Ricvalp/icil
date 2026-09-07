from icil_jax_rlbench.configs.metaworld_ml10_ttt_base import get_config as _base


def get_config():
    cfg = _base()
    cfg.adaptation.read_mode = 'delta'
    return cfg


__all__ = ['get_config']
