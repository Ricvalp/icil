from icil_jax_rlbench.configs.metaworld_ml10_kvb_delta_read import (
    get_config as _base,
)


def get_config():
    cfg = _base()
    cfg.adaptation.write_objective = 'action_bc'
    return cfg


__all__ = ['get_config']
