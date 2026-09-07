from icil_jax_rlbench.configs.metaworld_ml45_ttt_base import get_config as _base


def get_config():
    cfg = _base()
    cfg.adaptation.write_objective = 'action_bc'
    return cfg


__all__ = ['get_config']

