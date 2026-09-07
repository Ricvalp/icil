from icil_jax_rlbench.configs.ttt_state_base import get_config as _base


def get_config():
    cfg = _base()
    cfg.adaptation.first_order = True
    return cfg
