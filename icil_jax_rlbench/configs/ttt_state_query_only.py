from icil_jax_rlbench.configs.ttt_state_base import get_config as _base


def get_config():
    cfg = _base()
    cfg.conditioning.fast_weight_write = False
    cfg.conditioning.fast_weight_read = False
    return cfg
