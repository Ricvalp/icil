"""Shared policy dispatch for ordinary ICIL and fast-weight WRITE experiments."""

from pathlib import Path

from . import supervised_models


def model_config(value, method='icil'):
    if method == 'icil':
        return supervised_models.SupervisedModelConfig(**value)
    if method == 'kvb':
        from .kvb_models import KVBModelConfig
        return KVBModelConfig(**value)
    if method == 'support_bc':
        from .support_bc_models import SupportBCModelConfig
        return SupportBCModelConfig(**value)
    raise ValueError("method must be 'icil', 'kvb', or 'support_bc'")


def _backend(cfg):
    if type(cfg) is supervised_models.SupervisedModelConfig:
        return supervised_models
    from . import kvb_models, support_bc_models
    if isinstance(cfg, support_bc_models.SupportBCModelConfig):
        return support_bc_models
    if isinstance(cfg, kvb_models.KVBModelConfig):
        return kvb_models
    raise TypeError(f'Unknown policy config: {type(cfg).__name__}')


def method_name(cfg):
    if _backend(cfg) is supervised_models:
        return 'icil'
    from .support_bc_models import SupportBCModelConfig
    name = 'support_bc' if isinstance(cfg, SupportBCModelConfig) else 'kvb'
    return name + '_first_order' if cfg.first_order else name


def numerical_sources(cfg):
    root = Path(__file__).parent
    sources = {'supervised_models.py': root / 'supervised_models.py',
               'policy_backend.py': Path(__file__)}
    if method_name(cfg) != 'icil':
        sources.update({'kvb_models.py': root / 'kvb_models.py',
                        'fast_weight_ttt.py': root.parent / 'models' / 'fast_weight_ttt.py'})
    if method_name(cfg).startswith('support_bc'):
        sources['support_bc_models.py'] = root / 'support_bc_models.py'
    return sources


def init_model(key, cfg, support_count=4):
    return _backend(cfg).init_model(key, cfg, support_count=support_count)


def loss(params, batch, cfg, key, training=True):
    return _backend(cfg).loss(params, batch, cfg, key, training=training)


def generate(params, support_tokens, support_mask, cfg, key, deterministic=False):
    return _backend(cfg).generate(params, support_tokens, support_mask, cfg, key,
                                  deterministic=deterministic)
