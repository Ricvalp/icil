"""Shared policy dispatch for ordinary ICIL and fast-weight KVB experiments."""

from pathlib import Path

from . import supervised_models


def model_config(value, method='icil'):
    if method == 'icil':
        return supervised_models.SupervisedModelConfig(**value)
    if method == 'kvb':
        from .kvb_models import KVBModelConfig
        return KVBModelConfig(**value)
    raise ValueError("method must be 'icil' or 'kvb'")


def _backend(cfg):
    if type(cfg) is supervised_models.SupervisedModelConfig:
        return supervised_models
    from . import kvb_models
    if isinstance(cfg, kvb_models.KVBModelConfig):
        return kvb_models
    raise TypeError(f'Unknown policy config: {type(cfg).__name__}')


def method_name(cfg):
    if _backend(cfg) is supervised_models:
        return 'icil'
    return 'kvb_first_order' if cfg.first_order else 'kvb'


def numerical_sources(cfg):
    root = Path(__file__).parent
    sources = {'supervised_models.py': root / 'supervised_models.py',
               'policy_backend.py': Path(__file__)}
    if method_name(cfg) != 'icil':
        sources.update({'kvb_models.py': root / 'kvb_models.py',
                        'fast_weight_ttt.py': root.parent / 'models' / 'fast_weight_ttt.py'})
    return sources


def init_model(key, cfg, support_count=4):
    return _backend(cfg).init_model(key, cfg, support_count=support_count)


def loss(params, batch, cfg, key, training=True):
    return _backend(cfg).loss(params, batch, cfg, key, training=training)


def generate(params, support_tokens, support_mask, cfg, key, deterministic=False):
    return _backend(cfg).generate(params, support_tokens, support_mask, cfg, key,
                                  deterministic=deterministic)
