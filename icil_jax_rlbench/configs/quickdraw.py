"""Shared sketch defaults; set immutable cache/manifest paths before training."""
from ml_collections import ConfigDict

from icil_jax_rlbench.quickdraw.train import default_config


def get_config():
    return ConfigDict(default_config())
