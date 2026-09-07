from absl import app
from ml_collections import config_flags

from icil_jax_rlbench.eval.metaworld_hidden_goal_gate1 import (
    evaluate_metaworld_gate1,
)

_CONFIG = config_flags.DEFINE_config_file(
    'config', None, 'ML1 Push ordinary-adaptation Gate 1 config.', lock_config=False
)


def main(argv):
    del argv
    evaluate_metaworld_gate1(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
