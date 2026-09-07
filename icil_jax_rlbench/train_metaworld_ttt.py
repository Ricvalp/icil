from absl import app
from ml_collections import config_flags

from icil_jax_rlbench.train.metaworld_ttt_runner import (
    train_metaworld_ttt,
)

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    None,
    'MetaWorld hidden-goal fast-weight TTT training config.',
    lock_config=False,
)


def main(argv):
    del argv
    train_metaworld_ttt(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
