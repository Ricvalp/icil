from absl import app
from ml_collections import config_flags

from icil_jax_rlbench.train.metaworld_query_runner import (
    train_metaworld_query_only,
)

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    None,
    'MetaWorld hidden-goal query-only behavior cloning config.',
    lock_config=False,
)


def main(argv):
    del argv
    train_metaworld_query_only(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
