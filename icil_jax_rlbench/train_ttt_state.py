from absl import app, logging
from ml_collections import config_flags

from icil_jax_rlbench.train.ttt_runner import train_ttt_state


_CONFIG = config_flags.DEFINE_config_file(
    'config', None, 'Fast-weight TTT state training config.', lock_config=False
)


def main(argv):
    del argv
    logging.set_verbosity(logging.INFO)
    train_ttt_state(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
