from absl import app
from ml_collections import config_flags

from icil_jax_rlbench.visualization.ttt_state import visualize_ttt_state


_CONFIG = config_flags.DEFINE_config_file(
    'config', None, 'State TTT visualization config.', lock_config=False
)


def main(argv):
    del argv
    visualize_ttt_state(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
