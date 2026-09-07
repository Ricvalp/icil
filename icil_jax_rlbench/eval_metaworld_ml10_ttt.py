from absl import app
from ml_collections import config_flags

from icil_jax_rlbench.eval.metaworld_hidden_goal_ttt import evaluate_metaworld_ttt

_CONFIG = config_flags.DEFINE_config_file(
    'config', None, 'MetaWorld ML10 fast-weight TTT evaluation config.', lock_config=False
)


def main(argv):
    del argv
    evaluate_metaworld_ttt(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
