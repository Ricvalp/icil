from absl import app
from ml_collections import config_flags

from icil_jax_rlbench.analysis.metaworld_update_information import (
    analyze_metaworld_update_information,
)

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    None,
    'MetaWorld ML45 fast-update information diagnostic config.',
    lock_config=False,
)


def main(argv):
    del argv
    analyze_metaworld_update_information(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
