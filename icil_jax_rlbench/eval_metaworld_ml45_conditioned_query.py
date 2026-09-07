from absl import app
from ml_collections import config_flags

from icil_jax_rlbench.eval.metaworld_conditioned_query import (
    evaluate_metaworld_conditioned_query,
)

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    None,
    'MetaWorld ML45 explicitly conditioned query-policy evaluation config.',
    lock_config=False,
)


def main(argv):
    del argv
    evaluate_metaworld_conditioned_query(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
