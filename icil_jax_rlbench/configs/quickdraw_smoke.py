"""Small correctness runs; this capacity is not a research recommendation."""

from icil_jax_rlbench.quickdraw.train import default_config


def get_config():
    cfg = default_config()
    cfg.update(batch_size=1, support_count=2, query_count=1, num_steps=2,
               log_every=1, checkpoint_every=2)
    cfg['model'] = dict(hidden_dim=8, fast_dim=4, fast_hidden_dim=8,
                        mixture_components=2, max_steps=32, segment_size=8)
    return cfg
