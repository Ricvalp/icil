from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from phi_mujoco.offline import StandardNormalization

from icil_jax_rlbench.data.metaworld_hidden_goal import (
    SPLITS,
    MetaWorldTaskDataset,
    MetaWorldTaskSampler,
    TaskSplit,
    build_task_episode_split,
    normalization_identifier,
)


class ML1PushTaskDataset(MetaWorldTaskDataset):
    def __init__(
        self,
        cache_root: str | Path,
        *,
        normalization: StandardNormalization | Mapping[str, object] | None = None,
        normalization_eps: float = 1e-4,
        cache_prepared_episodes: bool = True,
    ) -> None:
        super().__init__(
            cache_root,
            integration_name='metaworld_ml1_push',
            normalization=normalization,
            normalization_eps=normalization_eps,
            cache_prepared_episodes=cache_prepared_episodes,
        )


ML1PushTaskSampler = MetaWorldTaskSampler

__all__ = [
    'SPLITS',
    'ML1PushTaskDataset',
    'ML1PushTaskSampler',
    'TaskSplit',
    'build_task_episode_split',
    'normalization_identifier',
]
