from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from phi_mujoco.integrations.metaworld_ml1_push import (
    MetaWorldML1PushIntegration,
    load_ml1_push_task_index,
)
from phi_mujoco.integrations.metaworld_ml1_reach import (
    MetaWorldML1ReachIntegration,
    load_ml1_reach_task_index,
)
from phi_mujoco.integrations.metaworld_ml10 import (
    MetaWorldML10Integration,
    load_ml10_task_index,
)
from phi_mujoco.integrations.metaworld_ml45 import (
    MetaWorldML45Integration,
    load_ml45_task_index,
)
from phi_mujoco.offline import (
    EpisodeSplit,
    SplitConfig,
    StandardNormalization,
    ValidatedCollection,
    fit_standard_normalization,
    validate_collection_bundle,
)

TaskSplit = Literal[
    'train', 'latent_validation', 'family_validation', 'validation', 'test'
]
SPLITS: tuple[TaskSplit, ...] = (
    'train',
    'latent_validation',
    'family_validation',
    'validation',
    'test',
)


@dataclass(frozen=True, slots=True)
class MetaWorldHiddenGoalBenchmark:
    integration_name: str
    label: str
    slug: str
    task_index_loader: Any
    integration_type: Any
    protocols: tuple[str, ...] = ('default',)
    split_names: tuple[TaskSplit, ...] = ('train', 'validation', 'test')
    family_aware: bool = False

    @property
    def query_mode(self) -> str:
        return f'{self.integration_name}_query_only'

    @property
    def ttt_mode(self) -> str:
        return f'{self.integration_name}_ttt'

    def create_integration(self, *, catalog_seed: int):
        return self.integration_type(catalog_seed=int(catalog_seed))

    def task_ids(
        self,
        task_index: Any,
        split: TaskSplit,
        *,
        protocol: str,
    ) -> tuple[str, ...]:
        if protocol not in self.protocols:
            raise ValueError(
                f'{self.label} protocol must be one of {self.protocols}, got {protocol!r}.'
            )
        if split not in self.split_names:
            raise ValueError(
                f'{self.label} split must be one of {self.split_names}, got {split!r}.'
            )
        if self.family_aware:
            return task_index.task_ids(split, protocol=protocol)
        return task_index.task_ids(split)

    def family(self, task_index: Any, task_id: str) -> str:
        if self.family_aware:
            return str(task_index.family(task_id))
        return self.slug


_BENCHMARKS = {
    benchmark.integration_name: benchmark
    for benchmark in (
        MetaWorldHiddenGoalBenchmark(
            integration_name='metaworld_ml1_reach',
            label='ML1 Reach',
            slug='ml1_reach',
            task_index_loader=load_ml1_reach_task_index,
            integration_type=MetaWorldML1ReachIntegration,
        ),
        MetaWorldHiddenGoalBenchmark(
            integration_name='metaworld_ml1_push',
            label='ML1 Push',
            slug='ml1_push',
            task_index_loader=load_ml1_push_task_index,
            integration_type=MetaWorldML1PushIntegration,
        ),
        MetaWorldHiddenGoalBenchmark(
            integration_name='metaworld_ml10',
            label='ML10',
            slug='ml10',
            task_index_loader=load_ml10_task_index,
            integration_type=MetaWorldML10Integration,
            protocols=('development', 'final'),
            split_names=(
                'train',
                'latent_validation',
                'family_validation',
                'validation',
                'test',
            ),
            family_aware=True,
        ),
        MetaWorldHiddenGoalBenchmark(
            integration_name='metaworld_ml45',
            label='ML45',
            slug='ml45',
            task_index_loader=load_ml45_task_index,
            integration_type=MetaWorldML45Integration,
            protocols=('development', 'final'),
            split_names=(
                'train',
                'latent_validation',
                'family_validation',
                'validation',
                'test',
            ),
            family_aware=True,
        ),
    )
}


def benchmark_for_integration(name: str) -> MetaWorldHiddenGoalBenchmark:
    try:
        return _BENCHMARKS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f'Unsupported MetaWorld hidden-goal integration {name!r}; '
            f'choose one of {tuple(_BENCHMARKS)}.'
        ) from exc


def benchmark_from_config(cfg: Any) -> MetaWorldHiddenGoalBenchmark:
    integration_name = str(cfg.dataset.get('integration', ''))
    if integration_name:
        return benchmark_for_integration(integration_name)
    mode = str(cfg.mode)
    matches = [
        benchmark
        for benchmark in _BENCHMARKS.values()
        if mode in (benchmark.query_mode, benchmark.ttt_mode)
    ]
    if len(matches) != 1:
        raise ValueError('Config must declare dataset.integration explicitly.')
    return matches[0]


def _episode_indices(
    task_index: Any,
    task_ids: Sequence[str],
) -> tuple[int, ...]:
    return tuple(
        episode_index
        for task_id in task_ids
        for episode_index in task_index.episode_indices(task_id)
    )


def build_task_episode_split(
    bundle: ValidatedCollection,
    task_index: Any,
    *,
    benchmark: MetaWorldHiddenGoalBenchmark | None = None,
    protocol: str = 'default',
) -> EpisodeSplit:
    """Translate the declared task split into phi-mujoco episode indices."""

    if bundle.episode_count != len(task_index.episodes):
        raise ValueError('Task metadata and processed bundle episode counts differ.')
    if any(not episode.success for episode in bundle.episodes):
        raise ValueError(
            'MetaWorld hidden-goal imitation caches must contain successful '
            'episodes only.'
        )
    resolved_benchmark = benchmark or benchmark_for_integration(bundle.spec.name)
    train = _episode_indices(
        task_index,
        resolved_benchmark.task_ids(task_index, 'train', protocol=protocol),
    )
    validation = _episode_indices(
        task_index,
        resolved_benchmark.task_ids(task_index, 'validation', protocol=protocol),
    )
    test = _episode_indices(
        task_index,
        resolved_benchmark.task_ids(task_index, 'test', protocol=protocol),
    )
    partitions = (set(train), set(validation), set(test))
    if any(
        left & right
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    ):
        raise ValueError('MetaWorld hidden-goal task splits contain overlapping episodes.')
    eligible = tuple(sorted(train + validation + test))
    expected = tuple(range(bundle.episode_count))
    if eligible != expected:
        raise ValueError(
            'MetaWorld hidden-goal task metadata does not cover every cache '
            'episode exactly once.'
        )
    return EpisodeSplit(
        config=SplitConfig(),
        successful_only=True,
        eligible_episode_indices=eligible,
        train_episode_indices=tuple(sorted(train)),
        val_episode_indices=tuple(sorted(validation)),
        test_episode_indices=tuple(sorted(test)),
    )


def normalization_identifier(
    normalization: StandardNormalization,
    *,
    data_sha256: str,
    slug: str = 'ml1_reach',
) -> str:
    payload = {
        'data_sha256': str(data_sha256),
        'normalization': normalization.to_dict(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return f'{slug}_train_{digest[:12]}'


class MetaWorldTaskDataset:
    """Task-aware, simulator-free policy view of one processed ML1 cache."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        integration_name: str,
        protocol: str = 'default',
        horizon_buckets: Sequence[int] = (),
        normalization: StandardNormalization | Mapping[str, object] | None = None,
        normalization_eps: float = 1e-4,
        cache_prepared_episodes: bool = True,
    ) -> None:
        self.benchmark = benchmark_for_integration(integration_name)
        self.integration_name = self.benchmark.integration_name
        self.protocol = str(protocol)
        if self.protocol not in self.benchmark.protocols:
            raise ValueError(
                f'{self.benchmark.label} protocol must be one of '
                f'{self.benchmark.protocols}, got {self.protocol!r}.'
            )
        buckets = tuple(sorted({int(value) for value in horizon_buckets}))
        if any(value < 1 for value in buckets):
            raise ValueError('horizon_buckets must contain positive integers.')
        self.horizon_buckets = buckets
        self.bundle = validate_collection_bundle(
            Path(cache_root).expanduser().resolve(),
            integration=self.integration_name,
        )
        self.task_index = self.benchmark.task_index_loader(self.bundle)
        self.episode_split = build_task_episode_split(
            self.bundle,
            self.task_index,
            benchmark=self.benchmark,
            protocol=self.protocol,
        )
        if self.bundle.spec.observations['state'].shape != (39,):
            raise ValueError(
                'MetaWorld hidden-goal policy observations must be 39-dimensional.'
            )
        if self.bundle.spec.action.shape != (4,):
            raise ValueError('MetaWorld hidden-goal policy actions must be four-dimensional.')

        if isinstance(normalization, Mapping):
            resolved_normalization = StandardNormalization.from_dict(normalization)
        elif normalization is None:
            resolved_normalization = fit_standard_normalization(
                self.bundle,
                self.episode_split,
                observation_modality='state',
                eps=float(normalization_eps),
            )
        elif isinstance(normalization, StandardNormalization):
            resolved_normalization = normalization
        else:
            raise TypeError(
                'normalization must be StandardNormalization, a mapping, or None.'
            )
        if (
            resolved_normalization.observation_modality != 'state'
            or resolved_normalization.source_episode_indices
            != self.episode_split.train_episode_indices
        ):
            raise ValueError(
                'Normalization must be fitted exclusively from declared training-task episodes.'
            )
        if len(resolved_normalization.obs_mean) != 39:
            raise ValueError('Observation normalization has the wrong dimension.')
        if len(resolved_normalization.action_mean) != 4:
            raise ValueError('Action normalization has the wrong dimension.')

        self.normalization = resolved_normalization
        self.normalization_id = normalization_identifier(
            resolved_normalization,
            data_sha256=self.bundle.data_sha256,
            slug=self.benchmark.slug,
        )
        self.horizon = max(episode.steps for episode in self.bundle.episodes)
        self._cache_prepared_episodes = bool(cache_prepared_episodes)
        self._prepared_episodes: dict[int, dict[str, np.ndarray]] = {}

    @property
    def observation_dim(self) -> int:
        return 39

    @property
    def action_dim(self) -> int:
        return 4

    def task_ids(self, split: TaskSplit) -> tuple[str, ...]:
        return self.benchmark.task_ids(
            self.task_index, split, protocol=self.protocol
        )

    def task_family(self, task_id: str) -> str:
        return self.benchmark.family(self.task_index, task_id)

    def balanced_task_ids(self, split: TaskSplit, limit: int = 0) -> tuple[str, ...]:
        """Select a bounded evaluation screen evenly across task families."""

        task_ids = self.task_ids(split)
        limit = int(limit)
        if limit <= 0 or limit >= len(task_ids):
            return task_ids
        by_family: dict[str, list[str]] = {}
        for task_id in task_ids:
            by_family.setdefault(self.task_family(task_id), []).append(task_id)
        selected = []
        family_offset = 0
        while len(selected) < limit:
            progressed = False
            for family_tasks in by_family.values():
                if family_offset < len(family_tasks):
                    selected.append(family_tasks[family_offset])
                    progressed = True
                    if len(selected) == limit:
                        break
            if not progressed:
                break
            family_offset += 1
        return tuple(selected)

    def task_descriptor(self, task_id: str) -> dict[str, object]:
        task = self.task_index.catalog.task(task_id)
        if hasattr(task, 'to_dict'):
            return dict(task.to_dict())
        return {'task_id': task_id}

    def same_family_wrong_task(self, task_id: str, split: TaskSplit) -> str:
        family = self.task_family(task_id)
        candidates = tuple(
            candidate
            for candidate in self.task_ids(split)
            if candidate != task_id and self.task_family(candidate) == family
        )
        if not candidates:
            raise ValueError(
                f'Split {split!r} has no second task in family {family!r}.'
            )
        return candidates[0]

    def different_family_task(self, task_id: str, split: TaskSplit) -> str:
        family = self.task_family(task_id)
        candidates = tuple(
            candidate
            for candidate in self.task_ids(split)
            if self.task_family(candidate) != family
        )
        if not candidates:
            raise ValueError(
                f'Split {split!r} has no task outside family {family!r}.'
            )
        return candidates[0]

    def resolve_horizon(self, episode_indices: Sequence[int]) -> int:
        if not episode_indices:
            raise ValueError('At least one episode is required to resolve a horizon.')
        required = max(self.bundle.episodes[int(index)].steps for index in episode_indices)
        if not self.horizon_buckets:
            return self.horizon
        for bucket in self.horizon_buckets:
            if required <= bucket:
                return bucket
        raise ValueError(
            f'Episode length {required} exceeds horizon buckets {self.horizon_buckets}.'
        )

    def _prepare_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        episode_index = int(episode_index)
        cached = self._prepared_episodes.get(episode_index)
        if cached is not None:
            return cached
        episode = self.bundle.load_episode(episode_index)
        steps = int(episode.actions.shape[0])
        observations = self.normalization.normalize_observations(
            episode.observations['state']
        )
        actions = self.normalization.normalize_actions(episode.actions)
        prepared = {
            'observation': np.asarray(observations[:-1], dtype=np.float32),
            'action': np.asarray(actions, dtype=np.float32),
            'next_observation': np.asarray(observations[1:], dtype=np.float32),
            'valid': np.ones((steps,), dtype=np.bool_),
            'episode_id': np.asarray(episode_index, dtype=np.int32),
            'episode_seed': np.asarray(episode.seed, dtype=np.uint32),
        }
        if self._cache_prepared_episodes:
            self._prepared_episodes[episode_index] = prepared
        return prepared

    def _stack_episodes(
        self, episode_indices: Sequence[int], *, horizon: int
    ) -> dict[str, np.ndarray]:
        episodes = [self._prepare_episode(index) for index in episode_indices]
        result: dict[str, np.ndarray] = {}
        for name in episodes[0]:
            if name in {'episode_id', 'episode_seed'}:
                result[name] = np.stack([episode[name] for episode in episodes], axis=0)
                continue
            trailing_shape = episodes[0][name].shape[1:]
            values = np.zeros(
                (len(episodes), int(horizon), *trailing_shape),
                dtype=episodes[0][name].dtype,
            )
            for index, episode in enumerate(episodes):
                steps = episode[name].shape[0]
                if steps > horizon:
                    raise ValueError(
                        f'Episode {episode_indices[index]} has {steps} steps, '
                        f'exceeding horizon {horizon}.'
                    )
                values[index, :steps] = episode[name]
            result[name] = values
        return result

    def section(
        self,
        episode_indices: Sequence[int],
        *,
        support: bool,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray]:
        if not episode_indices:
            raise ValueError('An episode section must contain at least one episode.')
        resolved_horizon = (
            self.resolve_horizon(episode_indices) if horizon is None else int(horizon)
        )
        section = self._stack_episodes(
            episode_indices, horizon=resolved_horizon
        )
        if support:
            section['write_mask'] = np.array(section['valid'], copy=True)
            section['outer_loss_mask'] = np.zeros_like(section['valid'])
        else:
            section['outer_loss_mask'] = np.array(section['valid'], copy=True)
        return section

    def provenance(self) -> dict[str, object]:
        return {
            'name': self.integration_name,
            'cache_root': str(self.bundle.root),
            'cache_data_sha256': self.bundle.data_sha256,
            'task_catalog_seed': self.task_index.catalog.catalog_seed,
            'protocol': self.protocol,
            'task_split_sizes': {
                split: len(self.task_ids(split))
                for split in self.benchmark.split_names
            },
            'episodes_per_task': self.task_index.episodes_per_task,
            'maximum_episode_steps': self.horizon,
            'horizon_buckets': list(self.horizon_buckets),
            'normalizer_id': self.normalization_id,
            'normalizer_fit_split': 'train_tasks_only',
            'support_query_separate_episodes': True,
            'goal_provided_to_policy': False,
        }

    def integrity_report(self) -> dict[str, object]:
        train_indices = set(self.episode_split.train_episode_indices)
        normalization_indices = set(self.normalization.source_episode_indices)
        return {
            **self.provenance(),
            'episode_count': self.bundle.episode_count,
            'expert_success_rate': (
                self.bundle.successful_episodes / self.bundle.episode_count
            ),
            'unique_episode_seeds': (
                len({episode.seed for episode in self.bundle.episodes})
                == self.bundle.episode_count
            ),
            'normalizer_uses_exact_training_task_episodes': (
                normalization_indices == train_indices
            ),
            'observation_mean': list(self.normalization.obs_mean),
            'observation_std': list(self.normalization.obs_std),
            'action_mean': list(self.normalization.action_mean),
            'action_std': list(self.normalization.action_std),
        }


class MetaWorldTaskSampler:
    """Sample explicit task, demonstration, and time axes from a fixed cache."""

    def __init__(
        self,
        dataset: MetaWorldTaskDataset,
        *,
        split: TaskSplit,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.task_ids = dataset.task_ids(split)
        if not self.task_ids:
            raise ValueError(
                f'Split {split!r} is empty under protocol {dataset.protocol!r}.'
            )
        self.rng = np.random.default_rng(int(seed))
        self._tasks_by_family: dict[str, tuple[str, ...]] = {}
        for task_id in self.task_ids:
            family = dataset.task_family(task_id)
            self._tasks_by_family.setdefault(family, ())
            self._tasks_by_family[family] = (*self._tasks_by_family[family], task_id)

    def _resolve_tasks(
        self,
        batch_size: int,
        task_ids: Sequence[str] | None,
    ) -> tuple[str, ...]:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        if task_ids is None:
            families = tuple(self._tasks_by_family)
            selected = []
            for family_index in self.rng.integers(
                0, len(families), size=(batch_size,)
            ):
                family_tasks = self._tasks_by_family[families[int(family_index)]]
                selected.append(
                    family_tasks[int(self.rng.integers(0, len(family_tasks)))]
                )
            return tuple(selected)
        resolved = tuple(str(task_id) for task_id in task_ids)
        if len(resolved) != batch_size:
            raise ValueError(
                f'task_ids must contain {batch_size} values, got {len(resolved)}.'
            )
        invalid = set(resolved) - set(self.task_ids)
        if invalid:
            raise ValueError(
                f'Tasks do not belong to split {self.split!r}: {sorted(invalid)}.'
            )
        return resolved

    def _choose_episodes(self, task_id: str, count: int) -> tuple[int, ...]:
        available = self.dataset.task_index.episode_indices(task_id)
        if int(count) > len(available):
            raise ValueError(
                f'Task {task_id!r} has {len(available)} episodes, fewer than {count} requested.'
            )
        selected = self.rng.choice(available, size=int(count), replace=False)
        return tuple(int(index) for index in selected)

    @staticmethod
    def _stack_tasks(
        tasks: Sequence[Mapping[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        return {
            name: np.stack([task[name] for task in tasks], axis=0)
            for name in tasks[0]
        }

    def build_batch(
        self,
        batch_size: int,
        *,
        support_episodes: int,
        query_episodes: int,
        task_ids: Sequence[str] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        support_episodes = int(support_episodes)
        query_episodes = int(query_episodes)
        if support_episodes < 1 or query_episodes < 1:
            raise ValueError('support_episodes and query_episodes must be positive.')
        selected_tasks = self._resolve_tasks(batch_size, task_ids)
        selections = [
            self._choose_episodes(task_id, support_episodes + query_episodes)
            for task_id in selected_tasks
        ]
        horizon = self.dataset.resolve_horizon(
            tuple(index for selection in selections for index in selection)
        )
        support_tasks = []
        query_tasks = []
        for episode_indices in selections:
            support_tasks.append(
                self.dataset.section(
                    episode_indices[:support_episodes],
                    support=True,
                    horizon=horizon,
                )
            )
            query_tasks.append(
                self.dataset.section(
                    episode_indices[support_episodes:],
                    support=False,
                    horizon=horizon,
                )
            )
        return {
            'support': self._stack_tasks(support_tasks),
            'query': self._stack_tasks(query_tasks),
        }

    def build_query_batch(
        self,
        batch_size: int,
        *,
        query_episodes: int,
        task_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        query_episodes = int(query_episodes)
        if query_episodes < 1:
            raise ValueError('query_episodes must be positive.')
        selected_tasks = self._resolve_tasks(batch_size, task_ids)
        selections = [
            self._choose_episodes(task_id, query_episodes)
            for task_id in selected_tasks
        ]
        horizon = self.dataset.resolve_horizon(
            tuple(index for selection in selections for index in selection)
        )
        query_tasks = [
            self.dataset.section(
                episode_indices,
                support=False,
                horizon=horizon,
            )
            for episode_indices in selections
        ]
        return {
            'query': self._stack_tasks(query_tasks),
            'task_ids': selected_tasks,
        }


__all__ = [
    'SPLITS',
    'MetaWorldHiddenGoalBenchmark',
    'MetaWorldTaskDataset',
    'MetaWorldTaskSampler',
    'TaskSplit',
    'benchmark_for_integration',
    'benchmark_from_config',
    'build_task_episode_split',
    'normalization_identifier',
]
