from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from phi_mujoco.integrations import family_contract

from icil_jax_rlbench.data.metaworld_hidden_goal import MetaWorldTaskDataset

ConditioningMode = Literal['family', 'family_task_latent']
CONDITIONING_MODES: tuple[ConditioningMode, ...] = (
    'family',
    'family_task_latent',
)
_SCHEMA = 'icil_jax_rlbench.metaworld_conditioning'
_SCHEMA_VERSION = 1


def _declared_task_latent(
    dataset: MetaWorldTaskDataset,
    task_id: str,
    *,
    latent_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    descriptor = dataset.task_descriptor(task_id)
    family = dataset.task_family(task_id)
    native_rand_vec = np.asarray(
        descriptor.get('native_rand_vec', ()), dtype=np.float32
    )
    if native_rand_vec.ndim != 1 or not np.all(np.isfinite(native_rand_vec)):
        raise ValueError(f'Task {task_id!r} has an invalid native reset vector.')

    contract = family_contract(family)
    if contract.reset_vector_dim != native_rand_vec.size:
        raise ValueError(
            f'Task {task_id!r} reset-vector dimension differs from the public '
            f'{family!r} family contract.'
        )
    fields = tuple(
        field for field in contract.reset_fields if field.role == 'task_latent'
    )
    components = [
        native_rand_vec[field.start : field.stop]
        for field in fields
    ]
    compact = (
        np.concatenate(components, axis=0)
        if components
        else np.zeros((0,), dtype=np.float32)
    )
    if compact.size > int(latent_dim):
        raise ValueError(
            f'Task {task_id!r} needs {compact.size} task-latent values, but the '
            f'conditioning schema allows {latent_dim}.'
        )
    values = np.zeros((int(latent_dim),), dtype=np.float32)
    mask = np.zeros_like(values)
    values[: compact.size] = compact
    mask[: compact.size] = 1.0
    return values, mask


def _maximum_task_latent_width(families: Sequence[str]) -> int:
    return max(
        (
            sum(
                field.stop - field.start
                for field in family_contract(family).reset_fields
                if field.role == 'task_latent'
            )
            for family in families
        ),
        default=0,
    )


@dataclass(frozen=True, slots=True)
class MetaWorldConditioning:
    """Explicit ML45 family/task-latent input for diagnostic baselines."""

    mode: ConditioningMode
    family_names: tuple[str, ...]
    training_families: tuple[str, ...]
    train_split: str
    latent_dim: int
    latent_mean: tuple[float, ...]
    latent_std: tuple[float, ...]
    normalization_eps: float

    def __post_init__(self) -> None:
        if self.mode not in CONDITIONING_MODES:
            raise ValueError(
                f'conditioning mode must be one of {CONDITIONING_MODES}.'
            )
        if not self.family_names or len(set(self.family_names)) != len(
            self.family_names
        ):
            raise ValueError('family_names must be non-empty and unique.')
        if not set(self.training_families).issubset(self.family_names):
            raise ValueError('training_families must be present in family_names.')
        if int(self.latent_dim) < 0:
            raise ValueError('latent_dim cannot be negative.')
        if (
            len(self.latent_mean) != self.latent_dim
            or len(self.latent_std) != self.latent_dim
        ):
            raise ValueError('Task-latent normalization has the wrong dimension.')
        if self.normalization_eps <= 0.0:
            raise ValueError('normalization_eps must be positive.')
        if any(value < self.normalization_eps for value in self.latent_std):
            raise ValueError('Task-latent standard deviations must be clamped by eps.')

    @classmethod
    def fit(
        cls,
        dataset: MetaWorldTaskDataset,
        *,
        mode: str,
        train_split: str = 'train',
        normalization_eps: float = 1e-4,
    ) -> MetaWorldConditioning:
        if dataset.integration_name != 'metaworld_ml45':
            raise ValueError(
                'Explicit task conditioning is currently defined for ML45.'
            )
        if mode not in CONDITIONING_MODES:
            raise ValueError(f'conditioning mode must be one of {CONDITIONING_MODES}.')
        eps = float(normalization_eps)
        if eps <= 0.0:
            raise ValueError('normalization_eps must be positive.')

        family_names = tuple(
            dict.fromkeys(
                str(task.family) for task in dataset.task_index.catalog.tasks
            )
        )
        train_task_ids = dataset.task_ids(train_split)
        if not train_task_ids:
            raise ValueError(f'Conditioning train split {train_split!r} is empty.')
        training_families = tuple(
            family
            for family in family_names
            if any(dataset.task_family(task_id) == family for task_id in train_task_ids)
        )
        latent_dim = _maximum_task_latent_width(family_names)

        values = []
        masks = []
        for task_id in train_task_ids:
            latent, mask = _declared_task_latent(
                dataset, task_id, latent_dim=latent_dim
            )
            values.append(latent)
            masks.append(mask)
        latent_values = np.stack(values, axis=0)
        latent_masks = np.stack(masks, axis=0)
        counts = np.sum(latent_masks, axis=0)
        means = np.divide(
            np.sum(latent_values * latent_masks, axis=0),
            counts,
            out=np.zeros((latent_dim,), dtype=np.float32),
            where=counts > 0,
        )
        variances = np.divide(
            np.sum(((latent_values - means) * latent_masks) ** 2, axis=0),
            counts,
            out=np.ones((latent_dim,), dtype=np.float32),
            where=counts > 0,
        )
        standard_deviations = np.maximum(np.sqrt(variances), eps)
        return cls(
            mode=mode,
            family_names=family_names,
            training_families=training_families,
            train_split=str(train_split),
            latent_dim=latent_dim,
            latent_mean=tuple(float(value) for value in means),
            latent_std=tuple(float(value) for value in standard_deviations),
            normalization_eps=eps,
        )

    @property
    def context_dim(self) -> int:
        if self.mode == 'family':
            return len(self.family_names)
        return len(self.family_names) + 2 * self.latent_dim

    def context(
        self,
        dataset: MetaWorldTaskDataset,
        task_id: str,
    ) -> np.ndarray:
        family = dataset.task_family(task_id)
        try:
            family_index = self.family_names.index(family)
        except ValueError as exc:
            raise ValueError(
                f'Task family {family!r} is absent from the conditioning schema.'
            ) from exc
        one_hot = np.zeros((len(self.family_names),), dtype=np.float32)
        one_hot[family_index] = 1.0
        if self.mode == 'family':
            return one_hot

        latent, mask = _declared_task_latent(
            dataset, task_id, latent_dim=self.latent_dim
        )
        mean = np.asarray(self.latent_mean, dtype=np.float32)
        std = np.asarray(self.latent_std, dtype=np.float32)
        normalized = ((latent - mean) / std) * mask
        return np.concatenate((one_hot, normalized, mask), axis=0)

    def augment_observations(
        self,
        dataset: MetaWorldTaskDataset,
        observations: np.ndarray,
        task_ids: Sequence[str],
    ) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim < 2 or values.shape[0] != len(task_ids):
            raise ValueError(
                'Observation task axis must match the number of task IDs; got '
                f'{values.shape} and {len(task_ids)} IDs.'
            )
        if values.shape[-1] != dataset.observation_dim:
            raise ValueError(
                f'Expected raw observation dimension {dataset.observation_dim}, '
                f'got {values.shape[-1]}.'
            )
        contexts = np.stack(
            [self.context(dataset, task_id) for task_id in task_ids], axis=0
        )
        # Broadcast [task, context] across every demonstration/time axis.
        context_shape = (
            contexts.shape[0],
            *(1 for _ in values.shape[1:-1]),
            contexts.shape[-1],
        )
        broadcast = np.broadcast_to(
            contexts.reshape(context_shape), (*values.shape[:-1], self.context_dim)
        )
        return np.concatenate((values, broadcast), axis=-1).astype(
            np.float32, copy=False
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema': _SCHEMA,
            'schema_version': _SCHEMA_VERSION,
            'mode': self.mode,
            'family_names': list(self.family_names),
            'training_families': list(self.training_families),
            'train_split': self.train_split,
            'latent_dim': self.latent_dim,
            'latent_mean': list(self.latent_mean),
            'latent_std': list(self.latent_std),
            'normalization_eps': self.normalization_eps,
            'contains_task_id': False,
            'contains_episode_nuisance': False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MetaWorldConditioning:
        if (
            value.get('schema') != _SCHEMA
            or value.get('schema_version') != _SCHEMA_VERSION
        ):
            raise ValueError('Unsupported MetaWorld conditioning schema.')
        return cls(
            mode=str(value['mode']),  # type: ignore[arg-type]
            family_names=tuple(str(item) for item in value['family_names']),
            training_families=tuple(
                str(item) for item in value['training_families']
            ),
            train_split=str(value['train_split']),
            latent_dim=int(value['latent_dim']),
            latent_mean=tuple(float(item) for item in value['latent_mean']),
            latent_std=tuple(float(item) for item in value['latent_std']),
            normalization_eps=float(value['normalization_eps']),
        )

    def validate_dataset(self, dataset: MetaWorldTaskDataset) -> None:
        expected = type(self).fit(
            dataset,
            mode=self.mode,
            train_split=self.train_split,
            normalization_eps=self.normalization_eps,
        )
        if self != expected:
            raise ValueError(
                'Conditioning metadata differs from the requested cache or task split.'
            )


__all__ = [
    'CONDITIONING_MODES',
    'ConditioningMode',
    'MetaWorldConditioning',
]
