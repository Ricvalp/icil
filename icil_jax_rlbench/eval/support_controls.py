from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from icil_jax_rlbench.models.fast_weight_ttt import (
    initial_fast_state,
    tree_difference_norm,
    tree_l2_norm,
)


SUPPORTED_CONDITIONS = (
    'no_update',
    'correct_support',
    'wrong_task_support',
    'same_family_wrong_instance',
    'different_family_support',
    'shuffled_actions',
    'shuffled_time',
    'observations_only',
    'actions_only',
    'duplicated_support',
    'random_update_matched_norm',
)


def _shuffle_valid(
    support: dict[str, np.ndarray],
    names: tuple[str, ...],
    rng: np.random.Generator,
) -> None:
    valid = np.asarray(support['write_mask'], dtype=np.bool_)
    count = int(np.count_nonzero(valid))
    if count < 2:
        return
    permutation = rng.permutation(count)
    for name in names:
        values = np.array(support[name][valid], copy=True)
        support[name][valid] = values[permutation]


def condition_support(
    condition: str,
    correct_support: Mapping[str, np.ndarray],
    wrong_support: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Construct one matched support-information control.

    Padding remains fixed. Shuffles operate only on valid support transitions,
    which keeps variable-length MetaWorld batches comparable across conditions.
    """

    wrong_conditions = {
        'wrong_task_support',
        'same_family_wrong_instance',
        'different_family_support',
    }
    source = wrong_support if condition in wrong_conditions else correct_support
    support = {name: np.array(value, copy=True) for name, value in source.items()}
    if condition in {'no_update', 'correct_support', *wrong_conditions}:
        return support
    if condition == 'shuffled_actions':
        _shuffle_valid(support, ('action',), rng)
        return support
    if condition == 'shuffled_time':
        _shuffle_valid(
            support,
            ('observation', 'action', 'next_observation'),
            rng,
        )
        return support
    if condition == 'observations_only':
        support['action'].fill(0.0)
        return support
    if condition == 'actions_only':
        support['observation'].fill(0.0)
        support['next_observation'].fill(0.0)
        return support
    if condition == 'duplicated_support':
        for name in ('observation', 'action', 'next_observation', 'write_mask'):
            support[name] = np.repeat(
                support[name][:1], support[name].shape[0], axis=0
            )
        return support
    if condition == 'random_update_matched_norm':
        return support
    raise ValueError(f'Unknown support condition {condition!r}.')


def random_fast_state_with_matched_delta(
    params: Mapping[str, Any],
    correct_adapted: Any,
    rng: np.random.Generator,
) -> Any:
    initial = initial_fast_state(params)
    target_norm = tree_difference_norm(correct_adapted, initial)
    random_direction = jax.tree_util.tree_map(
        lambda value: jnp.asarray(rng.normal(size=value.shape), dtype=value.dtype),
        initial,
    )
    direction_norm = tree_l2_norm(random_direction)
    scale = target_norm / (direction_norm + 1e-8)
    return jax.tree_util.tree_map(
        lambda value, direction: value + scale.astype(direction.dtype) * direction,
        initial,
        random_direction,
    )


def confidence_interval(successes: int, count: int) -> tuple[float, float]:
    """Wilson 95% interval for a pooled Bernoulli success rate."""

    if count <= 0:
        return 0.0, 0.0
    probability = successes / count
    z = 1.96
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    half_width = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return (
        float(max(0.0, center - half_width)),
        float(min(1.0, center + half_width)),
    )


def bootstrap_mean_confidence_interval(
    values: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> tuple[float, float, float]:
    """Return the mean and a percentile CI over independent task values."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError('At least one value is required for a confidence interval.')
    mean = float(np.mean(array))
    if array.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        array.size,
        size=(int(bootstrap_samples), array.size),
    )
    bootstrap_means = np.mean(array[indices], axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(low), float(high)


__all__ = [
    'SUPPORTED_CONDITIONS',
    'bootstrap_mean_confidence_interval',
    'condition_support',
    'confidence_interval',
    'random_fast_state_with_matched_delta',
]
