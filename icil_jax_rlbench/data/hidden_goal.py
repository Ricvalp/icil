from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence

import numpy as np


SPLITS = ('train', 'validation', 'test')


@dataclass(frozen=True)
class HiddenGoalConfig:
    """Small partially observable reach-and-grasp benchmark.

    The task latent is a 2-D goal. It is shared by support and query episodes,
    but deliberately omitted from observations. Initial states and episode IDs
    are independent of the latent and distinct between support and query.
    """

    seed: int = 0
    num_train_tasks: int = 48
    num_validation_tasks: int = 12
    num_test_tasks: int = 12
    horizon: int = 16
    support_episodes: int = 2
    query_episodes: int = 2
    world_limit: float = 1.0
    goal_limit: float = 0.65
    initial_limit: float = 0.85
    max_translation: float = 0.16
    expert_gain: float = 0.8
    close_radius: float = 0.10
    success_radius: float = 0.11
    expert_action_noise: float = 0.0
    min_goal_separation: float = 0.10

    @property
    def observation_dim(self) -> int:
        # Position, gripper state, and normalized phase. The goal is absent.
        return 4

    @property
    def action_dim(self) -> int:
        # Normalized planar delta and a binary gripper target.
        return 3


@dataclass(frozen=True)
class StateNormalizer:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    identifier: str

    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(observation, dtype=np.float32) - self.observation_mean)
            / self.observation_std
        ).astype(np.float32)

    def denormalize_observation(self, observation: np.ndarray) -> np.ndarray:
        return (
            np.asarray(observation, dtype=np.float32) * self.observation_std
            + self.observation_mean
        ).astype(np.float32)

    def to_dict(self) -> Dict[str, object]:
        return {
            'observation_mean': self.observation_mean.tolist(),
            'observation_std': self.observation_std.tolist(),
            'identifier': self.identifier,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> 'StateNormalizer':
        return cls(
            observation_mean=np.asarray(value['observation_mean'], dtype=np.float32),
            observation_std=np.asarray(value['observation_std'], dtype=np.float32),
            identifier=str(value['identifier']),
        )


class HiddenGoalTaskBank:
    def __init__(self, cfg: HiddenGoalConfig):
        self.cfg = cfg
        counts = {
            'train': int(cfg.num_train_tasks),
            'validation': int(cfg.num_validation_tasks),
            'test': int(cfg.num_test_tasks),
        }
        total = sum(counts.values())
        goals = self._sample_separated_goals(total)
        self._goals: Dict[str, np.ndarray] = {}
        self._global_ids: Dict[str, np.ndarray] = {}
        offset = 0
        for split in SPLITS:
            count = counts[split]
            self._goals[split] = goals[offset : offset + count]
            self._global_ids[split] = np.arange(offset, offset + count, dtype=np.int32)
            offset += count

    def _sample_separated_goals(self, count: int) -> np.ndarray:
        rng = np.random.default_rng(int(self.cfg.seed))
        accepted = []
        attempts = 0
        max_attempts = max(10_000, count * 2_000)
        while len(accepted) < count and attempts < max_attempts:
            attempts += 1
            candidate = rng.uniform(
                -float(self.cfg.goal_limit),
                float(self.cfg.goal_limit),
                size=(2,),
            ).astype(np.float32)
            if not accepted:
                accepted.append(candidate)
                continue
            distance = np.linalg.norm(np.asarray(accepted) - candidate[None, :], axis=-1)
            if np.all(distance >= float(self.cfg.min_goal_separation)):
                accepted.append(candidate)
        if len(accepted) != count:
            raise RuntimeError(
                f'Could only construct {len(accepted)}/{count} separated goals. '
                'Reduce min_goal_separation or the task count.'
            )
        return np.stack(accepted, axis=0).astype(np.float32)

    def goals(self, split: str) -> np.ndarray:
        self._validate_split(split)
        return np.array(self._goals[split], copy=True)

    def goal(self, split: str, local_task_id: int) -> np.ndarray:
        self._validate_split(split)
        return np.array(self._goals[split][int(local_task_id)], copy=True)

    def global_task_id(self, split: str, local_task_id: int) -> int:
        self._validate_split(split)
        return int(self._global_ids[split][int(local_task_id)])

    def num_tasks(self, split: str) -> int:
        self._validate_split(split)
        return int(self._goals[split].shape[0])

    @staticmethod
    def _validate_split(split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f'split must be one of {SPLITS}, got {split!r}.')


class HiddenGoalEnvironment:
    def __init__(self, cfg: HiddenGoalConfig, goal: np.ndarray, episode_id: int):
        self.cfg = cfg
        self.goal = np.asarray(goal, dtype=np.float32)
        self.episode_id = int(episode_id)
        rng = np.random.default_rng(self.episode_id)
        self.position = rng.uniform(
            -float(cfg.initial_limit),
            float(cfg.initial_limit),
            size=(2,),
        ).astype(np.float32)
        self.gripper = np.float32(0.0)
        self.t = 0

    def observation(self) -> np.ndarray:
        denom = max(1, int(self.cfg.horizon) - 1)
        phase = np.float32(min(self.t, int(self.cfg.horizon) - 1) / denom)
        return np.asarray(
            [self.position[0], self.position[1], self.gripper, phase],
            dtype=np.float32,
        )

    def expert_action(self, rng: np.random.Generator | None = None) -> np.ndarray:
        delta = float(self.cfg.expert_gain) * (self.goal - self.position)
        delta = np.clip(
            delta,
            -float(self.cfg.max_translation),
            float(self.cfg.max_translation),
        )
        if rng is not None and float(self.cfg.expert_action_noise) > 0.0:
            delta = delta + rng.normal(
                scale=float(self.cfg.expert_action_noise), size=delta.shape
            ).astype(np.float32)
            delta = np.clip(
                delta,
                -float(self.cfg.max_translation),
                float(self.cfg.max_translation),
            )
        next_distance = float(np.linalg.norm(self.goal - (self.position + delta)))
        grip = np.float32(next_distance <= float(self.cfg.close_radius))
        normalized_delta = delta / float(self.cfg.max_translation)
        return np.asarray(
            [normalized_delta[0], normalized_delta[1], grip], dtype=np.float32
        )

    def step(self, normalized_action: np.ndarray) -> tuple[np.ndarray, bool]:
        action = np.asarray(normalized_action, dtype=np.float32)
        if action.shape != (int(self.cfg.action_dim),):
            raise ValueError(
                f'Expected action shape {(self.cfg.action_dim,)}, got {action.shape}.'
            )
        delta = np.clip(action[:2], -1.0, 1.0) * float(self.cfg.max_translation)
        self.position = np.clip(
            self.position + delta,
            -float(self.cfg.world_limit),
            float(self.cfg.world_limit),
        ).astype(np.float32)
        self.gripper = np.float32(float(action[2]) >= 0.5)
        self.t += 1
        done = self.t >= int(self.cfg.horizon)
        return self.observation(), done

    def success(self) -> bool:
        return bool(
            np.linalg.norm(self.position - self.goal) <= float(self.cfg.success_radius)
            and self.gripper >= 0.5
        )

    def final_distance(self) -> float:
        return float(np.linalg.norm(self.position - self.goal))


def generate_expert_episode(
    cfg: HiddenGoalConfig,
    goal: np.ndarray,
    episode_id: int,
) -> Dict[str, np.ndarray]:
    env = HiddenGoalEnvironment(cfg, goal, episode_id)
    noise_rng = np.random.default_rng(int(episode_id) ^ 0x5EED123)
    observations = []
    actions = []
    next_observations = []
    initial_state = env.observation()
    for _ in range(int(cfg.horizon)):
        observation = env.observation()
        action = env.expert_action(noise_rng)
        next_observation, _ = env.step(action)
        observations.append(observation)
        actions.append(action)
        next_observations.append(next_observation)
    return {
        'observation': np.stack(observations).astype(np.float32),
        'action': np.stack(actions).astype(np.float32),
        'next_observation': np.stack(next_observations).astype(np.float32),
        'valid': np.ones((int(cfg.horizon),), dtype=np.bool_),
        'initial_state': initial_state.astype(np.float32),
        'success': np.asarray(env.success(), dtype=np.bool_),
        'final_distance': np.asarray(env.final_distance(), dtype=np.float32),
        'episode_id': np.asarray(int(episode_id), dtype=np.int32),
    }


def fit_state_normalizer(
    cfg: HiddenGoalConfig,
    task_bank: HiddenGoalTaskBank,
    *,
    episodes_per_task: int = 4,
    seed: int | None = None,
) -> StateNormalizer:
    rng = np.random.default_rng(int(cfg.seed) + 17 if seed is None else int(seed))
    observations = []
    for local_task_id in range(task_bank.num_tasks('train')):
        goal = task_bank.goal('train', local_task_id)
        for _ in range(int(episodes_per_task)):
            episode_id = int(rng.integers(1, np.iinfo(np.int32).max))
            episode = generate_expert_episode(cfg, goal, episode_id)
            observations.append(episode['observation'])
    all_observations = np.concatenate(observations, axis=0).astype(np.float32)
    mean = np.mean(all_observations, axis=0).astype(np.float32)
    std = np.maximum(np.std(all_observations, axis=0), 1e-4).astype(np.float32)
    payload = {
        'benchmark': asdict(cfg),
        'episodes_per_task': int(episodes_per_task),
        'mean': mean.tolist(),
        'std': std.tolist(),
    }
    identifier = 'hidden_goal_train_' + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode('utf-8')
    ).hexdigest()[:12]
    return StateNormalizer(mean, std, identifier)


class HiddenGoalMetaSampler:
    def __init__(
        self,
        cfg: HiddenGoalConfig,
        task_bank: HiddenGoalTaskBank,
        normalizer: StateNormalizer,
        *,
        split: str,
        seed: int,
    ):
        HiddenGoalTaskBank._validate_split(split)
        self.cfg = cfg
        self.task_bank = task_bank
        self.normalizer = normalizer
        self.split = split
        self.rng = np.random.default_rng(int(seed))

    def _episode_ids(self, count: int) -> np.ndarray:
        return self.rng.choice(
            np.iinfo(np.int32).max - 1,
            size=int(count),
            replace=False,
        ).astype(np.int32) + 1

    def _episodes(self, goal: np.ndarray, episode_ids: Sequence[int]) -> Dict[str, np.ndarray]:
        episodes = [
            generate_expert_episode(self.cfg, goal, int(episode_id))
            for episode_id in episode_ids
        ]
        return {
            'observation': np.stack(
                [self.normalizer.normalize_observation(ep['observation']) for ep in episodes]
            ).astype(np.float32),
            'action': np.stack([ep['action'] for ep in episodes]).astype(np.float32),
            'next_observation': np.stack(
                [self.normalizer.normalize_observation(ep['next_observation']) for ep in episodes]
            ).astype(np.float32),
            'valid': np.stack([ep['valid'] for ep in episodes]).astype(np.bool_),
            'episode_id': np.asarray(episode_ids, dtype=np.int32),
            'initial_state': np.stack([ep['initial_state'] for ep in episodes]).astype(np.float32),
            'expert_success': np.stack([ep['success'] for ep in episodes]).astype(np.bool_),
            'expert_final_distance': np.stack(
                [ep['final_distance'] for ep in episodes]
            ).astype(np.float32),
        }

    def build_batch(
        self,
        batch_size: int,
        *,
        support_episodes: int | None = None,
        query_episodes: int | None = None,
        task_ids: Sequence[int] | None = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        batch_size = int(batch_size)
        support_count = (
            int(self.cfg.support_episodes)
            if support_episodes is None
            else int(support_episodes)
        )
        query_count = (
            int(self.cfg.query_episodes)
            if query_episodes is None
            else int(query_episodes)
        )
        if support_count < 1 or query_count < 1:
            raise ValueError('support_episodes and query_episodes must both be positive.')
        if task_ids is None:
            local_task_ids = self.rng.integers(
                0, self.task_bank.num_tasks(self.split), size=(batch_size,)
            ).astype(np.int32)
        else:
            local_task_ids = np.asarray(task_ids, dtype=np.int32)
            if local_task_ids.shape != (batch_size,):
                raise ValueError(
                    f'task_ids must have shape {(batch_size,)}, got {local_task_ids.shape}.'
                )

        support_tasks = []
        query_tasks = []
        latents = []
        global_task_ids = []
        for local_task_id in local_task_ids:
            goal = self.task_bank.goal(self.split, int(local_task_id))
            episode_ids = self._episode_ids(support_count + query_count)
            support = self._episodes(goal, episode_ids[:support_count])
            query = self._episodes(goal, episode_ids[support_count:])
            support['write_mask'] = np.array(support['valid'], copy=True)
            support['outer_loss_mask'] = np.zeros_like(support['valid'], dtype=np.bool_)
            query['outer_loss_mask'] = np.array(query['valid'], copy=True)
            support_tasks.append(support)
            query_tasks.append(query)
            latents.append(goal)
            global_task_ids.append(
                self.task_bank.global_task_id(self.split, int(local_task_id))
            )

        def stack_tasks(tasks: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
            keys = tasks[0].keys()
            return {key: np.stack([task[key] for task in tasks]) for key in keys}

        return {
            'support': stack_tasks(support_tasks),
            'query': stack_tasks(query_tasks),
            'meta': {
                'local_task_id': local_task_ids.astype(np.int32),
                'global_task_id': np.asarray(global_task_ids, dtype=np.int32),
                'task_latent': np.stack(latents).astype(np.float32),
                'split_id': np.full(
                    (batch_size,), SPLITS.index(self.split), dtype=np.int32
                ),
            },
        }


def benchmark_integrity_report(
    cfg: HiddenGoalConfig,
    task_bank: HiddenGoalTaskBank,
    normalizer: StateNormalizer,
    *,
    samples: int = 512,
    seed: int = 123,
) -> Dict[str, object]:
    rng = np.random.default_rng(int(seed))
    all_goals = {split: task_bank.goals(split) for split in SPLITS}
    cross_split_distances = []
    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1 :]:
            distances = np.linalg.norm(
                all_goals[split_a][:, None, :] - all_goals[split_b][None, :, :],
                axis=-1,
            )
            cross_split_distances.append(float(np.min(distances)))

    initial_observations = []
    task_latents = []
    endpoints = []
    expert_successes = []
    for _ in range(int(samples)):
        local_task_id = int(rng.integers(0, task_bank.num_tasks('train')))
        goal = task_bank.goal('train', local_task_id)
        episode_id = int(rng.integers(1, np.iinfo(np.int32).max))
        episode = generate_expert_episode(cfg, goal, episode_id)
        initial_observations.append(episode['observation'][0])
        task_latents.append(goal)
        endpoints.append(episode['next_observation'][-1, :2])
        expert_successes.append(bool(episode['success']))

    x = np.asarray(initial_observations, dtype=np.float64)
    y = np.asarray(task_latents, dtype=np.float64)
    x_design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=-1)
    coefficients = np.linalg.lstsq(x_design, y, rcond=None)[0]
    prediction = x_design @ coefficients
    residual = np.sum((prediction - y) ** 2)
    total = np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2)
    query_only_linear_r2 = 1.0 - residual / max(total, 1e-12)
    endpoint_mse = float(
        np.mean((np.asarray(endpoints) - np.asarray(task_latents)) ** 2)
    )

    sampler = HiddenGoalMetaSampler(
        cfg, task_bank, normalizer, split='train', seed=int(seed) + 1
    )
    batch = sampler.build_batch(min(32, task_bank.num_tasks('train')))
    support_ids = batch['support']['episode_id']
    query_ids = batch['query']['episode_id']
    episode_overlap = any(
        np.intersect1d(support_ids[i], query_ids[i]).size > 0
        for i in range(support_ids.shape[0])
    )
    initial_state_equal = np.any(
        np.all(
            batch['support']['initial_state'][:, :, None, :]
            == batch['query']['initial_state'][:, None, :, :],
            axis=-1,
        )
    )
    return {
        'normalizer_id': normalizer.identifier,
        'split_sizes': {split: task_bank.num_tasks(split) for split in SPLITS},
        'minimum_cross_split_goal_distance': min(cross_split_distances),
        'query_only_initial_state_linear_r2': float(query_only_linear_r2),
        'support_endpoint_goal_mse': endpoint_mse,
        'oracle_expert_success_rate': float(np.mean(expert_successes)),
        'support_query_episode_overlap': bool(episode_overlap),
        'support_query_identical_initial_state': bool(initial_state_equal),
        'support_outer_loss_mask_nonzero': bool(
            np.any(batch['support']['outer_loss_mask'])
        ),
    }
