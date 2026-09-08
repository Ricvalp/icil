"""Public drawing frames and a small executed-motion pen benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


IDENTITY_FRAME = np.asarray([0.0, 0.0, 0.0, 1.0], np.float32)


def apply_frame(absolute: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Apply public [tx, ty, angle_radians, positive_isotropic_scale]."""
    frame = np.asarray(frame, dtype=np.float64)
    if frame.shape != (4,) or not np.all(np.isfinite(frame)) or frame[3] <= 0:
        raise ValueError('A frame must be finite [tx, ty, angle, positive_scale].')
    c, s = np.cos(frame[2]), np.sin(frame[2])
    rotation = np.asarray([[c, -s], [s, c]])
    return (np.asarray(absolute) @ rotation.T * frame[3] + frame[:2]).astype(np.float32)


def invert_frame(absolute: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Recover canonical coordinates without recentering a transformed drawing."""
    frame = np.asarray(frame, dtype=np.float64)
    if frame.shape != (4,) or not np.all(np.isfinite(frame)) or frame[3] <= 0:
        raise ValueError('A frame must be finite [tx, ty, angle, positive_scale].')
    c, s = np.cos(frame[2]), np.sin(frame[2])
    rotation = np.asarray([[c, -s], [s, c]])
    return ((np.asarray(absolute) - frame[:2]) @ rotation / frame[3]).astype(np.float32)


def transformed_replay(
    support_absolute: np.ndarray, support_frame: np.ndarray, query_frame: np.ndarray
) -> np.ndarray:
    return apply_frame(invert_frame(support_absolute, support_frame), query_frame)


@dataclass(frozen=True)
class PenConfig:
    """Commands are bounded XY displacements, incoming pen state, and STOP.

    Recovery follows a timed reference. Delay/noise profiles are separate from
    the default deterministic integrator. Positions are never canvas-clipped.
    """

    max_motion: float = 0.1
    motion_gain: float = 1.0
    noise_std: float = 0.0
    delay_steps: int = 0
    canvas_limit: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_motion) or self.max_motion <= 0:
            raise ValueError('max_motion must be finite and positive.')
        if not np.isfinite(self.motion_gain) or self.motion_gain <= 0:
            raise ValueError('motion_gain must be finite and positive.')
        if not np.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError('noise_std must be finite and nonnegative.')
        if self.delay_steps < 0 or int(self.delay_steps) != self.delay_steps:
            raise ValueError('delay_steps must be a nonnegative integer.')
        if not np.isfinite(self.canvas_limit) or self.canvas_limit <= 0:
            raise ValueError('canvas_limit must be finite and positive.')


def bounded_motion(delta: np.ndarray, max_motion: float) -> np.ndarray:
    delta = np.asarray(delta, np.float32)
    norm = float(np.linalg.norm(delta))
    return delta * min(1.0, max_motion / max(norm, 1e-12))


class PenEnvironment:
    """Simulator-free state contains actual [x, y, pen_down]."""

    def __init__(self, config: PenConfig = PenConfig(), *, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.reset(np.zeros(2, np.float32))

    def reset(self, start_xy: np.ndarray, *, pen_down: float = 0.0) -> np.ndarray:
        xy = np.asarray(start_xy, np.float32)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError('start_xy must contain two finite coordinates.')
        self.state = np.asarray([*xy, float(pen_down >= 0.5)], np.float32)
        self.terminated = False
        self.steps = 0
        self.pending = [np.zeros(3, np.float32) for _ in range(self.config.delay_steps)]
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, dict]:
        """Apply one bounded command; STOP ends without drawing a final segment."""
        action = np.asarray(action, np.float32)
        if action.shape != (4,):
            raise ValueError('A pen action has shape (4,): dx, dy, pen, STOP.')
        if self.terminated:
            raise RuntimeError('Reset the pen environment after STOP.')
        invalid = not bool(np.all(np.isfinite(action)))
        if invalid:
            action = np.zeros(4, np.float32)
        clipped = float(np.linalg.norm(action[:2])) > self.config.max_motion + 1e-7
        info = {'invalid_action': invalid, 'clipped_action': clipped, 'pen_up_travel': 0.0}
        self.steps += 1
        if action[3] >= 0.5:
            self.terminated = True
        else:
            command = np.asarray(
                [*bounded_motion(action[:2], self.config.max_motion), float(action[2] >= 0.5)],
                np.float32,
            )
            self.pending.append(command)
            executed = self.pending.pop(0)
            movement = executed[:2] * self.config.motion_gain
            movement += self.rng.normal(0.0, self.config.noise_std, 2).astype(np.float32)
            self.state[:2] += movement
            self.state[2] = executed[2]
            if self.state[2] < 0.5:
                info['pen_up_travel'] = float(np.linalg.norm(movement))
        info.update(
            terminated=self.terminated,
            out_of_bounds=bool(np.any(np.abs(self.state[:2]) > self.config.canvas_limit)),
        )
        return self.state.copy(), info


def timed_reference(
    absolute: np.ndarray,
    incoming_pen: np.ndarray,
    start_xy: np.ndarray,
    *,
    max_motion: float = 0.1,
    return_knots: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subdivide ordered segments and explicitly travel pen-up to the first point.

    This is a label/reference construction API, never a neural policy input.
    Each destination retains the original incoming-segment pen convention.
    Complete demonstrations may expose program-knot markers: original program
    vertices are distinguished from travel/interpolation destinations. They
    preserve collinear vertices and singletons for exact frame-aware replay.
    Future query knot markers are labels and must not enter query prediction.
    """
    absolute = np.asarray(absolute, np.float32)
    incoming_pen = np.asarray(incoming_pen, np.float32)
    if absolute.ndim != 2 or absolute.shape[1] != 2 or len(absolute) != len(incoming_pen):
        raise ValueError('Expected XY points and one incoming pen flag per point.')
    if not np.all(np.isfinite(absolute)) or max_motion <= 0:
        raise ValueError('Reference coordinates must be finite and max_motion positive.')
    position = np.asarray(start_xy, np.float32)
    positions, pens, knots = [], [], []
    for index, destination in enumerate(absolute):
        # The slightly conservative divisor avoids float32 bound overshoots.
        count = max(1, int(np.ceil(np.linalg.norm(destination - position) / (max_motion * 0.999))))
        segment = np.linspace(position, destination, count + 1, dtype=np.float32)[1:]
        positions.extend(segment)
        pens.extend([0.0 if index == 0 else float(incoming_pen[index])] * count)
        knots.extend([False] * (count - 1) + [True])
        position = destination
    result = (np.asarray(positions, np.float32).reshape(-1, 2), np.asarray(pens, np.float32))
    return (*result, np.asarray(knots, bool)) if return_knots else result


def rollout_policy(
    policy: Callable[[np.ndarray, int, np.ndarray, np.ndarray], np.ndarray],
    *,
    start_xy: np.ndarray,
    max_steps: int,
    config: PenConfig = PenConfig(),
    seed: int = 0,
    perturbations: dict[int, np.ndarray] | None = None,
) -> dict:
    """Call a policy with actual state and executed prefix, never an expert suffix.

    Prefix arrays contain pre-command states and issued actions; the new state
    is supplied separately. Empty prefixes are valid at step zero.
    """
    env = PenEnvironment(config, seed=seed)
    state = env.reset(start_xy)
    states, actions, positions, pens, info_records = [], [], [], [], []
    for step in range(max_steps):
        if perturbations and step in perturbations:
            env.state[:2] += np.asarray(perturbations[step], np.float32)
            state = env.state.copy()
        action = np.asarray(policy(
            state.copy(), step,
            np.asarray(states, np.float32).reshape(-1, 3),
            np.asarray(actions, np.float32).reshape(-1, 4),
        ), np.float32)
        states.append(state.copy())
        actions.append(action.copy())
        state, info = env.step(action)
        info_records.append(info)
        if info['terminated']:
            break
        positions.append(state[:2].copy())
        pens.append(state[2])
    return {
        'state': np.asarray(states, np.float32).reshape(-1, 3),
        'tokens': np.asarray(actions, np.float32).reshape(-1, 4),
        'absolute': np.asarray(positions, np.float32).reshape(-1, 2),
        'incoming_pen': np.asarray(pens, np.float32),
        'terminated': env.terminated,
        'final_state': env.state.copy(),
        'invalid_actions': sum(int(x['invalid_action']) for x in info_records),
        'clipped_actions': sum(int(x['clipped_action']) for x in info_records),
        'out_of_bounds_steps': sum(int(x['out_of_bounds']) for x in info_records),
        'pen_up_travel': sum(x['pen_up_travel'] for x in info_records),
        'config': asdict(config),
        'recovery_semantics': 'timed_reference',
    }


def replay_rollout(
    reference_xy: np.ndarray,
    reference_pen: np.ndarray,
    *,
    start_xy: np.ndarray,
    feedback: bool = True,
    config: PenConfig = PenConfig(),
    seed: int = 0,
    max_steps: int | None = None,
    perturbations: dict[int, np.ndarray] | None = None,
) -> dict:
    """Open-loop replay or feedback path tracking against the same timed target."""
    reference_xy = np.asarray(reference_xy, np.float32)
    reference_pen = np.asarray(reference_pen, np.float32)
    if len(reference_xy) != len(reference_pen):
        raise ValueError('Reference point/pen lengths must agree.')
    previous = np.concatenate([np.asarray(start_xy, np.float32)[None], reference_xy[:-1]], axis=0)
    commands = reference_xy - previous

    def policy(state, step, _states, _actions):
        if step >= len(reference_xy):
            return np.asarray([0.0, 0.0, 0.0, 1.0], np.float32)
        delta = reference_xy[step] - state[:2] if feedback else commands[step]
        return np.asarray([*bounded_motion(delta, config.max_motion), reference_pen[step], 0.0], np.float32)

    return rollout_policy(
        policy, start_xy=start_xy, max_steps=max_steps or len(reference_xy) + 1,
        config=config, seed=seed, perturbations=perturbations,
    )
