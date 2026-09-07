from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import imageio.v3 as imageio
import matplotlib
import numpy as np

from icil_jax_rlbench.visualization.state_rollouts import StateTaskVisualization

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402


_CONDITION_COLORS = {
    'no_update': '#4b5563',
    'correct_support': '#0f766e',
    'wrong_task_support': '#c2410c',
    'shuffled_actions': '#7c3aed',
    'shuffled_time': '#a21caf',
    'observations_only': '#b91c1c',
    'actions_only': '#0369a1',
    'duplicated_support': '#a16207',
    'random_update_matched_norm': '#be123c',
}


def _label(condition: str) -> str:
    return condition.replace('_', ' ')


def _condition_color(condition: str) -> str:
    return _CONDITION_COLORS.get(condition, '#111827')


def _workspace_axis(axis, data: StateTaskVisualization) -> None:
    margin = 0.05 * float(data.world_limit)
    limit = float(data.world_limit) + margin
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect('equal', adjustable='box')
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.add_patch(
        Circle(
            data.goal,
            float(data.success_radius),
            facecolor='#86efac',
            edgecolor='#15803d',
            alpha=0.25,
            linewidth=1.0,
        )
    )
    axis.scatter(
        data.goal[0],
        data.goal[1],
        marker='*',
        s=100,
        color='#15803d',
        edgecolor='white',
        linewidth=0.6,
        zorder=7,
        label='goal',
    )
    axis.set_xlabel('x')
    axis.set_ylabel('y')


def _subplot_grid(count: int, *, width: float = 4.0, height: float = 3.7):
    columns = min(3, max(1, int(count)))
    rows = int(math.ceil(int(count) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width * columns, height * rows),
        squeeze=False,
        constrained_layout=True,
    )
    return figure, axes.reshape(-1)


def plot_support_trajectories(
    data: StateTaskVisualization, output_path: Path, *, dpi: int
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    _workspace_axis(axis, data)
    for index, positions in enumerate(data.correct_support_positions):
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            color='#0f766e',
            linewidth=2.0,
            alpha=0.85,
            label='correct support' if index == 0 else None,
        )
        axis.scatter(positions[0, 0], positions[0, 1], color='#0f766e', s=20)
    for index, positions in enumerate(data.wrong_support_positions):
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            color='#c2410c',
            linewidth=1.7,
            linestyle='--',
            alpha=0.8,
            label='wrong-task support' if index == 0 else None,
        )
    axis.scatter(
        data.wrong_goal[0],
        data.wrong_goal[1],
        marker='X',
        s=70,
        color='#c2410c',
        label='wrong goal',
    )
    axis.set_title(
        f'Task {data.task_id}: support evidence for the hidden goal', fontsize=12
    )
    axis.legend(loc='upper right', fontsize=8)
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def plot_matched_trajectories(
    data: StateTaskVisualization, output_path: Path, *, dpi: int
) -> None:
    figure, axes = _subplot_grid(len(data.conditions))
    episode_colors = plt.get_cmap('tab10')
    for axis, condition in zip(axes, data.conditions):
        _workspace_axis(axis, data)
        support = (
            data.wrong_support_positions
            if condition == 'wrong_task_support'
            else data.correct_support_positions
        )
        for positions in support:
            axis.plot(
                positions[:, 0],
                positions[:, 1],
                color='#9ca3af',
                linewidth=1.0,
                linestyle='--',
                alpha=0.55,
            )
        traces = data.rollouts[condition]
        for episode_index, trace in enumerate(traces):
            color = episode_colors(episode_index % 10)
            positions = trace.positions
            axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=color,
                linewidth=2.0,
                alpha=0.9,
            )
            axis.scatter(
                positions[0, 0],
                positions[0, 1],
                marker='o',
                facecolor='white',
                edgecolor=color,
                s=26,
                zorder=6,
            )
            axis.scatter(
                positions[-1, 0],
                positions[-1, 1],
                marker='o' if trace.success else 'x',
                color=color,
                s=28,
                zorder=6,
            )
        successes = sum(int(trace.success) for trace in traces)
        axis.set_title(
            f'{_label(condition)}\n{successes}/{len(traces)} successful',
            color=_condition_color(condition),
            fontsize=10,
        )
    for axis in axes[len(data.conditions) :]:
        axis.set_visible(False)
    figure.suptitle(
        f'Matched query starts after fast-weight updates: task {data.task_id}',
        fontsize=13,
    )
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def plot_action_changes(
    data: StateTaskVisualization, output_path: Path, *, dpi: int
) -> None:
    baseline = np.asarray(data.reference_actions['no_update'])
    steps = np.arange(baseline.shape[1])
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.5, 6.5),
        sharex=True,
        constrained_layout=True,
    )
    for condition in data.conditions:
        if condition == 'no_update':
            continue
        actions = np.asarray(data.reference_actions[condition])
        translation_change = np.linalg.norm(
            actions[..., :2] - baseline[..., :2], axis=-1
        )
        gripper_change = np.abs(actions[..., 2] - baseline[..., 2])
        color = _condition_color(condition)
        axes[0].plot(
            steps,
            np.mean(translation_change, axis=0),
            color=color,
            label=_label(condition),
        )
        axes[1].plot(
            steps,
            np.mean(gripper_change, axis=0),
            color=color,
            label=_label(condition),
        )
    axes[0].set_ylabel('mean translation action change')
    axes[1].set_ylabel('mean gripper action change')
    axes[1].set_xlabel('query demonstration timestep')
    axes[0].set_title('Fast-weight effect at identical query observations')
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def plot_write_diagnostics(
    data: StateTaskVisualization, output_path: Path, *, dpi: int
) -> None:
    metric_names = (
        ('write_loss', 'KVB WRITE loss'),
        ('fast_update_norm', 'fast update norm'),
        ('fast_delta_norm', 'distance from W0'),
    )
    figure, axes = plt.subplots(
        1, len(metric_names), figsize=(13.5, 3.8), constrained_layout=True
    )
    for condition in data.conditions:
        trace = data.write_traces.get(condition, {})
        if not trace:
            continue
        for axis, (name, title) in zip(axes, metric_names):
            values = trace.get(name)
            if values is None:
                continue
            axis.plot(
                np.arange(1, len(values) + 1),
                values,
                marker='o',
                markersize=2.5,
                linewidth=1.3,
                color=_condition_color(condition),
                label=_label(condition),
            )
            axis.set_title(title)
            axis.set_xlabel('support segment')
            axis.grid(alpha=0.25)
    axes[0].set_ylabel('value')
    axes[-1].legend(fontsize=7, loc='best')
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def plot_fast_tensor_deltas(
    data: StateTaskVisualization, output_path: Path, *, dpi: int
) -> None:
    tensor_names = sorted(
        {
            name
            for condition in data.conditions
            for name in data.fast_tensor_delta_norms[condition]
        }
    )
    x = np.arange(len(tensor_names), dtype=np.float32)
    width = 0.8 / max(1, len(data.conditions))
    figure, axis = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    for condition_index, condition in enumerate(data.conditions):
        values = [
            data.fast_tensor_delta_norms[condition][name] for name in tensor_names
        ]
        offset = (condition_index - (len(data.conditions) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            color=_condition_color(condition),
            label=_label(condition),
        )
    axis.set_xticks(x, tensor_names, rotation=25, ha='right')
    axis.set_ylabel('L2 distance from W0')
    axis.set_title('Fast-state parameter changes')
    axis.grid(axis='y', alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def plot_vector_fields(
    data: StateTaskVisualization, output_path: Path, *, dpi: int
) -> None:
    figure, axes = _subplot_grid(len(data.conditions))
    for axis, condition in zip(axes, data.conditions):
        _workspace_axis(axis, data)
        field_x, field_y = data.vector_fields[condition]
        magnitude = np.sqrt(np.square(field_x) + np.square(field_y))
        axis.quiver(
            data.grid_x,
            data.grid_y,
            field_x,
            field_y,
            magnitude,
            cmap='viridis',
            angles='xy',
            scale_units='xy',
            scale=5.0,
            width=0.004,
            alpha=0.9,
        )
        axis.set_title(
            _label(condition),
            color=_condition_color(condition),
            fontsize=10,
        )
    for axis in axes[len(data.conditions) :]:
        axis.set_visible(False)
    figure.suptitle(
        f'Planar action field after adaptation: task {data.task_id}', fontsize=13
    )
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


def write_trajectory_video(
    data: StateTaskVisualization, output_path: Path, *, fps: float
) -> None:
    figure, axes = _subplot_grid(len(data.conditions), width=4.0, height=4.0)
    lines: dict[str, list[Any]] = {}
    episode_colors = plt.get_cmap('tab10')
    for axis, condition in zip(axes, data.conditions):
        _workspace_axis(axis, data)
        axis.set_title(
            _label(condition),
            color=_condition_color(condition),
            fontsize=10,
        )
        condition_lines = []
        for episode_index, _ in enumerate(data.rollouts[condition]):
            line, = axis.plot(
                [],
                [],
                color=episode_colors(episode_index % 10),
                linewidth=2.2,
            )
            condition_lines.append(line)
        lines[condition] = condition_lines
    for axis in axes[len(data.conditions) :]:
        axis.set_visible(False)
    figure.suptitle(f'Matched query trajectories: task {data.task_id}', fontsize=13)

    maximum_steps = max(
        len(trace.positions)
        for condition in data.conditions
        for trace in data.rollouts[condition]
    )
    with imageio.imopen(output_path, 'w', plugin='pyav') as writer:
        writer.init_video_stream('libx264', fps=float(fps))
        for frame_index in range(maximum_steps):
            for condition in data.conditions:
                for line, trace in zip(lines[condition], data.rollouts[condition]):
                    positions = trace.positions[: frame_index + 1]
                    line.set_data(positions[:, 0], positions[:, 1])
            figure.canvas.draw()
            frame = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            writer.write_frame(np.array(frame, copy=True))
    plt.close(figure)


def _write_task_arrays(data: StateTaskVisualization, path: Path) -> None:
    arrays: dict[str, np.ndarray] = {
        'goal': np.asarray(data.goal, dtype=np.float32),
        'wrong_goal': np.asarray(data.wrong_goal, dtype=np.float32),
        'grid_x': np.asarray(data.grid_x, dtype=np.float32),
        'grid_y': np.asarray(data.grid_y, dtype=np.float32),
    }
    for index, positions in enumerate(data.correct_support_positions):
        arrays[f'correct_support_{index}_positions'] = positions
    for index, positions in enumerate(data.wrong_support_positions):
        arrays[f'wrong_support_{index}_positions'] = positions
    for condition in data.conditions:
        arrays[f'{condition}_reference_actions'] = data.reference_actions[condition]
        arrays[f'{condition}_episode_ids'] = np.asarray(
            [trace.episode_id for trace in data.rollouts[condition]],
            dtype=np.int32,
        )
        arrays[f'{condition}_success'] = np.asarray(
            [trace.success for trace in data.rollouts[condition]],
            dtype=np.bool_,
        )
        arrays[f'{condition}_final_distance'] = np.asarray(
            [trace.final_distance for trace in data.rollouts[condition]],
            dtype=np.float32,
        )
        field_x, field_y = data.vector_fields[condition]
        arrays[f'{condition}_field_x'] = field_x
        arrays[f'{condition}_field_y'] = field_y
        for index, trace in enumerate(data.rollouts[condition]):
            arrays[f'{condition}_{index}_observations'] = trace.observations
            arrays[f'{condition}_{index}_actions'] = trace.actions
    np.savez_compressed(path, **arrays)


def render_task_artifacts(
    data: StateTaskVisualization,
    output_dir: Path,
    *,
    dpi: int,
    write_video: bool,
    video_fps: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {
        'support_trajectories': 'support_trajectories.png',
        'matched_trajectories': 'matched_trajectories.png',
        'action_changes': 'action_changes.png',
        'write_diagnostics': 'write_diagnostics.png',
        'fast_tensor_deltas': 'fast_tensor_deltas.png',
        'vector_fields': 'vector_fields.png',
        'trajectory_data': 'trajectory_data.npz',
    }
    plot_support_trajectories(
        data, output_dir / str(artifacts['support_trajectories']), dpi=dpi
    )
    plot_matched_trajectories(
        data, output_dir / str(artifacts['matched_trajectories']), dpi=dpi
    )
    plot_action_changes(data, output_dir / str(artifacts['action_changes']), dpi=dpi)
    plot_write_diagnostics(
        data, output_dir / str(artifacts['write_diagnostics']), dpi=dpi
    )
    plot_fast_tensor_deltas(
        data, output_dir / str(artifacts['fast_tensor_deltas']), dpi=dpi
    )
    plot_vector_fields(data, output_dir / str(artifacts['vector_fields']), dpi=dpi)
    _write_task_arrays(data, output_dir / str(artifacts['trajectory_data']))
    if write_video:
        artifacts['trajectory_video'] = 'matched_trajectories.mp4'
        write_trajectory_video(
            data,
            output_dir / str(artifacts['trajectory_video']),
            fps=float(video_fps),
        )

    task_summary = {
        'task_id': int(data.task_id),
        'wrong_task_id': int(data.wrong_task_id),
        'goal': np.asarray(data.goal).tolist(),
        'wrong_goal': np.asarray(data.wrong_goal).tolist(),
        'conditions': {
            condition: {
                'query_episode_ids': [
                    int(trace.episode_id) for trace in data.rollouts[condition]
                ],
                'successful_rollouts': sum(
                    int(trace.success) for trace in data.rollouts[condition]
                ),
                'rollout_count': len(data.rollouts[condition]),
                'mean_final_distance': float(
                    np.mean(
                        [trace.final_distance for trace in data.rollouts[condition]]
                    )
                ),
                'fast_tensor_delta_norms': dict(
                    data.fast_tensor_delta_norms[condition]
                ),
            }
            for condition in data.conditions
        },
        'artifacts': artifacts,
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(task_summary, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    return task_summary


def plot_evaluation_summary(
    summary: Mapping[str, Any], output_path: Path, *, dpi: int
) -> None:
    aggregate = summary['aggregate']
    support_counts = sorted(aggregate, key=int)
    conditions = [
        condition
        for condition in _CONDITION_COLORS
        if all(condition in aggregate[count] for count in support_counts)
    ]
    x = np.asarray([int(count) for count in support_counts])
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.5, 8.2),
        constrained_layout=True,
    )
    for condition in conditions:
        values = [aggregate[count][condition] for count in support_counts]
        color = _condition_color(condition)
        success = np.asarray([value['success_rate'] for value in values])
        low = np.asarray([value['success_ci95'][0] for value in values])
        high = np.asarray([value['success_ci95'][1] for value in values])
        axes[0, 0].plot(
            x,
            success,
            marker='o',
            color=color,
            label=_label(condition),
        )
        axes[0, 0].fill_between(x, low, high, color=color, alpha=0.10)
        axes[0, 1].plot(
            x,
            [value['mean_offline_loss'] for value in values],
            marker='o',
            color=color,
        )
        axes[0, 2].plot(
            x,
            [value['mean_fast_delta_norm'] for value in values],
            marker='o',
            color=color,
        )

    paired_metrics = (
        ('success_rate', 'paired success-rate gain'),
        ('offline_loss_reduction', 'paired offline-loss reduction'),
        ('final_distance_reduction', 'paired final-distance reduction'),
    )
    paired_available = all(
        'paired_gain_over_no_update' in aggregate[count]
        for count in support_counts
    )
    for metric_index, (metric_name, ylabel) in enumerate(paired_metrics):
        axis = axes[1, metric_index]
        axis.axhline(0.0, color='#111827', linewidth=0.8, alpha=0.65)
        axis.set_ylabel(ylabel)
        if not paired_available:
            axis.text(
                0.5,
                0.5,
                'Paired gains unavailable in this summary',
                ha='center',
                va='center',
                transform=axis.transAxes,
                fontsize=8,
            )
            continue
        for condition in conditions:
            if condition == 'no_update':
                continue
            paired_values = [
                aggregate[count]['paired_gain_over_no_update'][condition][
                    metric_name
                ]
                for count in support_counts
            ]
            means = np.asarray([value['mean'] for value in paired_values])
            low = np.asarray([value['ci95'][0] for value in paired_values])
            high = np.asarray([value['ci95'][1] for value in paired_values])
            color = _condition_color(condition)
            axis.plot(
                x,
                means,
                marker='o',
                linewidth=1.2,
                color=color,
            )
            axis.vlines(
                x,
                low,
                high,
                linewidth=1.0,
                color=color,
            )

    axes[0, 0].set_ylabel('closed-loop success rate')
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 1].set_ylabel('offline query loss')
    axes[0, 2].set_ylabel('fast-state delta norm')
    for axis in axes.reshape(-1):
        axis.set_xlabel('support demonstrations')
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc='outside lower center',
        fontsize=7,
        ncol=min(5, len(labels)),
    )
    figure.suptitle('Gate 3 held-out evaluation summary', fontsize=13)
    figure.savefig(output_path, dpi=int(dpi))
    plt.close(figure)


__all__ = ['plot_evaluation_summary', 'render_task_artifacts']
