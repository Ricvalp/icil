from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_UPDATE_REPRESENTATIONS = (
    'first_write_gradient',
    'final_fast_delta',
    'read_action_delta',
)

_SPLIT_SYMBOLS = {
    'train': 'circle',
    'latent_validation': 'square',
    'family_validation': 'diamond',
}


def _write_json(path: Path, value: Any) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _row_array(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    return np.asarray([row[name] for row in rows])


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _mean_cosine_groups(
    values: np.ndarray,
    task_ids: np.ndarray,
    families: np.ndarray,
) -> dict[str, float | int | None]:
    normalized = _l2_normalize(values)
    similarities = normalized @ normalized.T
    upper = np.triu(np.ones(similarities.shape, dtype=np.bool_), k=1)
    masks = {
        'same_instance': upper & (task_ids[:, None] == task_ids[None, :]),
        'same_family_different_instance': (
            upper
            & (task_ids[:, None] != task_ids[None, :])
            & (families[:, None] == families[None, :])
        ),
        'different_family': upper & (families[:, None] != families[None, :]),
    }
    result: dict[str, float | int | None] = {}
    for name, mask in masks.items():
        selected = similarities[mask]
        result[f'{name}_pairs'] = int(selected.size)
        result[f'{name}_mean'] = (
            float(np.mean(selected)) if selected.size else None
        )
    return result


def _nearest_neighbor_accuracy(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    candidate_mask: np.ndarray | None = None,
) -> float | None:
    normalized = _l2_normalize(values)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    if candidate_mask is not None:
        similarities = np.where(candidate_mask, similarities, -np.inf)
    valid = np.any(np.isfinite(similarities), axis=1)
    if not np.any(valid):
        return None
    nearest = np.argmax(similarities[valid], axis=1)
    return float(np.mean(labels[nearest] == labels[valid]))


def _reference_family_accuracy(
    reference: np.ndarray,
    reference_families: np.ndarray,
    query: np.ndarray,
    query_families: np.ndarray,
) -> float | None:
    if not len(reference) or not len(query):
        return None
    similarities = _l2_normalize(query) @ _l2_normalize(reference).T
    predicted = reference_families[np.argmax(similarities, axis=1)]
    return float(np.mean(predicted == query_families))


def _heldout_nearest_familiar_families(
    values: np.ndarray,
    splits: np.ndarray,
    families: np.ndarray,
) -> dict[str, list[dict[str, float | int | str]]]:
    train = splits == 'train'
    heldout = splits == 'family_validation'
    if not np.any(train) or not np.any(heldout):
        return {}
    similarities = _l2_normalize(values[heldout]) @ _l2_normalize(values[train]).T
    nearest = families[train][np.argmax(similarities, axis=1)]
    heldout_families = families[heldout]
    result: dict[str, list[dict[str, float | int | str]]] = {}
    for family in sorted(set(heldout_families.tolist())):
        selected = nearest[heldout_families == family]
        names, counts = np.unique(selected, return_counts=True)
        order = np.argsort(-counts)
        result[str(family)] = [
            {
                'familiar_family': str(names[index]),
                'count': int(counts[index]),
                'fraction': float(counts[index] / selected.size),
            }
            for index in order[:5]
        ]
    return result


def update_clustering_summary(
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, np.ndarray],
    *,
    condition: str,
    support_count: int,
    representations: Sequence[str] = DEFAULT_UPDATE_REPRESENTATIONS,
    splits: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Measure family and instance neighborhoods in original feature space."""

    row_condition = _row_array(rows, 'condition')
    row_count = _row_array(rows, 'support_count').astype(np.int32)
    selected = (row_condition == str(condition)) & (row_count == int(support_count))
    if splits is not None:
        selected &= np.isin(
            _row_array(rows, 'target_split'),
            tuple(str(value) for value in splits),
        )
    if not np.any(selected):
        raise ValueError('No rows match the requested visualization condition.')

    splits = _row_array(rows, 'target_split')[selected]
    task_ids = _row_array(rows, 'target_task_id')[selected]
    families = _row_array(rows, 'target_family')[selected]
    result: dict[str, Any] = {}
    for representation in representations:
        if representation not in features:
            raise ValueError(f'Unknown update representation {representation!r}.')
        values = np.asarray(features[representation])[selected]
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError(
                f'{representation} must be a finite rank-two feature matrix.'
            )
        representation_result: dict[str, Any] = {
            'cosine_geometry': _mean_cosine_groups(values, task_ids, families),
            'heldout_nearest_familiar_families': (
                _heldout_nearest_familiar_families(values, splits, families)
            ),
        }

        train = splits == 'train'
        latent = splits == 'latent_validation'
        heldout = splits == 'family_validation'
        representation_result['latent_to_train_1nn_family_accuracy'] = (
            _reference_family_accuracy(
                values[train], families[train], values[latent], families[latent]
            )
        )

        split_metrics = {}
        for split_name, mask in (
            ('train', train),
            ('latent_validation', latent),
            ('family_validation', heldout),
        ):
            if np.count_nonzero(mask) < 2:
                continue
            split_values = values[mask]
            split_tasks = task_ids[mask]
            split_families = families[mask]
            other_instance = split_tasks[:, None] != split_tasks[None, :]
            split_metrics[split_name] = {
                'rows': int(np.count_nonzero(mask)),
                'families': len(set(split_families.tolist())),
                'cosine_geometry': _mean_cosine_groups(
                    split_values, split_tasks, split_families
                ),
                'same_instance_1nn_accuracy': _nearest_neighbor_accuracy(
                    split_values, split_tasks
                ),
                'family_1nn_excluding_same_instance_accuracy': (
                    _nearest_neighbor_accuracy(
                        split_values,
                        split_families,
                        candidate_mask=other_instance,
                    )
                ),
            }
        representation_result['by_split'] = split_metrics
        result[representation] = representation_result
    return result


def _color_map(names: Sequence[str]) -> dict[str, str]:
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    unique = tuple(sorted(set(names)))
    colors = colormaps['turbo'](np.linspace(0.03, 0.97, max(1, len(unique))))
    return {name: to_hex(color) for name, color in zip(unique, colors, strict=True)}


def _hover_text(row: Mapping[str, Any]) -> str:
    phases = ' > '.join(str(value) for value in row.get('target_motion_phases', ()))
    return '<br>'.join(
        (
            f"task: {row['target_task_id']}",
            f"family: {row['target_family']}",
            f"instance: {row.get('target_instance_index', -1)}",
            f"split: {row['target_split']}",
            f"sample: {row['sample_index']}",
            f"motion: {phases or 'unknown'}",
            f"query gain: {float(row['query_gain']):.6f}",
            f"fast delta norm: {float(row['final_fast_delta_norm']):.6f}",
        )
    )


def _base_figure(title: str):
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.update_layout(
        title=(
            f'{title}<br><sup>circle: train; square: familiar-family unseen '
            'instance; diamond: held-out family</sup>'
        ),
        template='plotly_white',
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        legend_title='group',
        hovermode='closest',
        width=1200,
        height=850,
    )
    figure.update_xaxes(zeroline=False)
    figure.update_yaxes(zeroline=False, scaleanchor='x', scaleratio=1.0)
    return figure


def _add_trace(
    figure,
    coordinates: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    *,
    name: str,
    color: str,
    opacity: float = 0.82,
    showlegend: bool = True,
) -> None:
    import plotly.graph_objects as go

    symbols = [
        _SPLIT_SYMBOLS.get(str(rows[index]['target_split']), 'circle')
        for index in indices
    ]
    figure.add_trace(
        go.Scattergl(
            x=coordinates[indices, 0],
            y=coordinates[indices, 1],
            mode='markers',
            name=name,
            text=[_hover_text(rows[index]) for index in indices],
            hovertemplate='%{text}<extra></extra>',
            marker={
                'color': color,
                'opacity': float(opacity),
                'size': 8,
                'symbol': symbols,
                'line': {'color': 'rgba(0,0,0,0.22)', 'width': 0.5},
            },
            showlegend=showlegend,
        )
    )


def _write_family_plot(
    coordinates: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str,
) -> None:
    families = np.asarray([row['target_family'] for row in rows])
    colors = _color_map(families.tolist())
    figure = _base_figure(title)
    for family in sorted(colors):
        indices = np.flatnonzero(families == family)
        _add_trace(
            figure,
            coordinates,
            rows,
            indices,
            name=family,
            color=colors[family],
        )
    figure.write_html(path, include_plotlyjs='directory', full_html=True)


def _write_heldout_plot(
    coordinates: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str,
) -> None:
    import plotly.graph_objects as go

    splits = np.asarray([row['target_split'] for row in rows])
    families = np.asarray([row['target_family'] for row in rows])
    task_ids = np.asarray([row['target_task_id'] for row in rows])
    heldout = splits == 'family_validation'
    heldout_families = sorted(set(families[heldout].tolist()))
    colors = _color_map(heldout_families)
    figure = _base_figure(title)

    context = np.flatnonzero(~heldout)
    if context.size:
        _add_trace(
            figure,
            coordinates,
            rows,
            context,
            name='familiar-family context',
            color='#a8adb4',
            opacity=0.16,
        )
    for family in heldout_families:
        family_indices = np.flatnonzero(heldout & (families == family))
        for task_id in sorted(set(task_ids[family_indices].tolist())):
            task_indices = np.flatnonzero(heldout & (task_ids == task_id))
            if task_indices.size > 1:
                figure.add_trace(
                    go.Scattergl(
                        x=coordinates[task_indices, 0],
                        y=coordinates[task_indices, 1],
                        mode='lines',
                        line={'color': colors[family], 'width': 1},
                        opacity=0.35,
                        hoverinfo='skip',
                        showlegend=False,
                    )
                )
        _add_trace(
            figure,
            coordinates,
            rows,
            family_indices,
            name=family,
            color=colors[family],
            opacity=0.95,
        )
    figure.write_html(path, include_plotlyjs='directory', full_html=True)


def _write_motion_plot(
    coordinates: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str,
) -> None:
    phases = np.asarray(
        [str(row.get('target_terminal_phase', 'unknown')) for row in rows]
    )
    colors = _color_map(phases.tolist())
    figure = _base_figure(title)
    for phase in sorted(colors):
        indices = np.flatnonzero(phases == phase)
        _add_trace(
            figure,
            coordinates,
            rows,
            indices,
            name=phase,
            color=colors[phase],
        )
    figure.write_html(path, include_plotlyjs='directory', full_html=True)


def _write_embedding_csv(
    path: Path,
    coordinates: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        'x',
        'y',
        'target_task_id',
        'target_family',
        'target_instance_index',
        'target_split',
        'sample_index',
        'support_count',
        'target_motion_signature',
        'target_terminal_phase',
        'query_gain',
        'final_fast_delta_norm',
    )
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for coordinate, row in zip(coordinates, rows, strict=True):
            writer.writerow(
                {
                    'x': float(coordinate[0]),
                    'y': float(coordinate[1]),
                    **{name: row.get(name) for name in fields[2:]},
                }
            )


def write_update_tsne_artifacts(
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, np.ndarray],
    output_directory: Path,
    *,
    benchmark_label: str,
    condition: str,
    support_count: int,
    representations: Sequence[str] = DEFAULT_UPDATE_REPRESENTATIONS,
    splits: Sequence[str] = ('train', 'latent_validation', 'family_validation'),
    pca_components: int = 50,
    perplexities: Sequence[float] = (30.0, 80.0),
    max_iter: int = 1_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Write deterministic PCA+t-SNE embeddings and interactive HTML views."""

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, trustworthiness

    row_condition = _row_array(rows, 'condition')
    row_count = _row_array(rows, 'support_count').astype(np.int32)
    row_split = _row_array(rows, 'target_split')
    selected = (
        (row_condition == str(condition))
        & (row_count == int(support_count))
        & np.isin(row_split, tuple(str(value) for value in splits))
    )
    selected_indices = np.flatnonzero(selected)
    if selected_indices.size < 4:
        raise ValueError('t-SNE requires at least four selected update rows.')
    selected_rows = [rows[index] for index in selected_indices]

    output_directory.mkdir(parents=True, exist_ok=False)
    embedding_arrays: dict[str, np.ndarray] = {
        'source_row_index': selected_indices.astype(np.int32),
    }
    manifest: dict[str, Any] = {
        'condition': str(condition),
        'support_count': int(support_count),
        'rows': int(selected_indices.size),
        'splits': list(splits),
        'representations': list(representations),
        'random_projection_used': False,
        'preprocessing': 'row L2 normalization followed by deterministic PCA',
        'clustering_original_space': update_clustering_summary(
            rows,
            features,
            condition=condition,
            support_count=support_count,
            representations=representations,
            splits=splits,
        ),
        'embeddings': [],
    }

    for representation_index, representation in enumerate(representations):
        if representation not in features:
            raise ValueError(f'Unknown update representation {representation!r}.')
        matrix = np.asarray(features[representation])[selected_indices]
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f'{representation} must be a finite rank-two feature matrix.'
            )
        normalized = _l2_normalize(matrix).astype(np.float32)
        component_count = min(
            int(pca_components),
            normalized.shape[0] - 1,
            normalized.shape[1],
        )
        if component_count < 2:
            raise ValueError('PCA preprocessing requires at least two components.')
        pca = PCA(
            n_components=component_count,
            svd_solver='randomized',
            random_state=int(seed) + representation_index,
        )
        reduced = pca.fit_transform(normalized).astype(np.float32)
        explained = float(np.sum(pca.explained_variance_ratio_))

        for perplexity_index, requested_perplexity in enumerate(perplexities):
            perplexity = min(float(requested_perplexity), selected_indices.size - 1.0)
            if perplexity <= 0.0:
                raise ValueError('t-SNE perplexities must be positive.')
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate='auto',
                init='pca',
                max_iter=int(max_iter),
                metric='euclidean',
                random_state=(
                    int(seed) + 100 * representation_index + perplexity_index
                ),
            )
            coordinates = tsne.fit_transform(reduced).astype(np.float32)
            perplexity_name = f'{perplexity:g}'.replace('.', 'p')
            stem = f'{representation}_p{perplexity_name}'
            embedding_arrays[stem] = coordinates

            family_path = output_directory / f'{stem}_by_family.html'
            heldout_path = output_directory / f'{stem}_heldout_focus.html'
            motion_path = output_directory / f'{stem}_by_terminal_phase.html'
            csv_path = output_directory / f'{stem}.csv'
            title = (
                f'{benchmark_label} {representation.replace("_", " ")} '
                f'(perplexity {perplexity:g})'
            )
            _write_family_plot(
                coordinates,
                selected_rows,
                family_path,
                title=f'{title}: family',
            )
            _write_heldout_plot(
                coordinates,
                selected_rows,
                heldout_path,
                title=(
                    f'{title}: held-out families over familiar-family context '
                    '(lines join repeated samples of one instance)'
                ),
            )
            _write_motion_plot(
                coordinates,
                selected_rows,
                motion_path,
                title=f'{title}: terminal motion primitive',
            )
            _write_embedding_csv(csv_path, coordinates, selected_rows)
            neighbors = min(15, max(1, (selected_indices.size - 1) // 2))
            manifest['embeddings'].append(
                {
                    'representation': representation,
                    'requested_perplexity': float(requested_perplexity),
                    'effective_perplexity': perplexity,
                    'pca_components': component_count,
                    'pca_explained_variance': explained,
                    'trustworthiness_vs_pca_space': float(
                        trustworthiness(
                            reduced,
                            coordinates,
                            n_neighbors=neighbors,
                            metric='euclidean',
                        )
                    ),
                    'kl_divergence': float(tsne.kl_divergence_),
                    'family_plot': family_path.name,
                    'heldout_focus_plot': heldout_path.name,
                    'terminal_phase_plot': motion_path.name,
                    'coordinates_csv': csv_path.name,
                }
            )

    np.savez_compressed(output_directory / 'embeddings.npz', **embedding_arrays)
    manifest['embeddings_npz'] = 'embeddings.npz'
    _write_json(output_directory / 'manifest.json', manifest)
    return manifest


__all__ = [
    'DEFAULT_UPDATE_REPRESENTATIONS',
    'update_clustering_summary',
    'write_update_tsne_artifacts',
]
