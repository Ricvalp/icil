"""Preselected qualitative SVG galleries; separate from canonical raster metrics."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

from .metrics import file_sha256, load_trajectories


GALLERY_VERSION = 'quickdraw_qualitative_svg_v1'


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _population(metadata: dict) -> list[dict]:
    population = []
    for index, record in enumerate(metadata['records']):
        # Exclude support/control/model fields so paired controls reuse selection.
        identity = {
            'row_index': index,
            'intended_category': record['intended_category'],
            'task_id': record.get('task_id', record.get('intended_task', record.get('intended_base_id'))),
            'drawing_id': record.get('drawing_id', record.get('sample_id')),
            'query_index': record.get('query_index'),
            'sample_index': record.get('sample_index'),
            'query_seed': record.get('query_seed', record.get('seed')),
        }
        population.append({'record_id': _digest(identity), **identity})
    return population


def select_gallery(
    artifact: str | Path,
    selection_path: str | Path,
    *,
    count: int = 16,
    seed: int = 0,
) -> Path:
    """Freeze sample IDs using metadata only, before reading any trajectories.

    Reuse the selection file for controls with the same category/task/seed rows.
    This function never filters, ranks or replaces samples based on output quality.
    """
    artifact, selection_path = Path(artifact), Path(selection_path)
    metadata = json.loads((artifact / 'metadata.json').read_text())
    population = _population(metadata)
    if count < 1 or count > len(population):
        raise ValueError('Gallery count must be positive and cannot exceed the frozen population')
    if selection_path.exists():
        raise FileExistsError(f'Gallery selection already exists; preserve its fixed IDs: {selection_path}')
    indices = np.sort(np.random.default_rng(seed).choice(len(population), count, replace=False))
    selection = {
        'schema_version': 1,
        'gallery_version': GALLERY_VERSION,
        'selection_rule': 'seeded_uniform_without_replacement_from_metadata_only',
        'selection_seed': int(seed),
        'population_count': len(population),
        'selected_count': count,
        'population_sha256': _digest(population),
        'source_metadata_sha256': file_sha256(artifact / 'metadata.json'),
        'selected': [population[int(index)] for index in indices],
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2, allow_nan=False) + '\n')
    return selection_path


def _text(x, y, value, *, size=12, fill='#18212f') -> str:
    return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="{fill}">{html.escape(str(value))}</text>'


def _short(value, width=35):
    value = str(value)
    return value if len(value) <= width else value[:width - 3] + '...'


def _panel(tokens, point_mask, stopped, record, identity, x, y, width=260, height=355):
    points = np.asarray(tokens)[np.asarray(point_mask, dtype=bool), :3]
    invalid = bool(np.any(~np.isfinite(points)))
    finite_xy = np.all(np.isfinite(points[:, :2]), axis=1)
    outside = finite_xy & np.any(np.abs(points[:, :2]) > 1.0, axis=1)
    status = []
    if invalid:
        status.append('NONFINITE OUTPUT')
    if len(points) == 0:
        status.append('EMPTY')
    if not stopped:
        status.append('NO STOP')
    if np.any(outside):
        status.append(f'OUTSIDE CANVAS: {int(outside.sum())}')
    state = ', '.join(status) if status else 'STOP observed'
    color = '#b42318' if status else '#17613a'
    plot_x, plot_y, plot_size = x + 14, y + 89, 232
    parts = [f'<g data-record-id="{identity["record_id"]}" data-row-index="{identity["row_index"]}">',
             f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#d6dce4"/>',
             _text(x + 12, y + 20, _short(record['intended_category']), size=15),
             _text(x + 12, y + 37, f'task: {_short(identity["task_id"], 30)}'),
             _text(x + 12, y + 53, f'control: {_short(record.get("condition", "unspecified"), 27)}'),
             _text(x + 12, y + 69, f'seed: {_short(identity["query_seed"], 29)}'),
             _text(x + 12, y + 83, f'row {identity["row_index"]} | id {identity["record_id"][:12]}', size=10, fill='#566273'),
             f'<rect x="{plot_x}" y="{plot_y}" width="{plot_size}" height="{plot_size}" fill="#000000"/>']
    # Fixed normalized canvas, no inferred fit, rotation, translation or recentering.
    if invalid:
        parts += [f'<path d="M {plot_x + 20} {plot_y + 20} L {plot_x + plot_size - 20} {plot_y + plot_size - 20} M {plot_x + plot_size - 20} {plot_y + 20} L {plot_x + 20} {plot_y + plot_size - 20}" stroke="#ef6b62" stroke-width="3" fill="none"/>']
    elif len(points):
        xy = np.clip((points[:, :2] + 1.0) * 0.5, 0.0, 1.0) * (plot_size - 1)
        xy += [plot_x, plot_y]
        line_width = 2.0 * plot_size / 64.0
        for index in range(1, len(points)):
            pen_down = points[index, 2] >= 0.5
            start, end = xy[index - 1], xy[index]
            style = f'stroke="#ffffff" stroke-width="{line_width:.4f}"' if pen_down else 'stroke="#788391" stroke-width="1" stroke-dasharray="3 4"'
            segment_type = 'incoming-drawing' if pen_down else 'pen-up-travel'
            parts.append(f'<line data-segment="{segment_type}" x1="{start[0]:.5f}" y1="{start[1]:.5f}" x2="{end[0]:.5f}" y2="{end[1]:.5f}" {style} stroke-linecap="round"/>')
        starts = np.flatnonzero((np.arange(len(points)) == 0) | (points[:, 2] < 0.5))
        for index in starts:
            parts.append(f'<circle data-mark="stroke-start" cx="{xy[index, 0]:.5f}" cy="{xy[index, 1]:.5f}" r="{line_width * 0.5:.4f}" fill="#ffffff"/>')
        for index in np.flatnonzero(outside):
            parts.append(f'<circle data-mark="out-of-bounds" cx="{xy[index, 0]:.5f}" cy="{xy[index, 1]:.5f}" r="5" stroke="#ffac47" stroke-width="2" fill="none"/>')
    parts += [_text(x + 12, y + 338, _short(f'{len(points)} points | {state}', 42), size=10, fill=color), '</g>']
    return '\n'.join(parts), {'record_id': identity['record_id'], 'row_index': identity['row_index'], 'point_count': len(points), 'invalid': invalid, 'empty': len(points) == 0, 'no_stop': not bool(stopped), 'out_of_bounds_count': int(outside.sum()), 'record': record}


def export_gallery(
    artifact: str | Path,
    output_svg: str | Path,
    *,
    selection_path: str | Path,
    columns: int = 4,
) -> Path:
    """Render precisely the previously selected IDs, including failed outputs.

    SVG draws white incoming-pen strokes, gray dashed pen-up travel, and orange
    out-of-bounds indicators. It is a qualitative diagnostic, never Sketch-FD input
    or a replacement for the canonical Pillow rasterizer.
    """
    artifact, output_svg, selection_path = Path(artifact), Path(output_svg), Path(selection_path)
    if columns < 1:
        raise ValueError('Gallery columns must be positive')
    selection = json.loads(selection_path.read_text())
    if selection.get('schema_version') != 1 or selection.get('gallery_version') != GALLERY_VERSION:
        raise ValueError('Unknown gallery selection schema')
    metadata = json.loads((artifact / 'metadata.json').read_text())
    population = _population(metadata)
    if _digest(population) != selection['population_sha256']:
        raise ValueError('Gallery population/category/task/seed identities differ from frozen selection')
    selected = selection['selected']
    if len(selected) != selection['selected_count'] or not selected:
        raise ValueError('Gallery selected count is inconsistent')
    by_id = {record['record_id']: record for record in population}
    if len(set(record['record_id'] for record in selected)) != len(selected):
        raise ValueError('Gallery selected IDs must be distinct')
    for record in selected:
        if by_id.get(record['record_id']) != record:
            raise ValueError('Gallery selected record identity was changed')
    # Selection is fully validated before any output coordinates are loaded.
    arrays, metadata = load_trajectories(artifact)
    columns = min(columns, len(selected))
    rows = math.ceil(len(selected) / columns)
    width, height = columns * 274 + 14, rows * 369 + 100
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<title>QuickDraw preselected qualitative gallery</title>',
             '<desc>Fixed absolute canvas; destination pen flag controls each incoming segment. Qualitative SVG, not canonical metric rasters.</desc>',
             f'<rect width="{width}" height="{height}" fill="#f0f3f7"/>',
             _text(14, 25, 'QuickDraw | frozen sample selection', size=20),
             _text(14, 46, 'Qualitative SVG: fixed [-1,1] canvas; white drawing; dashed gray pen-up travel; orange outside-canvas points.', size=11),
             _text(14, 64, f'{len(selected)} selected outputs | seed {selection["selection_seed"]} | canonical Sketch-FD rasterization runs separately.', size=11),
             _text(14, 81, f'model: {_short(metadata.get("checkpoint_id", metadata.get("model_id", "unspecified")), 90)}', size=11)]
    records = []
    for cell, identity in enumerate(selected):
        index = identity['row_index']
        panel, record = _panel(arrays['tokens'][index], arrays['point_mask'][index], arrays['stopped'][index], metadata['records'][index], identity, 14 + cell % columns * 274, 94 + cell // columns * 369)
        parts.append(panel)
        records.append(record)
    parts.append('</svg>')
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text('\n'.join(parts) + '\n')
    sidecar = {
        'gallery_version': GALLERY_VERSION, 'artifact_metadata_sha256': file_sha256(artifact / 'metadata.json'),
        'trajectories_sha256': file_sha256(artifact / 'trajectories.npz'),
        'selection_sha256': file_sha256(selection_path), 'selected_count': len(selected),
        'rendering_role': 'qualitative_only_not_canonical_metric_input',
        'coordinate_mode': 'absolute', 'pen_semantics': 'incoming', 'frame_alignment': 'none',
        'selected_records': records,
    }
    output_svg.with_suffix(output_svg.suffix + '.json').write_text(json.dumps(sidecar, indent=2, allow_nan=False) + '\n')
    return output_svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    select = commands.add_parser('select', help='Freeze IDs from metadata before rendering')
    select.add_argument('artifact')
    select.add_argument('--output', required=True)
    select.add_argument('--count', type=int, default=16)
    select.add_argument('--seed', type=int, default=0)
    render = commands.add_parser('render', help='Render exactly the already frozen IDs')
    render.add_argument('artifact')
    render.add_argument('--selection', required=True)
    render.add_argument('--output', required=True)
    render.add_argument('--columns', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'select':
        result = select_gallery(args.artifact, args.output, count=args.count, seed=args.seed)
    else:
        result = export_gallery(args.artifact, args.output, selection_path=args.selection, columns=args.columns)
    print(result)


if __name__ == '__main__':
    main()
