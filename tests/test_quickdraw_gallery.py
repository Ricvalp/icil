from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from icil_jax_rlbench.quickdraw.gallery import export_gallery, select_gallery
from icil_jax_rlbench.quickdraw.metrics import save_trajectories


def _fixture(path):
    tokens = np.zeros((4, 5, 4), np.float32)
    tokens[0, :3, :3] = [[-0.5, 0, 0], [0, 0.5, 1], [0.5, 0.5, 0]]
    tokens[0, 3, 3] = 1
    tokens[1, 0, 3] = 1
    tokens[2, :2, :3] = [[0, 0, 0], [1.4, 0, 1]]
    tokens[3, 0, 0] = np.nan
    tokens[3, 1, 3] = 1
    lengths, stopped = np.asarray([3, 0, 2, 1]), np.asarray([1, 1, 0, 1], bool)
    arrays = {'tokens': tokens, 'lengths': lengths, 'stopped': stopped,
              'point_mask': np.arange(5)[None] < lengths[:, None],
              'event_mask': np.arange(5)[None] < (lengths + stopped)[:, None]}
    metadata = {'schema_version': 1, 'coordinate_mode': 'absolute', 'pen_semantics': 'incoming',
                'records': [{'intended_category': '<cat&dog>', 'task_id': 'task0', 'sample_index': i, 'seed': 4, 'condition': 'correct_support'} for i in range(4)]}
    save_trajectories(path, arrays, metadata)
    return metadata


def test_selection_reads_metadata_only_and_cannot_replace_fixed_ids(tmp_path):
    artifact = tmp_path / 'artifact'
    artifact.mkdir()
    (artifact / 'metadata.json').write_text(json.dumps({'records': [{'intended_category': 'a', 'seed': i} for i in range(8)]}))
    # There is deliberately no trajectories.npz to inspect during selection.
    left, right = tmp_path / 'selection1.json', tmp_path / 'selection2.json'
    select_gallery(artifact, left, count=4, seed=8)
    select_gallery(artifact, right, count=4, seed=8)
    assert left.read_text() == right.read_text()
    with pytest.raises(FileExistsError, match='preserve'):
        select_gallery(artifact, left, count=4, seed=9)
    with pytest.raises(ValueError, match='cannot exceed'):
        select_gallery(artifact, tmp_path / 'too-many.json', count=9)


def test_gallery_preserves_pen_lifts_reports_failures_and_reuses_paired_ids(tmp_path):
    artifact = tmp_path / 'artifact'
    metadata = _fixture(artifact)
    selection, output = tmp_path / 'selection.json', tmp_path / 'gallery.svg'
    select_gallery(artifact, selection, count=4)
    export_gallery(artifact, output, selection_path=selection, columns=2)
    original = output.read_text()
    export_gallery(artifact, output, selection_path=selection, columns=2)
    assert output.read_text() == original
    root = ET.parse(output).getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    panel = root.find("svg:g[@data-row-index='0']", ns)
    lines = panel.findall('svg:line', ns)
    assert [line.attrib['data-segment'] for line in lines] == ['incoming-drawing', 'pen-up-travel']
    assert len(panel.findall("svg:circle[@data-mark='stroke-start']", ns)) == 2
    assert 'EMPTY' in original and 'NO STOP' in original and 'NONFINITE OUTPUT' in original
    assert '&lt;cat&amp;dog&gt;' in original
    provenance = json.loads(output.with_suffix('.svg.json').read_text())
    assert provenance['selected_count'] == 4
    assert provenance['selected_records'][2]['out_of_bounds_count'] == 1
    assert provenance['selected_records'][3]['invalid']
    for record in metadata['records']:
        record['condition'] = 'wrong_support'
        record['support_ids'] = ['different-support']
    (artifact / 'metadata.json').write_text(json.dumps(metadata))
    export_gallery(artifact, tmp_path / 'wrong.svg', selection_path=selection)
    metadata['records'][0]['seed'] += 1
    (artifact / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='identities differ'):
        export_gallery(artifact, tmp_path / 'changed.svg', selection_path=selection)
