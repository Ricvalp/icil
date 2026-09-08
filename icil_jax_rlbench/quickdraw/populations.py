"""Freeze A support/training populations for offline copying measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import SketchStore, load_manifest, prepare_sequence
from .metrics import file_sha256, save_trajectories


def export_copy_populations(
    cache_root: str | Path,
    manifest_path: str | Path,
    episodes: str | Path,
    output: str | Path,
    training_count: int | None = None,
    seed: int = 0,
) -> Path:
    """Export actual canonical A support IDs and a fixed permitted train subset.

    Correct and wrong supports share one deduplicated feature reservoir. The
    scorer selects each generated sample's shown IDs from that reservoir. B's
    varying support frames and transformed-support controls require distinct
    execution IDs and are deliberately unsupported by this canonical exporter.
    """
    episodes, output = Path(episodes), Path(output)
    metadata = json.loads((episodes / 'metadata.json').read_text())
    payload = {key: value for key, value in metadata.items() if key != 'identifier'}
    expected_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if metadata.get('identifier') != expected_hash:
        raise ValueError('Frozen episode manifest hash mismatch')
    if metadata.get('experiment') != 'a':
        raise ValueError('Copy population export supports canonical A only; B requires unique framed execution IDs')
    if metadata.get('data_sha256') != file_sha256(episodes / 'episodes.npz'):
        raise ValueError('Frozen episode arrays differ from their recorded hash')
    store = SketchStore.open(cache_root)
    manifest = load_manifest(manifest_path, store)
    if metadata.get('cache_id') != store.identifier or metadata.get('manifest_id') != manifest['identifier']:
        raise ValueError('Frozen episodes refer to a different cache or manifest')
    split = metadata['split']
    max_steps = int(metadata['max_steps'])
    permitted_support_ids = {str(item) for ids in manifest['a_ids'][split].values() for item in ids}
    shown_roles = {}
    with np.load(episodes / 'episodes.npz', allow_pickle=False) as archive:
        roles = [('support', 'records'), ('wrong_support', 'wrong_records')]
        if metadata.get('same_category_wrong_records'):
            roles.append(('same_category_wrong_neighborhood', 'same_category_wrong_records'))
        for role, record_key in roles:
            records = metadata[record_key]
            tokens, mask = archive[f'{role}.tokens'], archive[f'{role}.point_mask']
            frames = archive[f'{role}.frame']
            if len(records) != len(tokens) or mask.shape != tokens.shape[:-1]:
                raise ValueError('Support record counts/masks differ from frozen arrays')
            if not np.array_equal(frames, np.broadcast_to(np.asarray([0, 0, 0, 1]), frames.shape)):
                raise ValueError('Canonical A support export cannot discard nonidentity frames')
            for task_index, record in enumerate(records):
                ids = record['support_ids']
                if len(ids) != tokens.shape[1]:
                    raise ValueError('Support demo count differs from frozen IDs')
                for demo_index, item in enumerate(ids):
                    item = str(item)
                    if item not in permitted_support_ids:
                        raise ValueError('Shown support ID is outside the declared evaluation reservoir')
                    drawing = store.get(item)
                    actual = tokens[task_index, demo_index][mask[task_index, demo_index].astype(bool), :3]
                    canonical = np.column_stack([drawing.absolute, drawing.incoming_pen])
                    if not np.array_equal(actual, canonical):
                        raise ValueError('Shown supports differ from canonical A records; export unique transformed execution IDs')
                    shown_roles.setdefault(item, set()).add('correct_support' if role == 'support' else role)
    support_ids = sorted(shown_roles)
    training_ids = sorted({str(item) for ids in manifest['a_ids']['train'].values() for item in ids})
    if training_count is None:
        training_count = len(training_ids)
    if not isinstance(training_count, (int, np.integer)) or not 1 <= training_count <= len(training_ids):
        raise ValueError('training_count must be positive and cannot exceed the permitted training reservoir')
    # This uses IDs and a declared seed, never generated sketches or features.
    selected_train = sorted(np.random.default_rng(seed).permutation(training_ids)[:training_count].tolist())
    selection = {
        'schema_version': 1, 'experiment': 'a', 'manifest_id': manifest['identifier'],
        'cache_id': store.identifier, 'evaluation_id': metadata['identifier'],
        'selection_seed': int(seed),
        'selection_rule': 'shown_support_id_union_and_seeded_training_ID_permutation',
        'support_ids': support_ids,
        'correct_support_ids': [item for item in support_ids if 'correct_support' in shown_roles[item]],
        'wrong_support_ids': [item for item in support_ids if 'wrong_support' in shown_roles[item]],
        'same_category_wrong_support_ids': [item for item in support_ids if 'same_category_wrong_neighborhood' in shown_roles[item]],
        'a_protocol': metadata.get('a_protocol', 'a_category'),
        'training_ids': selected_train,
        'selected_training_count': len(selected_train),
        'permitted_training_count': len(training_ids),
        'max_steps': max_steps,
    }
    selection['identifier'] = hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False) + '\n')

    def export(ids, directory, population_role):
        sequences, records = [], []
        for item in ids:
            drawing = store.get(item)
            sequences.append(prepare_sequence(drawing.absolute, drawing.incoming_pen, max_steps))
            records.append({
                'drawing_id': item, 'base_id': item, 'intended_category': drawing.category,
                'duplicate_cluster_id': drawing.duplicate_cluster_id,
                'support_ids': [], 'condition': population_role,
                'shown_roles': sorted(shown_roles.get(item, ())),
                'frame': [0.0, 0.0, 0.0, 1.0],
            })
        arrays = {key: np.stack([row[key] for row in sequences]) for key in ('tokens', 'point_mask', 'event_mask')}
        arrays.update(lengths=np.asarray([int(row['point_mask'].sum()) for row in sequences], np.int32), stopped=np.ones(len(sequences), bool))
        save_trajectories(output / directory, arrays, {
            'schema_version': 1, 'coordinate_mode': 'absolute', 'pen_semantics': 'incoming',
            'records': records, 'expected_count': len(records), 'manifest_id': manifest['identifier'],
            'cache_id': store.identifier, 'evaluation_id': metadata['identifier'], 'experiment': 'a',
            'population_role': population_role, 'id_selection_sha256': selection['identifier'],
            'selection_path': str(output / 'selection.json'),
        })

    export(support_ids, 'supports', 'actual_correct_and_wrong_support_reservoir')
    export(selected_train, 'training_reservoir', 'permitted_training_reservoir')
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('cache-root', 'manifest-path', 'episodes', 'output'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--training-count', type=int)
    parser.add_argument('--seed', type=int, default=0)
    print(export_copy_populations(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
