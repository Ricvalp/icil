"""Native, immutable QuickDraw records, manifests, and task sampling.

No donor loader, learned retrieval index, simulator, or tensor framework is
needed. IDs and private lengths remain metadata, outside model input fields.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .pen import (
    IDENTITY_FRAME,
    PenConfig,
    apply_frame,
    invert_frame,
    replay_rollout,
    timed_reference,
    transformed_replay,
)


SCHEMA_VERSION = 'quickdraw-v1'
DUPLICATE_RULE = 'same_point_count_and_incoming_pen_xy_rounded_to_2_decimal_places'
SPLITS = ('train', 'development', 'test')


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def absolute_to_deltas(absolute: np.ndarray) -> np.ndarray:
    """Origin is (0,0); pen-up movement remains a real displacement."""
    absolute = np.asarray(absolute, np.float32)
    return np.diff(absolute, axis=0, prepend=np.zeros((1, 2), np.float32))


def deltas_to_absolute(deltas: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(deltas, np.float32), axis=0)


def duplicate_hash(absolute: np.ndarray, incoming_pen: np.ndarray) -> str:
    xy = np.rint(np.asarray(absolute, np.float64) * 100).astype('<i4')
    pen = np.asarray(incoming_pen, np.uint8)
    return hashlib.sha256(xy.tobytes() + pen.tobytes()).hexdigest()


@dataclass(frozen=True)
class SketchRecord:
    base_id: str
    category: str
    absolute: np.ndarray
    incoming_pen: np.ndarray
    duplicate_cluster_id: str = ''
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        xy = np.array(self.absolute, dtype=np.float32, copy=True)
        pen = np.array(self.incoming_pen, dtype=np.float32, copy=True)
        if not self.base_id or not self.category:
            raise ValueError('Record ID and category must be nonempty strings.')
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or pen.shape != (len(xy),):
            raise ValueError('A record needs nonempty [points,2] XY and [points] pen.')
        if not np.all(np.isfinite(xy)) or not np.all((pen == 0) | (pen == 1)):
            raise ValueError('Coordinates must be finite and pen flags binary.')
        if pen[0] != 0:
            raise ValueError('The first point of a drawing must start with pen up.')
        xy.setflags(write=False)
        pen.setflags(write=False)
        object.__setattr__(self, 'base_id', str(self.base_id))
        object.__setattr__(self, 'category', str(self.category))
        object.__setattr__(self, 'absolute', xy)
        object.__setattr__(self, 'incoming_pen', pen)
        object.__setattr__(self, 'provenance', deepcopy(self.provenance))
        if not self.duplicate_cluster_id:
            object.__setattr__(self, 'duplicate_cluster_id', duplicate_hash(xy, pen))

    @property
    def length(self) -> int:
        return len(self.absolute)

    @property
    def deltas(self) -> np.ndarray:
        return absolute_to_deltas(self.absolute)


def _cache_identifier(records: Iterable[SketchRecord], provenance: dict) -> str:
    digest = hashlib.sha256(_json_bytes({'schema': SCHEMA_VERSION, 'provenance': provenance}))
    for record in records:
        digest.update(_json_bytes({
            'base_id': record.base_id, 'category': record.category,
            'duplicate_cluster_id': record.duplicate_cluster_id,
            'provenance': record.provenance,
        }))
        digest.update(np.asarray(record.absolute, '<f4').tobytes())
        digest.update(np.asarray(record.incoming_pen, np.uint8).tobytes())
    return digest.hexdigest()


class SketchStore:
    """Ragged NPZ numeric records and a JSON index; never pickle or donor imports."""

    def __init__(self, records: Iterable[SketchRecord], *, provenance: dict | None = None):
        self.records = tuple(sorted(records, key=lambda record: record.base_id))
        self.provenance = deepcopy(provenance or {})
        self._by_id = {record.base_id: record for record in self.records}
        if not self.records or len(self._by_id) != len(self.records):
            raise ValueError('A cache must be nonempty with globally unique base IDs.')
        self.identifier = _cache_identifier(self.records, self.provenance)
        self.categories = tuple(sorted({record.category for record in self.records}))

    def get(self, base_id: str) -> SketchRecord:
        return self._by_id[str(base_id)]

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def write(
        cls, root: str | Path, records: Iterable[SketchRecord], *, provenance: dict | None = None
    ) -> 'SketchStore':
        root = Path(root)
        if (root / 'records.npz').exists() or (root / 'index.json').exists():
            raise FileExistsError(f'Immutable cache already exists: {root}')
        store = cls(records, provenance=provenance)
        root.mkdir(parents=True, exist_ok=True)
        offsets = np.cumsum([0] + [record.length for record in store.records], dtype=np.int64)
        np.savez_compressed(
            root / 'records.npz', offsets=offsets,
            absolute=np.concatenate([r.absolute for r in store.records]),
            incoming_pen=np.concatenate([r.incoming_pen for r in store.records]),
        )
        index = {
            'schema_version': SCHEMA_VERSION, 'identifier': store.identifier,
            'coordinate_mode': 'absolute', 'delta_origin': [0.0, 0.0],
            'pen_semantics': 'destination_incoming_segment',
            'duplicate_rule': DUPLICATE_RULE, 'provenance': store.provenance,
            'records': [{
                'base_id': r.base_id, 'category': r.category,
                'duplicate_cluster_id': r.duplicate_cluster_id,
                'provenance': r.provenance,
            } for r in store.records],
        }
        (root / 'index.json').write_bytes(_json_bytes(index) + b'\n')
        return store

    @classmethod
    def open(cls, root: str | Path) -> 'SketchStore':
        root = Path(root)
        index = json.loads((root / 'index.json').read_text())
        if index.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported QuickDraw cache schema.')
        with np.load(root / 'records.npz', allow_pickle=False) as arrays:
            offsets = arrays['offsets']
            absolute, incoming_pen = arrays['absolute'], arrays['incoming_pen']
            if (offsets.shape != (len(index['records']) + 1,) or offsets[0] != 0
                    or offsets[-1] != len(absolute) or len(incoming_pen) != len(absolute)
                    or np.any(np.diff(offsets) <= 0)):
                raise ValueError('Invalid ragged cache offsets.')
            records = [SketchRecord(
                absolute=absolute[offsets[i]:offsets[i + 1]],
                incoming_pen=incoming_pen[offsets[i]:offsets[i + 1]], **metadata,
            ) for i, metadata in enumerate(index['records'])]
        store = cls(records, provenance=index['provenance'])
        if store.identifier != index['identifier']:
            raise ValueError('QuickDraw cache content hash mismatch.')
        return store


def preprocess_drawing(
    drawing: list, *, base_id: str, category: str, provenance: dict | None = None
) -> SketchRecord:
    """Match donor preprocessing with normalization and no extra resampling/RDP."""
    points, pens = [], []
    for stroke in drawing:
        if len(stroke) < 2 or len(stroke[0]) != len(stroke[1]):
            raise ValueError('Each raw stroke must have equally sized X and Y lists.')
        if not len(stroke[0]):
            continue
        xy = np.asarray(stroke[:2], np.float32).T
        points.append(xy)
        pens.extend([0.0] + [1.0] * (len(xy) - 1))
    if not points:
        raise ValueError('Drawing has no points.')
    absolute = np.concatenate(points)
    low, high = absolute.min(axis=0), absolute.max(axis=0)
    center = (low + high) / 2.0
    span = float(np.max(high - low))
    scale = (span if span > 0 else 1.0) / 2.0
    absolute = (absolute - center) / scale
    metadata = deepcopy(provenance or {})
    metadata['normalization'] = {'center': center.tolist(), 'scale': scale}
    return SketchRecord(base_id, category, absolute, np.asarray(pens), provenance=metadata)


def import_ndjson(
    raw_root: str | Path,
    cache_root: str | Path,
    *,
    max_points: int = 255,
    min_points: int = 1,
    max_drawings_per_category: int | None = None,
) -> SketchStore:
    """Convert local official simplified NDJSON; never download implicitly.

    A finite cap is an explicitly recorded source-prefix reservoir. It counts
    inspected raw records, so filtering cannot change the source population.
    """
    raw_root = Path(raw_root)
    paths = [raw_root] if raw_root.is_file() else sorted(raw_root.glob('*.ndjson'))
    if not paths:
        raise FileNotFoundError(f'No local QuickDraw NDJSON files in {raw_root}')
    if min_points < 1 or max_points < min_points:
        raise ValueError('Invalid point eligibility bounds.')
    if max_drawings_per_category is not None and max_drawings_per_category < 1:
        raise ValueError('max_drawings_per_category must be positive.')
    records, summaries, sources = [], {}, []
    for path in paths:
        category = path.stem
        stats = Counter()
        lengths = []
        source_hash = hashlib.sha256()
        with path.open('rb') as handle:
            for raw_index, line in enumerate(handle):
                if max_drawings_per_category is not None and raw_index >= max_drawings_per_category:
                    break
                source_hash.update(line)
                stats['inspected'] += 1
                try:
                    raw = json.loads(line)
                    raw_id = raw.get('key_id')
                    if raw_id is None:
                        raw_id = hashlib.sha256(line).hexdigest()
                    record = preprocess_drawing(
                        raw['drawing'], base_id=f'{category}/{raw_id}', category=category,
                        provenance={'raw_id': str(raw_id), 'source_file': path.name,
                                    'raw_index': raw_index, 'recognized': raw.get('recognized')},
                    )
                except (ValueError, KeyError, TypeError):
                    stats['invalid'] += 1
                    continue
                lengths.append(record.length)
                if not min_points <= record.length <= max_points:
                    stats['outside_length_bounds'] += 1
                    continue
                records.append(record)
                stats['retained'] += 1
        stats['raw_point_length_histogram'] = dict(Counter(map(str, lengths)))
        summaries[category] = dict(stats)
        sources.append({'path': str(path.resolve()), 'read_prefix_sha256': source_hash.hexdigest()})
    return SketchStore.write(cache_root, records, provenance={
        'kind': 'official_ndjson', 'sources': sources, 'retention': summaries,
        'preprocessing': 'donor_bbox_normalize_v1_no_resampling_no_simplification',
        'eligibility': {'min_points': min_points, 'max_points': max_points},
        'source_prefix_cap_per_category': max_drawings_per_category,
    })


def _rank(values: Iterable[str], seed: int, namespace: str) -> list[str]:
    return sorted(values, key=lambda value: _digest([namespace, seed, value]))


def _partition(values: list[str], train_fraction: float, development_fraction: float) -> dict:
    if len(values) < 3:
        raise ValueError('At least three eligible items are required for three disjoint splits.')
    development = max(1, int(len(values) * development_fraction))
    test = max(1, int(len(values) * (1 - train_fraction - development_fraction)))
    train = len(values) - development - test
    if train < 1:
        raise ValueError('Split fractions leave an empty training pool.')
    return dict(zip(SPLITS, [values[:train], values[train:train + development], values[train + development:]]))


def build_manifest(
    store: SketchStore,
    *,
    seed: int = 0,
    subset_seed: int = 0,
    train_fraction: float = 0.7,
    development_fraction: float = 0.15,
    family_count: int | None = None,
    program_count: int | None = None,
    unique_budget: int | None = None,
    drawings_per_category: int | None = None,
    reference_per_category: int = 4,
    max_points: int = 255,
) -> dict:
    """Fix category/base splits before nested F/N reservoir selection.

    Cross-category duplicate clusters are excluded. Other duplicate clusters
    contribute one stable representative, preventing cluster overlap by design.
    Real-reference halves are disjoint from all demonstration/query reservoirs.
    B familiar splits use distinct programs within the training categories;
    B heldout_category splits use A's development/test categories.
    """
    if not (0 < train_fraction < 1 and 0 < development_fraction < 1 - train_fraction):
        raise ValueError('Require positive train/development/test fractions.')
    if reference_per_category < 2 or reference_per_category % 2:
        raise ValueError('reference_per_category must be positive and even, at least two.')
    if unique_budget is not None and drawings_per_category is not None:
        raise ValueError('Choose a fixed unique budget or fixed drawings per category.')
    for name, value in [('family_count', family_count), ('program_count', program_count),
                        ('unique_budget', unique_budget), ('drawings_per_category', drawings_per_category)]:
        if value is not None and value < 1:
            raise ValueError(f'{name} must be positive.')
    clusters = defaultdict(list)
    for record in store.records:
        if record.length <= max_points:
            clusters[record.duplicate_cluster_id].append(record)
    category_ids = defaultdict(list)
    excluded_cross_category = []
    for cluster, records in sorted(clusters.items()):
        if len({r.category for r in records}) != 1:
            excluded_cross_category.append(cluster)
            continue
        representative = min(records, key=lambda record: record.base_id)
        category_ids[representative.category].append(representative.base_id)
    eligible = sorted(category for category, ids in category_ids.items() if len(ids) >= reference_per_category + 3)
    categories = _partition(_rank(eligible, seed, 'category_split'), train_fraction, development_fraction)
    references, pools = {}, {}
    for category in eligible:
        ids = _rank(category_ids[category], seed, 'drawing_split')
        references[category] = {
            'real_a': ids[:reference_per_category // 2],
            'real_b': ids[reference_per_category // 2:reference_per_category],
        }
        pools[category] = ids[reference_per_category:]
    category_order = _rank(categories['train'], subset_seed, 'nested_categories')
    selected_categories = category_order[:family_count] if family_count else category_order
    if family_count is not None and family_count > len(category_order):
        raise ValueError(f'Only {len(category_order)} training categories are available.')
    a_reservoir = {}
    for index, category in enumerate(selected_categories):
        count = len(pools[category])
        if unique_budget is not None:
            count = unique_budget // len(selected_categories) + int(index < unique_budget % len(selected_categories))
        elif drawings_per_category is not None:
            count = drawings_per_category
        if count < 1 or count > len(pools[category]):
            raise ValueError(f'Insufficient eligible drawings for requested budget in {category}.')
        a_reservoir[category] = _rank(pools[category], subset_seed, 'nested_drawings')[:count]
    a_ids = {
        'train': a_reservoir,
        'development': {cat: pools[cat] for cat in categories['development']},
        'test': {cat: pools[cat] for cat in categories['test']},
    }
    b_ids = {split: {} for split in SPLITS}
    for category in category_order:
        split_ids = _partition(pools[category], train_fraction, development_fraction)
        for split in SPLITS:
            b_ids[split][category] = split_ids[split]
    # Round-robin nesting fixes category coverage as N grows past its first cycle.
    ordered_programs = []
    ranked = {category: _rank(b_ids['train'][category], subset_seed, 'nested_programs') for category in category_order}
    for index in range(max(map(len, ranked.values()))):
        ordered_programs.extend(ids[index] for ids in ranked.values() if index < len(ids))
    if program_count is not None and program_count > len(ordered_programs):
        raise ValueError('Requested program count exceeds the training program pool.')
    if program_count is not None and program_count < len(category_order):
        raise ValueError('N must cover the fixed training categories; use a separate crossed category study.')
    selected_programs = set(ordered_programs[:program_count] if program_count else ordered_programs)
    b_ids['train'] = {cat: [item for item in ids if item in selected_programs] for cat, ids in ranked.items()}
    manifest = {
        'schema_version': SCHEMA_VERSION, 'cache_identifier': store.identifier,
        'a_protocol': 'a_category', 'split_regime': 'heldout_categories',
        'seed': seed, 'subset_seed': subset_seed,
        'categories': categories, 'nested_category_order': category_order,
        'category_splits': deepcopy(categories),
        'nested_program_order': ordered_programs,
        'a_ids': a_ids, 'b_ids': b_ids, 'reference_ids': references,
        'eligibility': {'min_points': 1, 'max_points': max_points,
                        'duplicate_rule': DUPLICATE_RULE,
                        'excluded_cross_category_clusters': excluded_cross_category},
        'budget': {'family_count': len(selected_categories),
                   'program_count': len(selected_programs),
                   'unique_a_train_drawings': sum(map(len, a_reservoir.values())),
                   'unique_budget': unique_budget, 'drawings_per_category': drawings_per_category,
                   'regime': 'fixed_unique_drawings' if unique_budget is not None else 'fixed_drawings_per_category'
                   if drawings_per_category is not None else 'all_eligible'},
        'retention': {cat: {'cache_records': sum(r.category == cat for r in store.records),
                           'unique_eligible_clusters': len(category_ids[cat]),
                           'reservoir_records': len(pools[cat])} for cat in eligible},
    }
    manifest['reservoir_identifiers'] = {
        experiment: {split: _digest(ids[split]) for split in SPLITS}
        for experiment, ids in [('a', a_ids), ('b_familiar', b_ids)]
    }
    manifest['identifier'] = _digest(manifest)
    validate_manifest(store, manifest)
    return manifest


def validate_manifest(store: SketchStore, manifest: dict) -> None:
    payload = {key: value for key, value in manifest.items() if key != 'identifier'}
    if manifest.get('identifier') != _digest(payload):
        raise ValueError('QuickDraw manifest hash mismatch.')
    if manifest.get('cache_identifier') != store.identifier or manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Manifest refers to a different cache/schema.')
    if 'neighborhood_schema_version' in manifest:
        from .neighborhoods import validate_neighborhood_manifest
        validate_neighborhood_manifest(store, manifest)
        return
    categories = manifest['categories']
    all_categories = [cat for split in SPLITS for cat in categories[split]]
    if len(set(all_categories)) != len(all_categories) or any(not categories[split] for split in SPLITS):
        raise ValueError('Category splits must be explicit, nonempty, and disjoint.')
    reference_clusters = set()
    for category, halves in manifest['reference_ids'].items():
        for name in ('real_a', 'real_b'):
            for item in halves[name]:
                record = store.get(item)
                if record.category != category or record.duplicate_cluster_id in reference_clusters:
                    raise ValueError('Real reference categories/clusters overlap or mismatch.')
                reference_clusters.add(record.duplicate_cluster_id)
    for key in ('a_ids', 'b_ids'):
        seen = set()
        for split in SPLITS:
            for category, ids in manifest[key][split].items():
                allowed = categories[split] if key == 'a_ids' else categories['train']
                if category not in allowed or not ids:
                    raise ValueError('Reservoir category does not match its declared split.')
                for item in ids:
                    record = store.get(item)
                    if (record.category != category or record.duplicate_cluster_id in seen
                            or record.duplicate_cluster_id in reference_clusters):
                        raise ValueError('Base/duplicate-cluster overlap or category mismatch.')
                    seen.add(record.duplicate_cluster_id)


def save_manifest(path: str | Path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(manifest) + b'\n'
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f'Refusing to replace immutable manifest: {path}')
        return
    with path.open('xb') as handle:
        handle.write(encoded)


def load_manifest(path: str | Path, store: SketchStore | None = None) -> dict:
    manifest = json.loads(Path(path).read_text())
    if store is not None:
        validate_manifest(store, manifest)
    elif manifest.get('identifier') != _digest({k: v for k, v in manifest.items() if k != 'identifier'}):
        raise ValueError('QuickDraw manifest hash mismatch.')
    return manifest


def prepare_sequence(
    absolute: np.ndarray, incoming_pen: np.ndarray, max_steps: int,
    *, frame: np.ndarray | None = None, state: np.ndarray | None = None,
    knot_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Pad one drawing; coordinate labels exclude STOP, event labels include it once.

    max_steps is a public fixed budget. Masks are loss masks, not inference
    inputs. For B2, the first argument contains motion commands instead of XY.
    """
    absolute = np.asarray(absolute, np.float32)
    incoming_pen = np.asarray(incoming_pen, np.float32)
    length = len(absolute)
    if absolute.shape != (length, 2) or incoming_pen.shape != (length,):
        raise ValueError('Expected [points,2] coordinates and [points] incoming pen.')
    if length + 1 > max_steps:
        raise ValueError('Drawing plus one STOP exceeds the fixed public budget.')
    if not np.all(np.isfinite(absolute)) or not np.all((incoming_pen == 0) | (incoming_pen == 1)):
        raise ValueError('Coordinates must be finite and pen flags binary.')
    tokens = np.zeros((max_steps, 4), np.float32)
    tokens[:length, :2] = absolute
    tokens[:length, 2] = incoming_pen
    tokens[length, 3] = 1.0
    result = {
        'tokens': tokens,
        'event_mask': np.arange(max_steps) <= length,
        'point_mask': np.arange(max_steps) < length,
        # Knots are public evidence on complete supports, labels on queries.
        'knot_mask': np.arange(max_steps) < length,
        'frame': np.array(IDENTITY_FRAME if frame is None else frame, np.float32),
        'state': np.zeros((max_steps, 3), np.float32),
    }
    if knot_mask is not None:
        knot_mask = np.asarray(knot_mask, bool)
        if knot_mask.shape != (length,):
            raise ValueError('Program-knot mask must have one entry per destination.')
        result['knot_mask'][:] = False
        result['knot_mask'][:length] = knot_mask
    if state is not None:
        state = np.asarray(state, np.float32)
        if state.shape != (length + 1, 3) or not np.all(np.isfinite(state)):
            raise ValueError('B2 pre-command states include points and one STOP.')
        result['state'][:length + 1] = state
    return result


def trim_generated(tokens: np.ndarray, *, coordinate_mode: str = 'absolute') -> dict:
    """Trim at the first predicted STOP, retaining missing-STOP and raw outputs."""
    tokens = np.asarray(tokens, np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 4:
        raise ValueError('Generated tokens must have shape [time,4].')
    stops = np.flatnonzero(tokens[:, 3] >= 0.5)
    length = int(stops[0]) if len(stops) else len(tokens)
    xy = tokens[:length, :2]
    if coordinate_mode == 'delta':
        xy = deltas_to_absolute(xy)
    elif coordinate_mode != 'absolute':
        raise ValueError('Use actual executed positions to export B2 control outputs.')
    return {'absolute': xy.copy(), 'incoming_pen': (tokens[:length, 2] >= 0.5).astype(np.float32),
            'length': length, 'has_stop': bool(len(stops)), 'raw_tokens': tokens.copy()}


class SketchSampler:
    """Category-first A and base-program B sampling, with replayable RNG state."""

    def __init__(
        self, store: SketchStore, manifest: dict, *, experiment: str = 'a',
        split: str = 'train', seed: int = 0, max_steps: int = 256,
        support_count: int = 4, query_count: int = 1, b_partition: str = 'familiar',
        pen_config: PenConfig = PenConfig(),
    ):
        validate_manifest(store, manifest)
        if experiment not in ('a', 'b1', 'b2') or split not in SPLITS:
            raise ValueError('Expected experiment a/b1/b2 and train/development/test split.')
        if b_partition not in ('familiar', 'heldout_category'):
            raise ValueError('B partition must be familiar or heldout_category.')
        if support_count < 1 or query_count < 1 or max_steps < 2:
            raise ValueError('Require positive support/query counts and at least two steps.')
        if experiment == 'b2' and (pen_config.noise_std or pen_config.delay_steps or pen_config.motion_gain != 1):
            raise ValueError('Training B2 uses deterministic expert demonstrations; evaluate disturbances separately.')
        self.store, self.manifest = store, deepcopy(manifest)
        self.experiment, self.split, self.b_partition = experiment, split, b_partition
        self.max_steps, self.support_count, self.query_count = max_steps, support_count, query_count
        self.pen_config = pen_config
        streams = np.random.SeedSequence(seed).spawn(4)
        self.category_rng, self.drawing_rng, self.frame_rng, self.start_rng = (
            np.random.default_rng(stream) for stream in streams
        )
        self.draw_count = 0
        key = 'a_ids' if experiment == 'a' or (b_partition == 'heldout_category' and split != 'train') else 'b_ids'
        if experiment != 'a' and b_partition == 'heldout_category' and split != 'train' and 'b_heldout_category_ids' in manifest:
            key = 'b_heldout_category_ids'
        self.pool = deepcopy(manifest[key][split])
        self.manifest_reservoir_counts = {category: len(ids) for category, ids in self.pool.items()}
        # Fix eligible pools once, never reject/resample a category after drawing it.
        for category, ids in self.pool.items():
            eligible = []
            for item in ids:
                record = store.get(item)
                if experiment == 'b2':
                    # Frames: scale <= .9, translation norm <= sqrt(2)*.1;
                    # starts have norm <= sqrt(2)*.8. Bound before sampling.
                    lengths = np.linalg.norm(np.diff(record.absolute, axis=0), axis=1)
                    travel = (.9 * np.linalg.norm(record.absolute[0]) + np.sqrt(2) * .9)
                    upper_steps = int(np.ceil(travel / (pen_config.max_motion * .999)))
                    upper_steps += int(np.maximum(1, np.ceil(.9 * lengths / (pen_config.max_motion * .999))).sum()) + 1
                else:
                    upper_steps = record.length + 1
                if upper_steps <= max_steps:
                    eligible.append(item)
            minimum = support_count + query_count if experiment == 'a' else 1
            if len(eligible) < minimum:
                raise ValueError(f'{category} has only {len(eligible)} eligible drawings; need {minimum}.')
            self.pool[category] = eligible
        if not self.pool:
            raise ValueError('The selected split has no categories.')
        self.categories = sorted(self.pool)
        self.a_protocol = manifest.get('a_protocol', 'a_category')
        self.tasks = (deepcopy(manifest.get('a_tasks', {}).get(split, {}))
                      if experiment == 'a' else {})
        self.task_ids = (sorted(self.tasks) if self.tasks else self.categories if experiment == 'a'
                         else sorted(item for ids in self.pool.values() for item in ids))
        if self.tasks:
            if self.a_protocol == 'a_nn' and query_count != 1:
                raise ValueError('A-NN uses one curated target per support set; query_count must be one.')
            for task in self.tasks.values():
                if any(item not in self.pool.get(task['category'], []) for item in task['member_ids']):
                    raise ValueError('Neighborhood members exceed the sampler length eligibility; rebuild the manifest.')
                available = len(task['neighbor_ids']) if self.a_protocol == 'a_nn' else len(task['member_ids'])
                minimum = support_count if self.a_protocol == 'a_nn' else support_count + query_count
                if available < minimum:
                    raise ValueError(f'Neighborhood has only {available} eligible drawings; need {minimum}.')
        self.pool_identifier = _digest(self.pool)
        self.identifier = _digest({
            'manifest': manifest['identifier'], 'experiment': experiment, 'split': split,
            'max_steps': max_steps, 'supports': support_count, 'queries': query_count,
            'b_partition': b_partition, 'max_motion': pen_config.max_motion, 'pool': self.pool,
        })

    @property
    def manifest_identifier(self) -> str:
        return self.manifest['identifier']

    @property
    def reservoir_counts(self) -> dict:
        return {category: len(ids) for category, ids in self.pool.items()}

    def state_dict(self) -> dict:
        return {'identifier': self.identifier,
                'rngs': {name: deepcopy(getattr(self, name + '_rng').bit_generator.state)
                         for name in ('category', 'drawing', 'frame', 'start')},
                'draw_count': self.draw_count}

    def load_state_dict(self, state: dict) -> None:
        if state['identifier'] != self.identifier:
            raise ValueError('Sampler resume configuration or manifest mismatch.')
        for name in ('category', 'drawing', 'frame', 'start'):
            getattr(self, name + '_rng').bit_generator.state = deepcopy(state['rngs'][name])
        self.draw_count = int(state['draw_count'])

    def _frame(self) -> np.ndarray:
        return np.asarray([*self.frame_rng.uniform(-.1, .1, 2), self.frame_rng.uniform(-.5, .5),
                           self.frame_rng.uniform(.6, .9)], np.float32)

    def _execution(
        self, record: SketchRecord, frame: np.ndarray, *, start_xy: np.ndarray | None = None
    ) -> tuple[dict, dict]:
        absolute = apply_frame(record.absolute, frame)
        metadata = {'base_id': record.base_id, 'duplicate_cluster_id': record.duplicate_cluster_id}
        if self.experiment != 'b2':
            return prepare_sequence(absolute, record.incoming_pen, self.max_steps, frame=frame), metadata
        # Public support starts can be fixed for paired wrong-program controls.
        start = (self.start_rng.uniform(-.8, .8, 2).astype(np.float32)
                 if start_xy is None else np.asarray(start_xy, np.float32))
        reference_xy, reference_pen, knots = timed_reference(
            absolute, record.incoming_pen, start, max_motion=self.pen_config.max_motion,
            return_knots=True,
        )
        execution = replay_rollout(reference_xy, reference_pen, start_xy=start, config=self.pen_config)
        commands = execution['tokens'][:-1]
        sequence = prepare_sequence(commands[:, :2], commands[:, 2], self.max_steps,
                                    frame=frame, state=execution['state'], knot_mask=knots)
        metadata.update(start_xy=start.tolist(), reference_absolute=reference_xy.tolist(),
                        reference_incoming_pen=reference_pen.tolist())
        return sequence, metadata

    def build_batch(self, batch_size: int, task_ids: list[str] | None = None) -> dict:
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        if task_ids is not None and len(task_ids) != batch_size:
            raise ValueError('task_ids must name exactly one task per batch row.')
        support_tasks, query_tasks, task_metadata = [], [], []
        for row in range(batch_size):
            task_id = None if task_ids is None else str(task_ids[row])
            local_task = None
            if self.tasks:
                task_id = str(self.category_rng.choice(self.task_ids)) if task_id is None else task_id
                if task_id not in self.tasks:
                    raise ValueError('Requested neighborhood is outside the selected reservoir.')
                local_task = self.tasks[task_id]
                category = local_task['category']
            elif task_id is None:
                category = str(self.category_rng.choice(self.categories))
            elif self.experiment == 'a':
                category = task_id
                if category not in self.pool:
                    raise ValueError('Requested category is outside the selected reservoir.')
            else:
                category = self.store.get(task_id).category
                if task_id not in self.pool.get(category, []):
                    raise ValueError('Requested program is outside the selected reservoir.')
            if local_task is not None and self.a_protocol == 'a_nn':
                neighbors = local_task['neighbor_ids']
                if self.manifest['retrieval']['selection_mode'] == 'sample_top_m':
                    support_ids = self.drawing_rng.choice(neighbors, self.support_count, replace=False).tolist()
                else:
                    support_ids = neighbors[:self.support_count]
                ids = support_ids + [local_task['target_id']]
            elif local_task is not None and self.a_protocol == 'a_category' and local_task.get('target_id'):
                if self.query_count != 1:
                    raise ValueError('Matched category-curation ablation uses one target per support set.')
                candidates = [item for item in local_task['member_ids'] if item != local_task['target_id']]
                ids = self.drawing_rng.choice(candidates, self.support_count, replace=False).tolist() + [local_task['target_id']]
            elif local_task is not None:
                ids = self.drawing_rng.permutation(local_task['member_ids'])[:self.support_count + self.query_count]
            elif self.experiment == 'a':
                ids = self.drawing_rng.permutation(self.pool[category])[:self.support_count + self.query_count]
            else:
                selected = task_id or str(self.drawing_rng.choice(self.pool[category]))
                ids = [selected] * (self.support_count + self.query_count)
            sequences, records = [], []
            for item in ids:
                frame = IDENTITY_FRAME if self.experiment == 'a' else self._frame()
                sequence, metadata = self._execution(self.store.get(str(item)), frame)
                sequences.append(sequence)
                records.append(metadata)
            support_tasks.append(_stack_sequences(sequences[:self.support_count]))
            query_tasks.append(_stack_sequences(sequences[self.support_count:]))
            task_metadata.append({
                'category': category, 'task_id': task_id if local_task is not None else category if self.experiment == 'a' else str(ids[0]),
                'a_protocol': self.a_protocol if self.experiment == 'a' else None,
                'intended_neighborhood_id': (task_id if local_task is not None and
                    (self.a_protocol != 'a_category' or local_task.get('target_id') is not None) else None),
                'anchor_id': local_task['anchor_id'] if local_task is not None else None,
                'split': self.split,
                'eligible_member_ids': local_task['member_ids'] if local_task is not None else [],
                'base_id': None if self.experiment == 'a' else str(ids[0]),
                'intended_category': category, 'intended_base_id': None if self.experiment == 'a' else str(ids[0]),
                'support_ids': [str(item) for item in ids[:self.support_count]],
                'query_ids': [str(item) for item in ids[self.support_count:]],
                'support': records[:self.support_count], 'query': records[self.support_count:],
                'episode_index': self.draw_count,
            })
            self.draw_count += 1
        return {
            'support': _stack_sequences(support_tasks), 'query': _stack_sequences(query_tasks),
            'meta': {'tasks': task_metadata, 'manifest_identifier': self.manifest_identifier,
                     'sampler_identifier': self.identifier, 'experiment': self.experiment,
                     'action_schema': 'bounded_xy_motion_incoming_pen_stop' if self.experiment == 'b2'
                     else 'absolute_xy_incoming_pen_stop', 'b_partition': self.b_partition},
        }


def _stack_sequences(sequences: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.stack([sequence[key] for sequence in sequences]) for key in sequences[0]}


def b1_baselines(record: SketchRecord, support_absolute: np.ndarray,
                 support_frame: np.ndarray, query_frame: np.ndarray,
                 prototype: SketchRecord | None = None) -> dict:
    """Explicit oracle and replay controls, separate from all trainable policies."""
    result = {
        'oracle': apply_frame(record.absolute, query_frame),
        'transformed_replay': transformed_replay(support_absolute, support_frame, query_frame),
        'untransformed_replay': np.asarray(support_absolute, np.float32).copy(),
    }
    if prototype is not None:
        result['category_prototype'] = apply_frame(prototype.absolute, query_frame)
    return result


def category_prototype(store: SketchStore, ids: list[str]) -> SketchRecord:
    """A fixed training-reservoir exemplar nearest median length/stroke count."""
    if not ids:
        raise ValueError('A category prototype requires an explicit reservoir.')
    records = [store.get(item) for item in ids]
    if len({record.category for record in records}) != 1:
        raise ValueError('A category prototype reservoir must contain one category.')
    features = np.asarray([[r.length, np.sum(r.incoming_pen == 0)] for r in records], np.float64)
    median = np.median(features, axis=0)
    scale = np.maximum(np.std(features, axis=0), 1)
    index = min(range(len(records)), key=lambda i: (float(np.sum(((features[i] - median) / scale) ** 2)), records[i].base_id))
    return records[index]


def create_fixture_cache(
    root: str | Path, *, categories: int = 12, drawings_per_category: int = 24, seed: int = 0
) -> SketchStore:
    """Explicit synthetic correctness fixture; never claim these are QuickDraw data."""
    rng = np.random.default_rng(seed)
    records = []
    for category in range(categories):
        for program in range(drawings_per_category):
            length = 8 + program % 8
            time = np.linspace(0, 1, length)
            frequency = 1 + category % 4
            xy = np.stack([time * 1.6 - .8,
                           .4 * np.sin(time * np.pi * frequency + category * .17)], axis=-1)
            xy += rng.normal(0, .08, xy.shape)
            xy *= rng.uniform(.75, 1.0)
            pen = np.ones(length, np.float32)
            pen[0] = 0
            if category % 2:
                pen[length // 2] = 0
            records.append(SketchRecord(
                f'fixture_{category:03d}/{program:05d}', f'fixture_{category:03d}', xy, pen,
                provenance={'synthetic_fixture': True},
            ))
    return SketchStore.write(root, records, provenance={
        'kind': 'synthetic_correctness_fixture', 'seed': seed,
        'categories': categories, 'drawings_per_category': drawings_per_category,
    })
