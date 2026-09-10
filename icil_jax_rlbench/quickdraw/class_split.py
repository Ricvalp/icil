"""Deterministic policy class holdouts over the existing drawing splits.

The heldout classes supply development drawings for validation and separate
test drawings for later evaluation. Category metadata never enters the policy.
"""

from functools import lru_cache
import hashlib
import json

import numpy as np


DEFAULT_HOLDOUT_SEED = 37


def validate_holdout_config(cfg):
    count = cfg.get('heldout_category_count', 0)
    seed = cfg.get('heldout_category_seed', DEFAULT_HOLDOUT_SEED)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('heldout_category_count must be a nonnegative integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError('heldout_category_seed must be an unsigned 32-bit integer')
    return count, seed


@lru_cache(maxsize=64)
def _heldout_ids(categories, count, seed):
    if not categories or len(set(categories)) != len(categories):
        raise ValueError('Class holdout requires distinct, nonempty dataset categories')
    if count >= len(categories):
        raise ValueError('heldout_category_count must leave at least one training category')
    # Rank names, not numeric IDs: the selected names survive category reordering
    # and do not depend on model choice, training seed, or NumPy RNG versions.
    ranked = sorted(categories, key=lambda name: (
        hashlib.sha256(f'quickdraw-policy-class-holdout-v1:{seed}:{name}'.encode()).digest(), name))
    selected = set(ranked[:count])
    return tuple(i for i, name in enumerate(categories) if name in selected)


def resolve_class_split(dataset, cfg):
    count, seed = validate_holdout_config(cfg)
    categories = tuple(dataset.categories)
    heldout = list(_heldout_ids(categories, count, seed))
    heldout_set = set(heldout)
    training = [i for i in range(len(categories)) if i not in heldout_set]
    evaluation = heldout if count else list(range(len(categories)))
    result = {
        'version': 1, 'dataset_id': dataset.identifier,
        'heldout_category_count': count, 'heldout_category_seed': seed,
        'selection_rule': 'sha256_seed_and_category_name_v1',
        'categories': list(categories), 'training_category_ids': training,
        'heldout_category_ids': heldout, 'evaluation_category_ids': evaluation,
        'training_categories': [categories[i] for i in training],
        'heldout_categories': [categories[i] for i in heldout],
        'validation_drawing_split': 'development', 'test_drawing_split': 'test',
        'validation_and_test_share_categories': True,
        'scope': 'policy_optimization_only; frozen_retrieval_and_metric_resources_unchanged',
    }
    result['identifier'] = hashlib.sha256(json.dumps(
        result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return result


def evaluation_category_ids(dataset, cfg, category_scope='auto'):
    split = resolve_class_split(dataset, cfg)
    if category_scope == 'auto':
        return split['evaluation_category_ids']
    if category_scope == 'all':
        return list(range(len(dataset.categories)))
    if category_scope == 'seen':
        return split['training_category_ids']
    if category_scope == 'heldout':
        if not split['heldout_category_ids']:
            raise ValueError('This checkpoint has no heldout categories')
        return split['heldout_category_ids']
    raise ValueError('category_scope must be auto, heldout, seen, or all')


def filter_rows(dataset, rows, category_ids):
    ids = np.asarray(category_ids)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
            or len(np.unique(ids)) != len(ids) or np.any(ids < 0)
            or np.any(ids >= len(dataset.categories))):
        raise ValueError('Category IDs must be a nonempty, unique, valid integer list')
    rows = np.asarray(rows)
    if len(ids) == len(dataset.categories):
        return rows
    return rows[np.isin(dataset.category_ids[rows], ids)]


def split_rows(dataset, cfg, split, *, reference=False):
    specification = resolve_class_split(dataset, cfg)
    ids = (specification['training_category_ids'] if split == 'train'
           else specification['evaluation_category_ids'])
    rows = dataset.reference_rows(split) if reference else dataset.rows(split)
    return filter_rows(dataset, rows, ids)


def validate_saved_class_split(saved, dataset, cfg):
    expected = resolve_class_split(dataset, cfg)
    if saved is None:
        if cfg.get('heldout_category_count', 0):
            raise ValueError('A class-holdout checkpoint must contain its frozen class split')
    elif saved != expected:
        raise ValueError('Checkpoint class split differs from the dataset or configured class holdout')
    return expected
