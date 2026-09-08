"""Aggregate frozen outputs at category/base-program level, retaining failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import paired_bootstrap
from .metrics import file_sha256, load_features, load_trajectories, score_feature_artifacts


def _paired_signature(records):
    return [(row['task_index'], row['sample_index'], row['query_index'],
             row['intended_category'], row.get('intended_base_id'),
             row.get('intended_neighborhood_id'), row.get('a_protocol'), row.get('query_ids'),
             row['query_seed'], row['frame']) for row in records]


def paired_neighborhood_summary(left_scores, right_scores, records, *, seed=0, draws=2000):
    """Equal task effects, bootstrapped in connected overlapping-task blocks."""
    if set(left_scores) != set(right_scores):
        raise ValueError('Conditional scores require identical intended neighborhoods.')
    tasks = sorted(left_scores)
    members = {task: set() for task in tasks}
    for row in records:
        task = row.get('intended_neighborhood_id')
        if task in members:
            members[task].update(row.get('eligible_member_ids', row.get('member_ids', [])))
            members[task].update(row.get('support_ids', []))
            members[task].update(row.get('query_ids', []))
            refs = row.get('reference_ids', {})
            if isinstance(refs, dict):
                members[task].update(item for ids in refs.values() for item in ids)
    parent = list(range(len(tasks)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    owners = {}
    for index, task in enumerate(tasks):
        for item in members[task]:
            if item in owners:
                parent[find(index)] = find(owners[item])
            else:
                owners[item] = index
    blocks = {}
    for index in range(len(tasks)):
        blocks.setdefault(find(index), []).append(index)
    effects = np.asarray([left_scores[task]['sketch_mmd'] - right_scores[task]['sketch_mmd'] for task in tasks])
    blocks = list(blocks.values())
    ci = None
    if len(blocks) > 1:
        rng = np.random.default_rng(seed)
        estimates = [float(effects[np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])].mean())
                     for _ in range(draws)]
        ci = np.quantile(estimates, [.025, .975]).tolist()
    return {
        'mean': float(effects.mean()), 'ci95': ci, 'unit': 'neighborhood_overlap_component',
        'metric': 'macro_neighborhood_sketch_mmd', 'neighborhood_count': len(tasks),
        'independent_overlap_components': len(blocks), 'bootstrap_seed': int(seed),
        'point_weighting': 'uniform across intended neighborhoods',
        'uncertainty': 'connected shared member/reference blocks; a single block provides no interval',
    }


def paired_control_summary(left, right, *, experiment, seed=0):
    """Positive differences favor right; generated samples are not replicates.

    For B, collapse all repeats of each base program before resampling. Missing
    trajectories are retained as failure differences; geometry is explicitly
    conditional on pairs with finite scores in both conditions.
    """
    if _paired_signature(left) != _paired_signature(right):
        raise ValueError('Support controls require identical ordered tasks and query keys.')
    groups = {}
    for a, b in zip(left, right):
        key = str(a['intended_base_id'])
        group = groups.setdefault(key, {'error': [], 'failure': []})
        if experiment == 'b2':
            def values(record):
                metric = record['timed_tracking']
                failed = (not metric['complete_output'] or metric['no_stop_failure']
                          or metric['invalid_pair_count'] > 0)
                return metric['raw_timestep_xy_rmse'], failed
        elif experiment == 'b1':
            def values(record):
                metric = record['ordered_tracking']
                failed = (not metric['complete_output'] or metric['no_stop_failure']
                          or metric['invalid_pair_count'] > 0 or metric['extra_step_count'] > 0)
                return metric['raw_timestep_xy_rmse'], failed
        else:
            raise ValueError('Trajectory control aggregation is defined for b1/b2.')
        x, bad_x = values(a)
        y, bad_y = values(b)
        group['failure'].append(float(bad_x)-float(bad_y))
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            group['error'].append(float(x)-float(y))
    errors = [np.mean(g['error']) for g in groups.values() if g['error']]
    failures = [np.mean(g['failure']) for g in groups.values()]
    return {
        'error_difference': paired_bootstrap(errors, seed=seed) if errors else None,
        'failure_fraction_difference': paired_bootstrap(failures, seed=seed),
        'unit': 'base_program', 'programs': len(groups),
        'programs_with_paired_evaluable_error': len(errors),
        'paired_evaluable_samples': sum(len(g['error']) for g in groups.values()),
        'requested_samples': len(left),
        'error_metric': 'raw_timestep_xy_rmse' if experiment == 'b2' else 'waypoint_index_xy_rmse',
        'error_population': 'finite paired errors; assess alongside failures and raw per-condition counts',
    }


def aggregate_runs(specification, output, *, seed=0):
    """A JSON list names generated directories and optional frozen features.

    Each item can record family_count/program_count/model_seed/subset_seed and
    capacity. Feature paths are keyed by condition; reference repeats are fixed
    independent sets. This performs no generation, training or model selection.
    """
    if isinstance(specification, (str, Path)):
        specification = json.loads(Path(specification).read_text())
    if not isinstance(specification, list) or not specification:
        raise ValueError('Provide a nonempty list of run specifications.')
    rows = []
    for spec in specification:
        root = Path(spec['generated'])
        summary = json.loads((root/'summary.json').read_text())
        conditions = summary['conditions']
        metadata = {c:load_trajectories(root/c)[1] for c in conditions}
        first = next(iter(metadata.values()))
        for condition, value in metadata.items():
            for key in ('checkpoint_id','evaluation_id','manifest_id','model_type','experiment','pen_config','perturbations','a_protocol'):
                if value.get(key) != first.get(key):
                    raise ValueError(f'Control {condition} provenance differs: {key}')
            for key in ('checkpoint_id','evaluation_id'):
                if value.get(key) != summary.get(key):
                    raise ValueError(f'Control {condition} does not match the run summary: {key}')
        row = {key:value for key,value in spec.items() if key != 'features'}
        row.update(model_type=first['model_type'],experiment=first['experiment'],
                   checkpoint_id=first['checkpoint_id'],evaluation_id=first['evaluation_id'],
                   output_statistics=conditions,controls={})
        features = spec.get('features',{})
        scores = {}
        if features:
            if set(features) != set(conditions):
                raise ValueError('Features must cover every exported control, including negative controls.')
            for condition,path in features.items():
                _, feature_metadata = load_features(path)
                if (feature_metadata.get('artifact_sha256') != file_sha256(root/condition/'trajectories.npz')
                        or feature_metadata.get('artifact_metadata_sha256') != file_sha256(root/condition/'metadata.json')):
                    raise ValueError('Feature outputs must correspond exactly to the declared control artifacts.')
                scores[condition] = score_feature_artifacts(path,spec['reference'],
                    reference_repeats=spec.get('reference_repeats',()))
            row['feature_scores'] = scores
        for name, control in (('support_gain','no_update'),('support_specificity','wrong_support'),
                              ('within_category_support_specificity','same_category_wrong_neighborhood')):
            if control not in metadata or 'correct_support' not in metadata:
                continue
            left,right = metadata[control]['records'],metadata['correct_support']['records']
            if _paired_signature(left) != _paired_signature(right):
                raise ValueError('Control outputs are not paired.')
            if first['experiment'] != 'a':
                row['controls'][name] = paired_control_summary(left,right,
                    experiment=first['experiment'],seed=seed)
            elif scores and scores[control].get('neighborhood_conditional'):
                row['controls'][name] = paired_neighborhood_summary(
                    scores[control]['neighborhood_conditional']['per_neighborhood'],
                    scores['correct_support']['neighborhood_conditional']['per_neighborhood'], right, seed=seed)
            elif scores:
                a = scores[control]['conditional']['per_category']
                b = scores['correct_support']['conditional']['per_category']
                if set(a) != set(b):
                    raise ValueError('Conditional scores require identical categories.')
                row['controls'][name] = {
                    **paired_bootstrap([a[c]['sketch_mmd']-b[c]['sketch_mmd'] for c in sorted(a)],seed=seed),
                    'unit':'category','metric':'macro_sketch_mmd',
                }
            else:
                row['controls'][name] = {'status':'pending_original_extractor_features'}
            if first['experiment'] == 'a' and all('conditional_target_loss' in item for item in left + right):
                losses = {}
                for a, b in zip(left, right):
                    task = a.get('intended_neighborhood_id') or a['intended_category']
                    losses.setdefault(str(task), {}).setdefault(a['task_index'],
                        a['conditional_target_loss']['loss'] - b['conditional_target_loss']['loss'])
                # Generation repeats contain the same deterministic target loss;
                # reduce them before equal-task paired uncertainty.
                values = [float(np.mean(list(items.values()))) for items in losses.values()]
                row['controls'][name]['conditional_target_loss'] = {
                    'mean': float(np.mean(values)), 'intended_task_count': len(values),
                    'per_task_difference': dict(zip(losses, values)),
                    'objective': 'paired autoregressive mixture NLL plus pen/STOP BCE',
                    'uncertainty': 'descriptive task means; repetitions and overlapping local tasks are not independent replicates',
                }
        rows.append(row)
    report = {
        'schema_version':1,'runs':rows,'positive_control_difference':'favors correct support',
        'uncertainty_scope':'Within-checkpoint neighborhood-overlap/category/base-program units only; one model/subset seed is not independent training replication.',
        'scientific_threshold_claim':False,'checkpoint_selection_performed':False,
    }
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=0)
    print(aggregate_runs(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
