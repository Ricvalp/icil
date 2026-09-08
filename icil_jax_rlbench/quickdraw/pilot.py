"""An explicitly launched, bounded matched A comparison."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from icil_jax_rlbench.quickdraw.data import SketchStore, build_manifest, save_manifest, validate_manifest
from icil_jax_rlbench.quickdraw.evaluate import prepare_evaluation, generate_run, export_references, load_evaluation
from icil_jax_rlbench.quickdraw.train import train


def run_category_pilot(cache_root, output, *, levels=(8,128), steps, unique_budget,
              max_steps=128, seed=0, subset_seed=0, config=None):
    if len(levels) != 2 or min(levels) < 1 or levels[0] >= levels[1]:
        raise ValueError('The bounded pilot takes exactly two increasing F levels.')
    if steps < 1 or steps > 1000:
        raise ValueError('Pilot steps must be 1..1000; launch longer runs explicitly with train.')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    store = SketchStore.open(cache_root)
    runs = []
    for level in levels:
        manifest = build_manifest(store,seed=0,subset_seed=subset_seed,
            family_count=level,unique_budget=unique_budget,
            reference_per_category=8,max_points=max_steps-1)
        manifest_path = output/f'F{level}_manifest.json'
        save_manifest(manifest_path,manifest)
        base = copy.deepcopy(config or {})
        base.update(cache_root=str(Path(cache_root).resolve()),manifest_path=str(manifest_path.resolve()),
                    experiment='a',seed=seed,num_steps=steps,output_dir=str(output/f'F{level}'/'training'))
        base['model'] = {**base.get('model',{}),'max_steps':max_steps}
        support_count = int(base.get('support_count',4))
        query_count = int(base.get('query_count',1))
        episodes = prepare_evaluation(cache_root,manifest_path,output/f'F{level}'/'episodes',
            tasks=2*len(manifest['categories']['development']),
            support_count=support_count,query_count=query_count,max_steps=max_steps,seed=61001)
        for half in ('real_a','real_b'):
            export_references(cache_root,manifest_path,output/f'F{level}'/half,
                              half=half,max_steps=max_steps)
        for mode in ('ttt_kvb_full','explicit_context','no_support'):
            cfg = {**base,'model_type':mode}
            run_dir = train(cfg)
            conditions = ('no_update','correct_support','wrong_support','random_update_matched_norm','support_copy') if mode.startswith('ttt_') else (
                ('no_update','correct_support','wrong_support') if mode=='explicit_context' else ('correct_support',))
            generated = generate_run(run_dir/'last.pkl',episodes,output/f'F{level}'/mode,
                conditions=conditions,samples_per_task=2,seed=71001)
            runs.append({'family_count':level,'model_type':mode,'checkpoint':str(run_dir/'last.pkl'),
                         'generated':str(generated),'manifest_id':manifest['identifier'],
                         'unique_drawings':manifest['budget']['unique_a_train_drawings'],
                         'steps':steps,'model_seed':seed,'subset_seed':subset_seed})
    report = {'schema_version':1,'runs':runs,'data_provenance':store.provenance,
              'metric_status':'Exported; score with original frozen extractor in separate environment.',
              'scientific_threshold_claim':False,'untouched_test_evaluated':False}
    (output/'pilot.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return output


def _category_ablation_manifest(store, original, embeddings):
    """Keep A-NN targets/references fixed while sampling independent supports."""
    manifest = copy.deepcopy(original)
    manifest['a_protocol'] = 'a_category'
    manifest['ablation_parent_manifest'] = original['identifier']
    manifest['ablation_parent_diagnostics'] = copy.deepcopy(original['construction_diagnostics'])
    manifest['retrieval']['selection_mode'] = 'random_same_category'
    from .neighborhoods import _overlap, _task_statistics
    rows = {record['base_id']:index for index, record in enumerate(embeddings.records)}
    for split, tasks in manifest['a_tasks'].items():
        statistics = {category:_task_statistics(store, ids, np.asarray(
            embeddings.cosine[[rows[item] for item in ids]]))
            for category, ids in manifest['a_ids'][split].items()}
        for task in tasks.values():
            target = task['target_id']
            members = list(manifest['a_ids'][split][task['category']])
            task['member_ids'] = members
            task['neighbor_ids'] = [item for item in members if item != target]
            task['neighbor_scores'] = []
            task['duplicate_clusters'] = {item:store.get(item).duplicate_cluster_id for item in members}
            task['intended_target_neighborhood_statistics'] = {
                key:copy.deepcopy(task[key]) for key in statistics[task['category']]}
            task.update(copy.deepcopy(statistics[task['category']]))
        manifest['construction_diagnostics']['overlap'][split] = _overlap(tasks)
    manifest['budget']['regime'] = 'matched_A_NN_targets_random_same_category_supports'
    manifest['budget']['N_also_limits_training_targets'] = True
    manifest['identifier'] = hashlib.sha256(json.dumps(
        {key:value for key,value in manifest.items() if key != 'identifier'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    validate_manifest(store, manifest)
    return manifest


def run_pilot(cache_root, output, *, levels=None, steps, unique_budget=None,
              max_steps=128, seed=0, subset_seed=0, config=None,
              a_protocol='a_local', embedding_root=None, family_count=8,
              split_regime='familiar_drawings', neighborhood_size=64,
              evaluation_neighborhood_count=16, selection_mode='exact_top_k',
              top_m=32, region_count=8, wrong_neighborhood_min_distance=.05,
              wrong_neighborhood_max_jaccard=.1, budget_regime='fixed_reservoir'):
    """Bounded A-NN/comparator or fixed-category A-local diversity comparison."""
    if budget_regime not in ('fixed_reservoir','per_neighborhood'):
        raise ValueError('Unknown pilot budget regime.')
    if budget_regime == 'fixed_reservoir' and (unique_budget is None or unique_budget < 1):
        raise ValueError('The fixed-reservoir pilot requires an explicit positive unique_budget.')
    if budget_regime == 'per_neighborhood' and (a_protocol != 'a_local' or unique_budget is not None):
        raise ValueError('Per-neighborhood allowance requires A-local and no fixed unique_budget.')
    if a_protocol == 'a_category':
        return run_category_pilot(cache_root, output, levels=levels or (8,128),
                                  steps=steps, unique_budget=unique_budget,
                                  max_steps=max_steps, seed=seed, subset_seed=subset_seed, config=config)
    if a_protocol not in ('a_nn', 'a_local') or not embedding_root:
        raise ValueError('Neighborhood pilots require a_nn/a_local and a frozen embedding_root.')
    levels = list(levels or ([32,256] if a_protocol == 'a_local' else [32]))
    if (any(level < 1 for level in levels) or not 1 <= steps <= 1000
            or (a_protocol == 'a_local' and (len(levels) != 2 or levels[0] >= levels[1]))
            or (a_protocol == 'a_nn' and len(levels) != 1)):
        raise ValueError('A-local needs two increasing N levels; A-NN one target budget; steps must be 1..1000.')
    from .embeddings import EmbeddingStore
    from .neighborhoods import build_neighborhood_manifest

    store, embeddings = SketchStore.open(cache_root), EmbeddingStore.open(embedding_root)
    manifests = []
    for level in levels:
        manifest = build_neighborhood_manifest(
            store, embeddings, a_protocol=a_protocol, split_regime=split_regime,
            family_count=family_count, neighborhood_count=level,
            evaluation_neighborhood_count=evaluation_neighborhood_count,
            unique_budget=unique_budget, neighborhood_size=neighborhood_size,
            max_points=max_steps-1, seed=0, subset_seed=subset_seed,
            selection_mode=selection_mode, top_m=top_m, region_count=region_count,
            budget_regime=budget_regime,
        )
        manifests.append((f'{a_protocol}_N{level}', manifest))
    if a_protocol == 'a_nn':
        manifests.append(('a_category_matched_targets', _category_ablation_manifest(store, manifests[0][1], embeddings)))
    else:
        first, last = manifests[0][1], manifests[-1][1]
        for split in ('development', 'test'):
            if first['a_tasks'][split] != last['a_tasks'][split]:
                raise ValueError('N levels changed held-out neighborhoods.')
        if budget_regime == 'fixed_reservoir' and first['a_ids']['train'] != last['a_ids']['train']:
            raise ValueError('N levels changed the fixed available drawing reservoir.')
        if first['budget']['fixed_training_categories'] != last['budget']['fixed_training_categories']:
            raise ValueError('N levels changed the coarse training categories.')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared, first_arrays = [], None
    for label, manifest in manifests:
        manifest_path = output / f'{label}_manifest.json'
        save_manifest(manifest_path, manifest)
        base = copy.deepcopy(config or {})
        base.update(cache_root=str(Path(cache_root).resolve()), manifest_path=str(manifest_path),
                    experiment='a', a_protocol=manifest['a_protocol'], seed=seed, num_steps=steps,
                    output_dir=str(output / label / 'training'))
        base['model'] = {**base.get('model', {}), 'max_steps':max_steps}
        episodes = prepare_evaluation(
            cache_root, manifest_path, output / label / 'episodes',
            tasks=2*len(manifest['a_tasks']['development']),
            support_count=int(base.get('support_count',4)), query_count=int(base.get('query_count',1)),
            max_steps=max_steps, seed=61001,
            wrong_neighborhood_min_distance=wrong_neighborhood_min_distance,
            wrong_neighborhood_max_jaccard=wrong_neighborhood_max_jaccard,
        )
        arrays, _ = load_evaluation(episodes)
        if first_arrays is None:
            first_arrays = arrays
        else:
            roles = arrays if a_protocol == 'a_local' else ('query',)
            for role in roles:
                for field, value in arrays[role].items():
                    if not np.array_equal(value, first_arrays[role][field]):
                        raise ValueError(f'Pilot changed paired {role}/{field} arrays across conditions.')
        for half in ('real_a', 'real_b'):
            export_references(cache_root, manifest_path, output / label / half,
                              half=half, max_steps=max_steps, episodes=episodes)
        prepared.append((label, manifest, base, episodes))
    runs = []
    for label, manifest, base, episodes in prepared:
        for mode in ('ttt_kvb_full', 'explicit_context', 'no_support'):
            run_dir = train({**base, 'model_type':mode})
            conditions = ['correct_support']
            if mode != 'no_support':
                conditions = ['no_update','correct_support','wrong_support','support_copy',
                              'nearest_training_example']
                if manifest['a_protocol'] != 'a_category':
                    conditions.append('same_category_wrong_neighborhood')
                if mode.startswith('ttt_'):
                    conditions.append('random_update_matched_norm')
            generated = generate_run(run_dir/'last.pkl', episodes, output/label/mode,
                                     conditions=conditions, samples_per_task=2, seed=71001)
            runtime = json.loads((run_dir/'runtime.json').read_text())
            runs.append({'a_protocol':manifest['a_protocol'], 'family_count':family_count,
                         'neighborhood_count':manifest['budget']['neighborhood_count'],
                         'model_type':mode, 'checkpoint':str(run_dir/'last.pkl'),
                         'generated':str(generated), 'episodes':str(episodes),
                         'references':{half:str(output/label/half) for half in ('real_a','real_b')},
                         'manifest_id':manifest['identifier'], 'steps':steps,
                         'model_seed':seed, 'subset_seed':subset_seed,
                         'exposure':runtime['exposure'], 'available_budget':manifest['budget']})
    report = {'schema_version':2, 'runs':runs, 'data_provenance':store.provenance,
              'primary_axis':'neighborhood_count_at_fixed_coarse_categories' if a_protocol == 'a_local'
              else 'A_NN_versus_A_category_with_matched_targets',
              'metric_status':'Score frozen generated/reference artifacts with the declared raw-feature evaluator.',
              'budget_regime':budget_regime,
              'exposure_matching':('Optimizer steps and available reservoir matched; realized token/ID exposure reported.'
                                   if budget_regime == 'fixed_reservoir' else
                                   'Optimizer steps and per-neighborhood membership allowance matched; total eligible data grows with N and overlap.'),
              'scientific_threshold_claim':False, 'untouched_test_evaluated':False}
    (output/'pilot.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return output
