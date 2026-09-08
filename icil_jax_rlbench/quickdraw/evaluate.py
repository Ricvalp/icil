"""Immutable evaluation episodes and generation from frozen adapted states."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from icil_jax_rlbench.models.fast_weight_ttt import initial_fast_state
from icil_jax_rlbench.quickdraw.data import (
    SketchSampler, SketchStore, category_prototype, load_manifest, prepare_sequence,
)
from icil_jax_rlbench.quickdraw.metrics import (
    RENDERER_CONFIG, file_sha256, paired_geometry, save_trajectories, trajectory_statistics,
    timed_tracking_metrics, geometry_descriptor,
)
from icil_jax_rlbench.quickdraw.model import (
    SketchModelConfig, adapt_sketch, encode_context, generate,
    initial_query_carry, query_step, sample_token, query_loss,
)
from icil_jax_rlbench.quickdraw.pen import (
    PenConfig, apply_frame, replay_rollout, rollout_policy, timed_reference,
    transformed_replay,
)
from icil_jax_rlbench.quickdraw.train import load_run


CONDITIONS = (
    'no_update', 'correct_support', 'wrong_support',
    'random_update_matched_norm', 'support_reordering', 'stroke_reversal',
    'action_corruption', 'support_copy', 'transformed_replay',
    'untransformed_replay', 'category_prototype', 'oracle',
    'open_loop_replay', 'feedback_replay',
    'same_category_wrong_neighborhood', 'nearest_training_example',
)


def _dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def _complexity(store, ids):
    return np.mean([[store.get(item).length,
                     np.sum(store.get(item).incoming_pen == 0)] for item in ids], axis=0)


def select_wrong_neighborhood(tasks, intended_id, *, min_cosine_distance=0.05,
                              max_member_jaccard=0.1, exclude_member_ids=()):
    """Separate shapes first, then approximately match length/stroke complexity."""
    target = tasks[intended_id]
    if not 0 <= min_cosine_distance <= 2 or not 0 <= max_member_jaccard <= 1:
        raise ValueError('Wrong-neighborhood distance/Jaccard thresholds must be in [0,2]/[0,1].')
    excluded = set(exclude_member_ids)
    center = np.asarray(target['centroid'], np.float64)
    members = set(target['member_ids'])
    target_complexity = np.asarray([target['mean_length'], target['mean_strokes']])
    eligible = []
    for task_id, candidate in tasks.items():
        if task_id == intended_id or candidate['category'] != target['category']:
            continue
        other = set(candidate['member_ids'])
        if other & excluded:
            continue
        overlap = len(members & other) / max(len(members | other), 1)
        distance = float(np.clip(1.0 - np.dot(center, candidate['centroid']), 0.0, 2.0))
        if distance < min_cosine_distance or overlap > max_member_jaccard:
            continue
        complexity = np.asarray([candidate['mean_length'], candidate['mean_strokes']])
        mismatch = float(np.linalg.norm((complexity - target_complexity) / np.maximum(target_complexity, 1)))
        eligible.append((mismatch, -distance, str(task_id), overlap))
    if not eligible:
        raise ValueError(f'No separated same-category wrong neighborhood for {intended_id!r}; '
                         'increase the eligible evaluation task pool or explicitly revise the separation rule.')
    mismatch, negative_distance, source, overlap = min(eligible)
    return source, {
        'source_neighborhood_id': source, 'intended_neighborhood_id': str(intended_id),
        'centroid_cosine_distance': -negative_distance, 'member_jaccard': overlap,
        'relative_length_stroke_mismatch': mismatch,
        'minimum_centroid_cosine_distance': float(min_cosine_distance),
        'maximum_member_jaccard': float(max_member_jaccard),
        'excluded_intended_query_ids': sorted(excluded),
        'selection_rule': 'same category; separation thresholds; minimum relative length/stroke L2; distance then ID ties',
    }


def prepare_evaluation(cache_root, manifest_path, output, *, experiment='a',
                       split='development', tasks=8, support_count=4,
                       query_count=1, max_steps=128, seed=0,
                       b_partition='familiar', allow_test=False, motion_bound=0.1,
                       wrong_neighborhood_min_distance=0.05, wrong_neighborhood_max_jaccard=0.1):
    if split == 'test' and not allow_test:
        raise ValueError('Untouched test episodes require explicit --allow-test.')
    if tasks < 1:
        raise ValueError('tasks must be positive.')
    store = SketchStore.open(cache_root)
    manifest = load_manifest(manifest_path, store)
    kwargs = dict(experiment=experiment, split=split, max_steps=max_steps,
                  support_count=support_count, query_count=query_count, b_partition=b_partition,
                  pen_config=PenConfig(max_motion=motion_bound))
    sampler = SketchSampler(store, manifest, seed=seed, **kwargs)
    wrong_sampler = SketchSampler(store, manifest, seed=seed + 8101, **kwargs)
    # Balanced task/category schedule is fixed before inspecting any outputs.
    task_ids = getattr(sampler, 'task_ids', sampler.categories) if experiment == 'a' else [
        item for category in sampler.categories for item in sampler.pool[category]
    ]
    schedule = [task_ids[i % len(task_ids)] for i in range(tasks)]
    if experiment == 'a' and getattr(sampler, 'tasks', {}):
        by_category = [[item for item in task_ids if sampler.tasks[item]['category'] == category]
                       for category in sampler.categories]
        schedule = [by_category[i % len(by_category)][(i // len(by_category)) % len(by_category[i % len(by_category)])]
                    for i in range(tasks)]
    if experiment != 'a':
        by_category = [sampler.pool[category] for category in sampler.categories]
        schedule = [by_category[i % len(by_category)][(i // len(by_category)) % len(by_category[i % len(by_category)])]
                    for i in range(tasks)]
    batch = sampler.build_batch(tasks, task_ids=schedule)
    wrongs, wrong_metadata, local_wrongs, local_wrong_metadata, local_pairings = [], [], [], [], []
    all_a_tasks = getattr(sampler, 'tasks', {}) if experiment == 'a' else {}
    neighborhood_tasks = all_a_tasks if manifest.get('a_protocol') in ('a_nn', 'a_local') else {}
    for task_index, record in enumerate(batch['meta']['tasks']):
        category = record['intended_category']
        target = _complexity(store, record['support_ids'])
        if experiment == 'a':
            candidates = [c for c in sampler.categories if c != category]
            if not candidates:
                raise ValueError('Wrong-category controls require at least two categories.')
            source_category = min(candidates, key=lambda c: (
                float(np.linalg.norm((_complexity(store, sampler.pool[c]) - target) / [max(target[0],1), max(target[1],1)])), c))
            source = (next(item for item in task_ids if all_a_tasks[item]['category'] == source_category)
                      if all_a_tasks else source_category)
        else:
            candidates = [item for item in sampler.pool[category] if item != record['intended_base_id']]
            if not candidates:
                raise ValueError('Same-category wrong-instance control needs a second program.')
            source = min(candidates, key=lambda item: (
                float(np.linalg.norm((_complexity(store, [item]) - target) / [max(target[0],1), max(target[1],1)])), item))
        wrong = wrong_sampler.build_batch(1, task_ids=[source])
        if experiment != 'a':
            # The program alone changes. Pool eligibility was checked against
            # every allowed frame/start before selecting the wrong program.
            for demo in range(support_count):
                frame = batch['support']['frame'][task_index, demo]
                start = batch['support']['state'][task_index, demo, 0, :2] if experiment == 'b2' else None
                paired, paired_metadata = wrong_sampler._execution(store.get(source), frame, start_xy=start)
                for name, value in paired.items():
                    wrong['support'][name][0, demo] = value
                wrong['meta']['tasks'][0]['support'][demo] = paired_metadata
        wrongs.append({name: value[0] for name, value in wrong['support'].items()})
        wrong_metadata.append(wrong['meta']['tasks'][0])
        if neighborhood_tasks:
            intended = record['intended_neighborhood_id']
            local_source, pairing = select_wrong_neighborhood(neighborhood_tasks, intended,
                min_cosine_distance=wrong_neighborhood_min_distance,
                max_member_jaccard=wrong_neighborhood_max_jaccard,
                exclude_member_ids=record['query_ids'])
            local_wrong = wrong_sampler.build_batch(1, task_ids=[local_source])
            local_wrongs.append({name: value[0] for name, value in local_wrong['support'].items()})
            local_wrong_metadata.append(local_wrong['meta']['tasks'][0])
            local_pairings.append(pairing)
    arrays = {f'{role}.{name}': value for role in ('support', 'query')
              for name, value in batch[role].items()}
    arrays.update({f'wrong_support.{name}': np.stack([item[name] for item in wrongs])
                   for name in wrongs[0]})
    if local_wrongs:
        arrays.update({f'same_category_wrong_neighborhood.{name}': np.stack([item[name] for item in local_wrongs])
                       for name in local_wrongs[0]})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'episodes.npz', **arrays)
    metadata = {
        'schema_version': 1, 'experiment': experiment, 'split': split,
        'b_partition': b_partition, 'cache_id': store.identifier,
        'manifest_id': manifest['identifier'], 'seed': seed, 'max_steps': max_steps,
        'motion_bound': float(motion_bound),
        'support_count': support_count, 'query_count': query_count,
        'records': batch['meta']['tasks'], 'wrong_records': wrong_metadata,
        'a_protocol': manifest.get('a_protocol', 'a_category') if experiment == 'a' else None,
        'same_category_wrong_records': local_wrong_metadata,
        'same_category_wrong_pairings': local_pairings,
        'data_sha256': file_sha256(output / 'episodes.npz'),
        'sampling': 'balanced fixed task schedule; protocol-declared offline curation; metadata excluded from policy inputs',
        'wrong_support_pairing': 'category replacement' if experiment == 'a'
            else 'different same-category program with original support frames and starts',
    }
    metadata['identifier'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    _dump(output / 'metadata.json', metadata)
    return output


def load_evaluation(path):
    path = Path(path)
    metadata = json.loads((path / 'metadata.json').read_text())
    identifier = metadata.pop('identifier')
    if hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest() != identifier:
        raise ValueError('Evaluation manifest has changed.')
    metadata['identifier'] = identifier
    if file_sha256(path / 'episodes.npz') != metadata['data_sha256']:
        raise ValueError('Evaluation episode data has changed.')
    with np.load(path / 'episodes.npz', allow_pickle=False) as archive:
        sections = {role: {key.split('.',1)[1]: archive[key] for key in archive.files
                           if key.startswith(role + '.')}
                    for role in ('support','query','wrong_support','same_category_wrong_neighborhood')}
    return sections, metadata


def export_references(cache_root, manifest_path, output, *, split='development',
                      half='real_a', max_steps=128, allow_test=False, episodes=None):
    if split == 'test' and not allow_test:
        raise ValueError('Test references require explicit --allow-test.')
    if half not in ('real_a','real_b'):
        raise ValueError('Reference half must be real_a or real_b.')
    store = SketchStore.open(cache_root)
    manifest = load_manifest(manifest_path,store)
    outputs,records = [],[]
    task_specs = manifest.get('a_tasks', {}).get(split, {})
    if task_specs:
        task_ids = sorted(task_specs)
        if episodes is not None:
            _, evaluation = load_evaluation(episodes)
            if evaluation['manifest_id'] != manifest['identifier'] or evaluation['split'] != split:
                raise ValueError('Reference selection and frozen evaluation protocol differ.')
            task_ids = sorted({row.get('intended_neighborhood_id') or row['task_id'] for row in evaluation['records']})
        populations = [(task_specs[item]['category'], item, task_specs[item]['reference_ids'][half])
                       for item in task_ids]
    else:
        populations = [(category, None, manifest['reference_ids'][category][half])
                       for category in manifest['categories'][split]]
    for category, task_id, ids in populations:
        for item in ids:
            drawing = store.get(item)
            sequence = prepare_sequence(drawing.absolute,drawing.incoming_pen,max_steps)
            outputs.append(sequence)
            records.append({'intended_category':category,'base_id':item,'drawing_id':item,
                            'support_ids':[],'condition':'independent_real_reference',
                            'duplicate_cluster_id':drawing.duplicate_cluster_id})
            if task_id is not None and (manifest['a_protocol'] != 'a_category'
                                       or task_specs[task_id].get('target_id') is not None):
                records[-1].update(intended_neighborhood_id=task_id,
                                   a_protocol=manifest['a_protocol'],
                                   synthetic_fixture=bool(drawing.provenance.get('synthetic_fixture', False)))
                records[-1]['local_reference_role'] = task_specs[task_id].get('reference_role')
    arrays = {key:np.stack([row[key] for row in outputs])
              for key in ('tokens','event_mask','point_mask')}
    arrays.update(lengths=np.asarray([int(row['point_mask'].sum()) for row in outputs],np.int32),
                  stopped=np.ones(len(outputs),bool))
    return save_trajectories(output,arrays,{
        'schema_version':1,'coordinate_mode':'absolute','pen_semantics':'incoming',
        'records':records,'expected_count':len(records),'renderer':RENDERER_CONFIG,
        'manifest_id':manifest['identifier'],'reference_half':half,
        'reference_protocol':('globally_support_disjoint_reserved_neighborhood_ids' if task_specs
                              else 'independent_balanced_reserved_ids'),
    })


def _condition_support(support, wrong, condition, rng, *, experiment=None):
    if experiment == 'b2' and condition == 'stroke_reversal':
        raise ValueError('B2 stroke_reversal is unavailable: executed-program reversal needs re-execution.')
    result = {key: np.array(value, copy=True) for key, value in
              (wrong if condition in ('wrong_support', 'same_category_wrong_neighborhood') else support).items()}
    if condition == 'support_reordering':
        permutation = rng.permutation(len(result['tokens']))
        return {key: value[permutation] for key, value in result.items()}
    if condition == 'action_corruption':
        mask = result['point_mask']
        coords = result['tokens'][..., :2][mask].copy()
        result['tokens'][..., :2][mask] = coords[rng.permutation(len(coords))]
    if condition == 'stroke_reversal':
        for demo in range(len(result['tokens'])):
            count = int(result['point_mask'][demo].sum())
            tokens = result['tokens'][demo]
            starts = np.flatnonzero(tokens[:count, 2] == 0)
            if count and (not len(starts) or starts[0] != 0):
                starts = np.r_[0, starts]
            for left, right in zip(starts, np.r_[starts[1:], count]):
                tokens[left:right,:2] = tokens[left:right,:2][::-1]
                result['state'][demo,left:right] = result['state'][demo,left:right][::-1]
    return result


def _sequence_result(xy, pen, maximum, *, stopped=True, extra=None):
    xy, pen = np.asarray(xy,np.float32), np.asarray(pen,np.float32)
    count = min(len(xy), maximum)
    stopped = bool(stopped and count < maximum)
    tokens = np.zeros((maximum,4),np.float32)
    tokens[:count,:2], tokens[:count,2] = xy[:count], pen[:count]
    if stopped:
        tokens[count,3] = 1
    return {'tokens': tokens, 'point_mask': np.arange(maximum)<count,
            'event_mask': np.arange(maximum)<count+int(stopped),
            'length': count, 'stopped': stopped, **(extra or {})}


def _nearest_training(store, manifest, support):
    # Selection uses only shown support geometry and the permitted training
    # reservoir. It never looks at target, anchor, category, or query features.
    descriptors = [geometry_descriptor(tokens[mask, :3]) for tokens, mask
                   in zip(support['tokens'], support['point_mask'])]
    target = np.mean([value for value in descriptors if value is not None], axis=0)
    best = None
    for ids in manifest['a_ids']['train'].values():
        for item in ids:
            record = store.get(item)
            descriptor = geometry_descriptor(np.column_stack([record.absolute, record.incoming_pen]))
            distance = float(np.linalg.norm(descriptor - target))
            candidate = (distance, str(item))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError('Nearest-training baseline needs a permitted training reservoir.')
    return store.get(best[1]), best[0]


def _network_b2(params, fast, model_cfg, key, frame, start, context, pen_config, perturbations):
    carry = initial_query_carry(model_cfg)
    def policy(state, time, states, actions):
        nonlocal carry
        previous = np.zeros(4,np.float32) if time == 0 else actions[-1].copy()
        if time:
            previous[:2] = state[:2] - states[-1,:2]
            previous[2] = state[2]
        carry, distribution = _compiled_query_step(params, fast, model_cfg, carry,
            jnp.asarray(previous), jnp.asarray(time), jnp.asarray(frame), jnp.asarray(state), context)
        return np.asarray(sample_token(distribution, model_cfg, jax.random.fold_in(key,time), deterministic=True))
    return rollout_policy(policy, start_xy=start, max_steps=model_cfg.max_steps,
                          config=pen_config, seed=int(np.asarray(key)[0]), perturbations=perturbations)


_compiled_query_step = jax.jit(query_step, static_argnums=(2,))


def _b1_ordered_tracking(generated, target, *, stopped):
    """Compare authored point indices without time normalization or alignment."""
    result = timed_tracking_metrics(generated, target, stopped=stopped)
    result.update(
        raw_program_index_xy_rmse=result['raw_timestep_xy_rmse'],
        requested_final_program_index_error=result['requested_final_timestep_error'],
        alignment_axis='authored_program_point_index',
        time_alignment='same_program_point_index_no_resampling',
        shared_metric_key_semantics='raw_timestep_xy_rmse denotes authored point indices for B1',
    )
    return result


def generate_run(checkpoint, episodes, output, *, cache_root=None, manifest_path=None,
                 conditions=None,
                 samples_per_task=4, seed=0, pen_config=None, perturbations=None):
    payload, cfg, store, manifest = load_run(checkpoint, cache_root=cache_root, manifest_path=manifest_path)
    default_conditions = conditions is None
    if default_conditions:
        conditions = (('no_update','correct_support','wrong_support','random_update_matched_norm')
                      if cfg['model_type'].startswith('ttt_') else
                      (('no_update','correct_support','wrong_support')
                       if cfg['model_type']=='explicit_context' else ('correct_support',)))
    if (samples_per_task < 1 or not conditions or set(conditions)-set(CONDITIONS)
            or len(set(conditions)) != len(conditions)):
        raise ValueError('Provide positive sample count and distinct known conditions.')
    if 'random_update_matched_norm' in conditions and not cfg['model_type'].startswith('ttt_'):
        raise ValueError('Random fast updates require a TTT checkpoint.')
    batch, evaluation = load_evaluation(episodes)
    if default_conditions and batch.get('same_category_wrong_neighborhood') and cfg['model_type'] != 'no_support':
        conditions = (*conditions, 'same_category_wrong_neighborhood')
    if 'same_category_wrong_neighborhood' in conditions and not batch.get('same_category_wrong_neighborhood'):
        raise ValueError('Same-category neighborhood controls require frozen similarity-task episodes.')
    if 'nearest_training_example' in conditions and cfg['experiment'] != 'a':
        raise ValueError('Nearest-training geometry baseline is defined for A only.')
    if evaluation['cache_id'] != store.identifier or evaluation['manifest_id'] != manifest['identifier']:
        raise ValueError('Evaluation episodes and checkpoint manifests differ.')
    model_cfg = SketchModelConfig(**cfg['model'])
    if evaluation['experiment'] != cfg['experiment'] or evaluation['max_steps'] != model_cfg.max_steps:
        raise ValueError('Evaluation public protocol differs from checkpoint.')
    if cfg['experiment'] == 'b2' and evaluation.get('motion_bound') != model_cfg.motion_bound:
        raise ValueError('Evaluation motion bound differs from the checkpoint target schedule.')
    if cfg['experiment'] == 'b2' and 'stroke_reversal' in conditions:
        raise ValueError('B2 stroke_reversal is unavailable: executed-program reversal needs re-execution.')
    perturbation_profile = {}
    for step, displacement in (perturbations or {}).items():
        displacement = np.asarray(displacement, np.float32)
        if (not isinstance(step, (int, np.integer)) or step < 0
                or displacement.shape != (2,) or not np.all(np.isfinite(displacement))):
            raise ValueError('Perturbations require nonnegative integer steps and finite XY displacements.')
        perturbation_profile[str(int(step))] = displacement.tolist()
    params = jax.tree_util.tree_map(jnp.asarray, payload['params'])
    initial = initial_fast_state(params)
    adapt = jax.jit(lambda support: adapt_sketch(params, support, model_cfg))
    generate_compiled = jax.jit(lambda fast,key,frame,context: generate(
        params,fast,model_cfg,key,frame=frame,context=context))
    conditional_loss = jax.jit(lambda fast, query, context: query_loss(params, fast, query, model_cfg, context))
    pen_config = pen_config or PenConfig(max_motion=model_cfg.motion_bound)
    if pen_config.max_motion != model_cfg.motion_bound:
        raise ValueError('Evaluation motion bound must match the trained public action contract.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    checkpoint_id = file_sha256(checkpoint)
    for condition in conditions:
        results, records, geometries, target_losses = [], [], [], []
        for task_index, record in enumerate(evaluation['records']):
            support = {key:value[task_index] for key,value in batch['support'].items()}
            wrong = {key:value[task_index] for key,value in batch['wrong_support'].items()}
            shown_record = record
            if condition in ('wrong_support', 'same_category_wrong_neighborhood'):
                shown_record = evaluation['wrong_records' if condition == 'wrong_support'
                                          else 'same_category_wrong_records'][task_index]
            if condition == 'same_category_wrong_neighborhood':
                wrong = {key:value[task_index] for key,value in batch[condition].items()}
            control_rng = np.random.default_rng(np.random.SeedSequence([seed,task_index,CONDITIONS.index(condition)]))
            chosen = _condition_support(support, wrong, condition, control_rng, experiment=cfg['experiment'])
            fast, context = initial, None
            if condition != 'no_update':
                fast, _ = adapt(jax.tree_util.tree_map(jnp.asarray, chosen))
                if model_cfg.model_type == 'explicit_context':
                    context = encode_context(params, jax.tree_util.tree_map(jnp.asarray, chosen), model_cfg)
            if condition == 'random_update_matched_norm':
                if not model_cfg.model_type.startswith('ttt_'):
                    raise ValueError('Random fast updates require a TTT checkpoint.')
                delta = jax.tree_util.tree_map(lambda a,b:a-b,fast,initial)
                scale = np.sqrt(sum(float(jnp.sum(x*x)) for x in jax.tree_util.tree_leaves(delta)))
                direction = jax.tree_util.tree_map(lambda x:jnp.asarray(control_rng.normal(size=x.shape),x.dtype),initial)
                norm = np.sqrt(sum(float(jnp.sum(x*x)) for x in jax.tree_util.tree_leaves(direction)))
                fast = jax.tree_util.tree_map(lambda x,d:x+d*scale/max(norm,1e-12),initial,direction)
            baseline = condition in ('support_copy','transformed_replay','untransformed_replay','category_prototype',
                                     'oracle','open_loop_replay','feedback_replay','nearest_training_example')
            target_loss = None
            if cfg['experiment'] == 'a' and not baseline:
                query = {name: jnp.asarray(value[task_index]) for name, value in batch['query'].items()}
                _, loss_parts = conditional_loss(fast, query, context)
                target_loss = {name: float(value) for name, value in loss_parts.items()
                               if name != 'coordinate_mse'}
                target_losses.append({'task_index': task_index, 'intended_category': record['intended_category'],
                                      'intended_neighborhood_id': record.get('intended_neighborhood_id'), **target_loss})
            nearest = _nearest_training(store, manifest, support) if condition == 'nearest_training_example' else None
            for sample_index in range(samples_per_task):
                query_index = sample_index % evaluation['query_count']
                key = jax.random.fold_in(jax.random.PRNGKey(seed), task_index*samples_per_task+sample_index)
                frame = batch['query']['frame'][task_index,query_index]
                query_meta = record['query'][query_index]
                if baseline:
                    demo_index = sample_index % len(record['support_ids'])
                    point_mask = support['point_mask'][demo_index]
                    if cfg['experiment'] == 'b2':
                        point_mask = support['knot_mask'][demo_index]
                    pen = support['tokens'][demo_index,point_mask,2]
                    xy = support['tokens'][demo_index,point_mask,:2]
                    if cfg['experiment'] == 'b2':
                        xy = support['state'][demo_index,point_mask,:2] + xy
                    if cfg['experiment'] != 'a' and condition not in ('untransformed_replay','support_copy'):
                        xy = transformed_replay(xy,support['frame'][demo_index],frame)
                    if condition == 'category_prototype':
                        pool_name = 'a_ids' if cfg['experiment']=='a' else 'b_ids'
                        ids = manifest[pool_name]['train'].get(record['intended_category'], [])
                        canonical = category_prototype(store, ids)
                        xy,pen = canonical.absolute,canonical.incoming_pen
                        if cfg['experiment'] != 'a':
                            xy = apply_frame(xy,frame)
                    if nearest is not None:
                        xy, pen = nearest[0].absolute, nearest[0].incoming_pen
                    if condition == 'oracle':
                        if cfg['experiment'] == 'a':
                            raise ValueError('A has no single target-program oracle.')
                        canonical = store.get(record['intended_base_id'])
                        xy,pen = apply_frame(canonical.absolute,frame),canonical.incoming_pen
                    if cfg['experiment'] == 'b2':
                        start = np.asarray(query_meta['start_xy'],np.float32)
                        reference, pens = timed_reference(xy,pen,start,max_motion=model_cfg.motion_bound)
                        rollout = replay_rollout(reference,pens,start_xy=start,
                            feedback=condition != 'open_loop_replay',config=pen_config,
                            seed=int(np.asarray(key)[0]),max_steps=model_cfg.max_steps,perturbations=perturbations)
                        result = _sequence_result(rollout['absolute'],rollout['incoming_pen'],model_cfg.max_steps,stopped=rollout['terminated'])
                    else:
                        result = _sequence_result(xy,pen,model_cfg.max_steps)
                elif cfg['experiment'] == 'b2':
                    rollout = _network_b2(params,fast,model_cfg,key,frame,np.asarray(query_meta['start_xy']),context,pen_config,perturbations)
                    result = _sequence_result(rollout['absolute'],rollout['incoming_pen'],model_cfg.max_steps,stopped=rollout['terminated'])
                else:
                    result = generate_compiled(fast,key,jnp.asarray(frame),context)
                    result = jax.tree_util.tree_map(np.asarray,result)
                results.append(result)
                metadata = {**record,'query_index':query_index,'condition':condition,
                    'support_ids': shown_record['support_ids'], 'support': shown_record['support'],
                    'original_support_ids': record['support_ids'],
                    'source_neighborhood_id': shown_record.get('intended_neighborhood_id'),
                    'query_seed':np.asarray(key).astype(int).tolist(),'sample_index':sample_index,
                    'checkpoint_id':checkpoint_id,'frame':frame.tolist(),'task_index':task_index,
                    'length':int(result['length']),'stopped':bool(result['stopped'])}
                if (condition == 'no_update' or model_cfg.model_type == 'no_support') and not baseline:
                    metadata['support_ids'] = []
                if target_loss is not None:
                    metadata['conditional_target_loss'] = target_loss
                if nearest is not None:
                    metadata['copied_training_id'] = nearest[0].base_id
                    metadata['nearest_training_support_geometry_distance'] = nearest[1]
                    metadata['nearest_training_selection'] = 'fixed 16-point raw support XY plus length/stroke descriptor; all permitted training IDs'
                if cfg['experiment'] == 'a':
                    point_rows = np.asarray(result['tokens'])[result['point_mask'], :3]
                    support_geometries = [paired_geometry(point_rows, drawing[mask, :3])
                                          for drawing, mask in zip(chosen['tokens'], chosen['point_mask'])]
                    valid_shapes = [row for row in support_geometries if row.get('valid_pair')]
                    metadata['raw_support_geometry'] = {
                        'evaluable': bool(valid_shapes),
                        'minimum_chamfer_squared': min((row['symmetric_chamfer_squared'] for row in valid_shapes), default=None),
                        'minimum_ordered_point_rmse': min((row['resampled_ordered_point_rmse'] for row in valid_shapes), default=None),
                        'generated_stroke_starts': int(np.sum(point_rows[:, 2] < .5)),
                        'support_stroke_starts': [int(np.sum(drawing[mask, 2] < .5))
                                                 for drawing, mask in zip(chosen['tokens'], chosen['point_mask'])],
                        'interpretation': 'copying/geometry diagnostic; no paired target reconstruction claim',
                    }
                if cfg['experiment'] == 'b2':
                    metadata['execution'] = {name:rollout[name] for name in ('invalid_actions','clipped_actions','out_of_bounds_steps','pen_up_travel','recovery_semantics')}
                    metadata['issued_actions'] = rollout['tokens'].tolist()
                    metadata['executed_states_before_action'] = rollout['state'].tolist()
                    target = np.column_stack((query_meta['reference_absolute'],query_meta['reference_incoming_pen']))
                else:
                    mask = batch['query']['point_mask'][task_index,query_index]
                    target = batch['query']['tokens'][task_index,query_index][mask,:3]
                if cfg['experiment'] != 'a':
                    geometry = paired_geometry(np.asarray(result['tokens'])[result['point_mask'],:3],target)
                    metadata['geometry'] = geometry
                    geometries.append(geometry)
                    if cfg['experiment'] == 'b2':
                        metadata['timed_tracking'] = timed_tracking_metrics(
                            np.asarray(result['tokens'])[result['point_mask'],:3],target,
                            perturbation_steps=tuple((perturbations or {}).keys()),
                            stopped=bool(result['stopped']))
                    else:
                        metadata['ordered_tracking'] = _b1_ordered_tracking(
                            np.asarray(result['tokens'])[result['point_mask'], :3], target,
                            stopped=bool(result['stopped']))
                records.append(metadata)
        arrays = {name:np.stack([item[name] for item in results]) for name in ('tokens','event_mask','point_mask')}
        arrays['lengths'] = np.asarray([int(item['length']) for item in results],np.int32)
        arrays['stopped'] = np.asarray([bool(item['stopped']) for item in results])
        metadata = {'schema_version':1,'coordinate_mode':'absolute','pen_semantics':'incoming',
                    'records':records,'expected_count':len(evaluation['records'])*samples_per_task,
                    'renderer':RENDERER_CONFIG,'evaluation_id':evaluation['identifier'],
                    'manifest_id':manifest['identifier'],'checkpoint_id':checkpoint_id,
                    'model_type':cfg['model_type'],'experiment':cfg['experiment'],
                    'a_protocol': evaluation.get('a_protocol'),
                    'protocol':'empty_prefix_free_generation' if cfg['experiment']=='a' else 'public_frame_reproduction',
                    'pen_config':asdict(pen_config) if cfg['experiment']=='b2' else None,
                    'perturbations':perturbation_profile}
        save_trajectories(output/condition,arrays,metadata)
        summaries[condition] = trajectory_statistics(arrays)
        if target_losses:
            summaries[condition]['conditional_target_loss'] = {
                'mean': float(np.mean([row['loss'] for row in target_losses])),
                'task_count': len(target_losses), 'per_task': target_losses,
                'objective': 'teacher-forced autoregressive Gaussian-mixture XY NLL plus weighted pen/STOP BCE',
                'pairing': 'identical held-out targets and permitted prefixes in every support condition; no sampling noise in this objective',
                'weighting': 'equal saved support/query task episodes; collapse intended tasks before uncertainty',
            }
        if cfg['experiment'] == 'b2':
            tracking = [row['timed_tracking'] for row in records]
            complete = [row['raw_timestep_xy_rmse'] for row in tracking
                        if row['raw_timestep_xy_rmse'] is not None]
            summaries[condition]['tracking'] = {
                'complete_output_count':sum(int(row['complete_output']) for row in tracking),
                'invalid_pair_count':sum(row['invalid_pair_count'] for row in tracking),
                'missing_step_count':sum(row['missing_step_count'] for row in tracking),
                'no_stop_failure_count':sum(int(row['no_stop_failure']) for row in tracking),
                'complete_target_xy_rmse':float(np.mean(complete)) if complete else None,
                'rmse_evaluable_count':len(complete),
                'mean_pen_up_travel':float(np.mean([row['execution']['pen_up_travel'] for row in records])),
            }
        elif cfg['experiment'] == 'b1':
            tracking = [row['ordered_tracking'] for row in records]
            complete = [row['raw_program_index_xy_rmse'] for row in tracking
                        if row['raw_program_index_xy_rmse'] is not None]
            summaries[condition]['ordered_tracking'] = {
                'complete_output_count':sum(int(row['complete_output']) for row in tracking),
                'invalid_pair_count':sum(row['invalid_pair_count'] for row in tracking),
                'missing_point_count':sum(row['missing_step_count'] for row in tracking),
                'extra_point_count':sum(row['extra_step_count'] for row in tracking),
                'no_stop_failure_count':sum(int(row['no_stop_failure']) for row in tracking),
                'complete_program_index_xy_rmse':float(np.mean(complete)) if complete else None,
                'rmse_evaluable_count':len(complete),
                'alignment_axis':'authored_program_point_index',
            }
        if geometries:
            valid = [row for row in geometries if row.get('valid_pair',False)]
            summaries[condition]['geometry_valid_pairs'] = len(valid)
            summaries[condition]['geometry_invalid_pairs'] = len(geometries)-len(valid)
            summaries[condition]['geometry'] = {
                key:float(np.mean([row[key] for row in valid]))
                for key in (valid[0] if valid else {})
                if isinstance(valid[0][key],(int,float)) and key != 'valid_pair'
            }
    _dump(output/'summary.json',{'conditions':summaries,'checkpoint_id':checkpoint_id,
                                'evaluation_id':evaluation['identifier'],
                                'pen_config':asdict(pen_config) if cfg['experiment']=='b2' else None,
                                'perturbations':perturbation_profile,'scientific_gate_passed':False})
    return output
