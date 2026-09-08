"""Explicit preparation, training, generation and bounded-pilot commands."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _configuration(path):
    path = Path(path)
    if path.suffix == '.py':
        spec = importlib.util.spec_from_file_location('quickdraw_run_config',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.get_config()
        return cfg.to_dict() if hasattr(cfg,'to_dict') else dict(cfg)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    fixture = sub.add_parser('fixture',help='Explicit synthetic correctness data, not QuickDraw evidence')
    fixture.add_argument('--output',required=True)
    fixture.add_argument('--categories',type=int,default=16)
    fixture.add_argument('--drawings-per-category',type=int,default=64)
    fixture.add_argument('--seed',type=int,default=0)
    prepare = sub.add_parser('prepare',help='Convert official NDJSON to a neutral cache')
    prepare.add_argument('--raw-root',required=True)
    prepare.add_argument('--output',required=True)
    prepare.add_argument('--max-points',type=int,default=127)
    prepare.add_argument('--max-drawings-per-category',type=int,
                         help='Declared input-prefix cap before eligibility filtering; bounds host memory')
    manifest = sub.add_parser('manifest')
    manifest.add_argument('--cache-root',required=True)
    manifest.add_argument('--output',required=True)
    for field in ('family-count','program-count','unique-budget','drawings-per-category'):
        manifest.add_argument('--'+field,type=int)
    manifest.add_argument('--reference-per-category',type=int,default=8)
    manifest.add_argument('--max-points',type=int,default=127)
    manifest.add_argument('--seed',type=int,default=0)
    manifest.add_argument('--subset-seed',type=int,default=0)
    manifest.add_argument('--a-protocol', choices=('a_nn','a_local','a_category'), default='a_nn')
    manifest.add_argument('--embedding-root', help='Frozen offline embedding export for neighborhood construction')
    manifest.add_argument('--split-regime', choices=('familiar_drawings','heldout_regions','heldout_categories'),
                          default='familiar_drawings')
    manifest.add_argument('--neighborhood-count', type=int)
    manifest.add_argument('--evaluation-neighborhood-count', type=int, default=16,
                          help='Balanced task cap in each held-out split; keep fixed across the N sweep')
    manifest.add_argument('--neighborhood-size', type=int, default=64)
    manifest.add_argument('--reference-per-neighborhood', type=int, default=8)
    manifest.add_argument('--selection-mode', choices=('exact_top_k','sample_top_m'), default='exact_top_k')
    manifest.add_argument('--top-m', type=int, default=32)
    manifest.add_argument('--region-count', type=int, default=8)
    manifest.add_argument('--min-neighbor-cosine', type=float, default=-1.0)
    manifest.add_argument('--budget-regime', choices=('fixed_reservoir','per_neighborhood'), default='fixed_reservoir')
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--config',required=True,help='JSON or Python get_config()')
    train_parser.add_argument('--set',action='append',default=[],metavar='FIELD=JSON',help='Nested config override, e.g. model.hidden_dim=32')
    episodes = sub.add_parser('episodes',help='Freeze prompts/targets/control sources before generation')
    for field in ('cache-root','manifest-path','output'):
        episodes.add_argument('--'+field,required=True)
    episodes.add_argument('--experiment',choices=('a','b1','b2'),default='a')
    episodes.add_argument('--split',choices=('train','development','test'),default='development')
    episodes.add_argument('--b-partition',choices=('familiar','heldout_category'),default='familiar')
    for field,default in [('tasks',8),('support-count',4),('query-count',1),('max-steps',128),('seed',0)]:
        episodes.add_argument('--'+field,type=int,default=default)
    episodes.add_argument('--allow-test',action='store_true')
    episodes.add_argument('--motion-bound',type=float,default=.1)
    episodes.add_argument('--wrong-neighborhood-min-distance', type=float, default=.05)
    episodes.add_argument('--wrong-neighborhood-max-jaccard', type=float, default=.1)
    generation = sub.add_parser('generate')
    for field in ('checkpoint','episodes','output'):
        generation.add_argument('--'+field,required=True)
    for field in ('cache-root','manifest-path'):
        generation.add_argument('--'+field)
    generation.add_argument('--conditions',nargs='+',help='Defaults to controls supported by the checkpoint type')
    generation.add_argument('--samples-per-task',type=int,default=4)
    generation.add_argument('--seed',type=int,default=0)
    generation.add_argument('--noise-std',type=float,default=0)
    generation.add_argument('--delay-steps',type=int,default=0)
    generation.add_argument('--perturbation-step',type=int)
    generation.add_argument('--perturbation-xy',nargs=2,type=float,default=[.1,-.1])
    refs = sub.add_parser('references')
    for field in ('cache-root','manifest-path','output'):
        refs.add_argument('--'+field,required=True)
    refs.add_argument('--split',choices=('train','development','test'),default='development')
    refs.add_argument('--half',choices=('real_a','real_b'),default='real_a')
    refs.add_argument('--max-steps',type=int,default=128)
    refs.add_argument('--allow-test',action='store_true')
    refs.add_argument('--episodes', help='Use the exact intended neighborhood set of these frozen episodes')
    pilot = sub.add_parser('pilot',help='Explicitly launch a bounded matched A pilot; never an automatic sweep')
    pilot.add_argument('--cache-root',required=True)
    pilot.add_argument('--output',required=True)
    pilot.add_argument('--levels',nargs='+',type=int,
                       help='A-local: two N levels; A-NN: one target budget; A-category: two F levels')
    pilot.add_argument('--a-protocol', choices=('a_nn','a_local','a_category'), default='a_local')
    pilot.add_argument('--embedding-root')
    pilot.add_argument('--family-count', type=int, default=8)
    pilot.add_argument('--split-regime', choices=('familiar_drawings','heldout_regions','heldout_categories'),
                       default='familiar_drawings')
    pilot.add_argument('--neighborhood-size', type=int, default=64)
    pilot.add_argument('--evaluation-neighborhood-count', type=int, default=16)
    pilot.add_argument('--selection-mode', choices=('exact_top_k','sample_top_m'), default='exact_top_k')
    pilot.add_argument('--top-m', type=int, default=32)
    pilot.add_argument('--region-count', type=int, default=8)
    pilot.add_argument('--wrong-neighborhood-min-distance', type=float, default=.05)
    pilot.add_argument('--wrong-neighborhood-max-jaccard', type=float, default=.1)
    pilot.add_argument('--steps',type=int,required=True,help='Choose after a smoke profile')
    pilot.add_argument('--unique-budget',type=int, help='Required for fixed_reservoir; omit for per_neighborhood')
    pilot.add_argument('--budget-regime', choices=('fixed_reservoir','per_neighborhood'), default='fixed_reservoir')
    pilot.add_argument('--max-steps',type=int,default=128)
    pilot.add_argument('--seed',type=int,default=0)
    pilot.add_argument('--subset-seed',type=int,default=0)
    pilot.add_argument('--config',help='Shared JSON/Python config; architecture fixed across protocols and diversity levels')
    analysis = sub.add_parser('analyze')
    for field in ('checkpoint','episodes','output'):
        analysis.add_argument('--'+field,required=True)
    analysis.add_argument('--cache-root')
    analysis.add_argument('--manifest-path')
    gates = sub.add_parser('gates',help='Explicit bounded fixed-batch fit and fast-subspace feasibility checks')
    for field in ('cache-root','manifest-path','output'):
        gates.add_argument('--'+field,required=True)
    gates.add_argument('--experiment',choices=('a','b1','b2'),default='a')
    gates.add_argument('--steps',type=int,default=50)
    gates.add_argument('--seed',type=int,default=0)
    gates.add_argument('--config',help='Shared config; only its model fields define gate architecture')
    gallery = sub.add_parser('gallery',help='Render a previously selected qualitative SVG gallery')
    gallery.add_argument('--artifact',required=True)
    gallery.add_argument('--output',required=True)
    gallery.add_argument('--selection',required=True,help='Frozen selection from quickdraw.gallery select')
    aggregate = sub.add_parser('aggregate',help='Aggregate saved controls at category/base-program level')
    aggregate.add_argument('--specification',required=True)
    aggregate.add_argument('--output',required=True)
    aggregate.add_argument('--seed',type=int,default=0)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'fixture':
        from .data import create_fixture_cache
        args['root'] = args.pop('output')
        result = create_fixture_cache(**args).identifier
    elif command == 'prepare':
        from .data import import_ndjson
        result = import_ndjson(args['raw_root'],args['output'],max_points=args['max_points'],
                               max_drawings_per_category=args['max_drawings_per_category']).identifier
    elif command == 'manifest':
        from .data import SketchStore,build_manifest,save_manifest
        store = SketchStore.open(args.pop('cache_root'))
        output = args.pop('output')
        embedding_root = args.pop('embedding_root')
        if args['a_protocol'] == 'a_category' and not embedding_root:
            if args['budget_regime'] != 'fixed_reservoir':
                parser.error('Per-neighborhood allowance requires --a-protocol a_local and --embedding-root.')
            fields = ('seed','subset_seed','family_count','program_count','unique_budget',
                      'drawings_per_category','reference_per_category','max_points')
            manifest = build_manifest(store, **{name:args[name] for name in fields})
        else:
            if not embedding_root:
                parser.error('A-NN/A-local require --embedding-root from the frozen classifier export.')
            if args['family_count'] is None:
                args['family_count'] = 8
            if args.pop('program_count') is not None:
                parser.error('--program-count belongs to a separate B manifest; use --a-protocol a_category.')
            from .embeddings import EmbeddingStore
            from .neighborhoods import build_neighborhood_manifest
            manifest = build_neighborhood_manifest(store, EmbeddingStore.open(embedding_root), **args)
        save_manifest(output,manifest)
        result = manifest['identifier']
    elif command == 'train':
        from .train import train
        cfg = _configuration(args['config'])
        for override in args['set']:
            name,value = override.split('=',1)
            target = cfg
            fields = name.split('.')
            for field in fields[:-1]:
                target = target.setdefault(field,{})
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            target[fields[-1]] = value
        result = train(cfg)
    elif command == 'episodes':
        from .evaluate import prepare_evaluation
        result = prepare_evaluation(**args)
    elif command == 'generate':
        from .evaluate import generate_run
        from .pen import PenConfig
        from .train import load_run
        payload,cfg,_,_ = load_run(args['checkpoint'],cache_root=args['cache_root'],manifest_path=args['manifest_path'])
        args['pen_config'] = PenConfig(max_motion=cfg['model']['motion_bound'],noise_std=args.pop('noise_std'),delay_steps=args.pop('delay_steps'))
        step,delta = args.pop('perturbation_step'),args.pop('perturbation_xy')
        args['perturbations'] = None if step is None else {step:delta}
        result = generate_run(**args)
    elif command == 'references':
        from .evaluate import export_references
        result = export_references(**args)
    elif command == 'pilot':
        from .pilot import run_pilot
        args['config'] = _configuration(args['config']) if args['config'] else {}
        result = run_pilot(**args)
    elif command == 'gates':
        from .gates import run_gates
        config_path = args.pop('config')
        config = _configuration(config_path) if config_path else {}
        args['model_config'] = (config or {}).get('model',{})
        result = run_gates(**args)
    elif command == 'gallery':
        from .gallery import export_gallery
        result = export_gallery(args['artifact'],args['output'],selection_path=args['selection'])
    elif command == 'aggregate':
        from .aggregate import aggregate_runs
        result = aggregate_runs(**args)
    else:
        from .analysis import extract_checkpoint_updates
        result = extract_checkpoint_updates(**args)
    print(result)


if __name__ == '__main__':
    main()
