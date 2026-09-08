"""Train a frozen curation/evaluation grayscale ResNet18 in an isolated Torch environment.

This is a newly trained evaluator with new score provenance. It does not recreate
the unavailable legacy checkpoint or introduce learned features into JAX policy
training. Execute this file with the isolated metric Python interpreter.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import platform
import random
import sys
import uuid
from pathlib import Path

import numpy as np


RENDERER_CONFIG = {'img_size': 64, 'antialias': 2, 'line_width': 2.0,
                   'background_value': 0.0, 'stroke_value': 1.0, 'normalize_inputs': False}
CLASS_COUNT = 345
SPLITS = {'train': 0, 'validation': 1, 'val': 1, 'test': 2}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_cache(cache_root: str | Path) -> dict:
    """Check immutable raster/schema/ID/split boundaries before importing Torch."""
    root = Path(cache_root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('renderer_config') != RENDERER_CONFIG:
        raise ValueError('Classifier cache needs schema 1 and the frozen QuickDraw renderer configuration')
    if manifest.get('cache_id') != _json_digest({key: value for key, value in manifest.items() if key != 'cache_id'}):
        raise ValueError('Classifier cache manifest hash mismatch')
    for name in ('images.npy', 'labels.npy', 'splits.npy', 'records.jsonl', 'label_map.json'):
        if manifest.get('files', {}).get(name) != sha256(root / name):
            raise ValueError(f'Classifier cache file hash mismatch: {name}')
    label_map = json.loads((root / 'label_map.json').read_text())
    if len(label_map) != CLASS_COUNT or set(label_map.values()) != set(range(CLASS_COUNT)):
        raise ValueError('Classifier label_map.json must map 345 categories one-to-one onto 0..344')
    images = np.load(root / 'images.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(root / 'labels.npy', mmap_mode='r', allow_pickle=False)
    splits = np.load(root / 'splits.npy', mmap_mode='r', allow_pickle=False)
    if images.dtype != np.float32 or images.ndim != 3 or images.shape[1:] != (64, 64):
        raise ValueError('Classifier images must be float32 [N,64,64]')
    if labels.dtype != np.int64 or labels.shape != (len(images),):
        raise ValueError('Classifier labels must be int64 [N]')
    if splits.dtype != np.uint8 or splits.shape != labels.shape or not np.all(np.isin(splits, [0, 1, 2])):
        raise ValueError('Classifier splits must be uint8 [N] with train=0, validation=1, test=2')
    if np.any(labels < 0) or np.any(labels >= CLASS_COUNT):
        raise ValueError('Classifier labels are outside the 345-class mapping')
    for start in range(0, len(images), 1024):
        block = images[start:start + 1024]
        if not np.all(np.isfinite(block)) or np.any(block < 0) or np.any(block > 1):
            raise ValueError('Classifier rasters must contain finite raw intensities in [0,1]')
    ids, clusters = set(), {}
    records_count = 0
    with (root / 'records.jsonl').open() as handle:
        for index, line in enumerate(handle):
            if index >= len(images):
                raise ValueError('Classifier record count exceeds array length')
            record = json.loads(line)
            drawing_id, cluster = str(record['drawing_id']), str(record['duplicate_cluster_id'])
            split = SPLITS.get(record['split'], record['split'])
            if drawing_id in ids:
                raise ValueError('Classifier drawing IDs must be unique')
            if cluster in clusters and clusters[cluster] != int(split):
                raise ValueError('Classifier duplicate cluster crosses train/validation/test')
            if split != int(splits[index]) or label_map.get(record['category']) != int(labels[index]):
                raise ValueError('Classifier record category/split disagrees with label arrays')
            ids.add(drawing_id)
            clusters[cluster] = int(split)
            records_count += 1
    if records_count != len(images):
        raise ValueError('Classifier record and array counts differ')
    counts = {name: int(np.sum(splits == value)) for name, value in [('train', 0), ('validation', 1), ('test', 2)]}
    if any(count < 1 for count in counts.values()):
        raise ValueError('Classifier train, validation and untouched test splits must all be nonempty')
    return {'manifest': manifest, 'label_map': label_map, 'counts': counts,
            'observed_labels': {name: sorted(map(int, np.unique(labels[splits == code])))
                                for name, code in [('train', 0), ('validation', 1), ('test', 2)]}}


class CachedSketchImages:
    """Map-style readonly memmap dataset; works with fork or spawn workers."""

    def __init__(self, root, split):
        self.root = Path(root)
        self.indices = np.flatnonzero(np.load(self.root / 'splits.npy', mmap_mode='r') == split)
        self.images = None
        self.labels = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        return {**self.__dict__, 'images': None, 'labels': None}

    def __getitem__(self, index):
        import torch
        if self.images is None:
            self.images = np.load(self.root / 'images.npy', mmap_mode='r')
            self.labels = np.load(self.root / 'labels.npy', mmap_mode='r')
        source = int(self.indices[index])
        return torch.from_numpy(np.array(self.images[source], copy=True))[None], int(self.labels[source])


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _atomic_torch(path, value, torch):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _seed_worker(_):
    import torch
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _measure(model, loader, device, torch, max_batches=None) -> tuple[dict, np.ndarray]:
    model.eval()
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    loss_sum, count = 0.0, 0
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum').item())
            predicted = logits.argmax(dim=-1)
            np.add.at(confusion, (labels.cpu().numpy(), predicted.cpu().numpy()), 1)
            count += len(labels)
    class_counts = confusion.sum(axis=1)
    observed = class_counts > 0
    accuracy = np.diag(confusion)[observed] / class_counts[observed]
    return {'loss': loss_sum / count, 'micro_accuracy': float(np.trace(confusion) / count),
            'macro_accuracy': float(accuracy.mean()), 'sample_count': count,
            'observed_class_count': int(observed.sum()),
            'per_class_accuracy': {str(index): float(confusion[index, index] / class_counts[index])
                                   for index in np.flatnonzero(observed)}}, confusion


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _epoch_metrics(row):
    metrics = {key: row[key] for key in ('epoch', 'global_step', 'best_epoch')}
    for split, names in (
        ('train', ('loss', 'micro_accuracy', 'sample_count')),
        ('validation', ('loss', 'micro_accuracy', 'macro_accuracy', 'sample_count')),
    ):
        metrics.update({f'{split}/{name}': row[split][name] for name in names})
    return metrics


@contextmanager
def _wandb_logging(args, output, config, audited, runtime, history):
    """Log directly to W&B; local checkpoints remain the training authority."""
    project = getattr(args, 'wandb_project', None)
    mode = getattr(args, 'wandb_mode', 'online')
    if not project or mode == 'disabled':
        yield None
        return
    identity_path = output / 'wandb_run.json'
    saved = json.loads(identity_path.read_text()) if identity_path.exists() else None
    entity = getattr(args, 'wandb_entity', None)
    if saved and (saved['project'] != project or (entity and entity != saved['entity'])):
        raise ValueError('Use the saved W&B project/entity when continuing this classifier run')
    # W&B startup must not change the classifier's seeded Python/NumPy streams.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    run = None
    exit_code = 1
    try:
        try:
            import wandb
            resume = bool(saved and saved['mode'] == mode == 'online')
            run = wandb.init(
                project=project, entity=entity or (saved['entity'] if saved else None),
                name=getattr(args, 'wandb_name', None) or output.name,
                id=saved['id'] if resume else uuid.uuid4().hex,
                resume='allow' if resume else 'never', mode=mode, dir=str(output),
                job_type='classifier-training',
                config={**config, 'cache_id': audited['manifest']['cache_id'],
                        'split_counts': audited['counts'], 'runtime': runtime,
                        'logging_frequency': 'completed_epoch'},
            )
            _atomic_json(identity_path, {'id': run.id, 'project': project,
                                        'entity': run.entity, 'mode': mode})
            run.define_metric('epoch')
            for name in ('train/*', 'validation/*', 'global_step', 'best_epoch'):
                run.define_metric(name, step_metric='epoch')
            # This also enables logging when resuming an older, unlogged run.
            for row in history:
                if row['epoch'] >= run.step:
                    run.log(_epoch_metrics(row), step=row['epoch'])
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        yield run
        exit_code = 0
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)


def run(args) -> Path:
    root, output = Path(args.cache_root).resolve(), Path(args.output).resolve()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0 or args.threads < 1:
        raise ValueError('Invalid classifier training settings')
    if args.smoke_batches is not None and args.smoke_batches < 1:
        raise ValueError('--smoke-batches must be a positive explicit diagnostic limit')
    audited = validate_cache(root)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import torchvision
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generators = {name: torch.Generator().manual_seed(args.seed + offset)
                  for name, offset in [('train', 101), ('validation', 202), ('test', 303)]}
    loaders = {name: torch.utils.data.DataLoader(
        CachedSketchImages(root, code), batch_size=args.batch_size,
        shuffle=name == 'train', num_workers=args.workers,
        generator=generators[name], worker_init_fn=_seed_worker,
        persistent_workers=False, drop_last=False,
    ) for name, code in [('train', 0), ('validation', 1), ('test', 2)]}
    model = torchvision.models.resnet18(weights=None, num_classes=CLASS_COUNT)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(device)
    config = {'batch_size': args.batch_size, 'lr': args.lr, 'seed': args.seed,
              'workers': args.workers, 'device': device, 'threads': args.threads,
              'architecture': 'torchvision_resnet18_grayscale_345', 'augmentation': 'none',
              'pretrained_weights': None, 'optimizer': 'Adam_foreach_false',
              'smoke_batches': args.smoke_batches, 'smoke_only': args.smoke_batches is not None}
    runtime = {'python': platform.python_version(), 'torch': torch.__version__,
               'torchvision': torchvision.__version__, 'numpy': np.__version__,
               'device': device, 'deterministic_algorithms': True,
               'cuda_runtime': torch.version.cuda,
               'cuda_device_name': torch.cuda.get_device_name() if device == 'cuda' else None}
    if args.evaluate_only:
        checkpoint = Path(args.checkpoint) if args.checkpoint else output / 'resnet18_best.pt'
        provenance = json.loads((output / 'provenance.json').read_text())
        if provenance['cache_id'] != audited['manifest']['cache_id'] or provenance['extractor_sha256'] != sha256(checkpoint):
            raise ValueError('Evaluation must use the validation-selected checkpoint and its original cache')
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        result, confusion = _measure(model, loaders[args.split], device, torch, args.smoke_batches)
        result.update(split=args.split, extractor_sha256=sha256(checkpoint),
                      cache_id=audited['manifest']['cache_id'], selection_uses_test=False,
                      smoke_only=args.smoke_batches is not None or provenance.get('smoke_only', False))
        _atomic_json(output / f'evaluation_{args.split}.json', result)
        np.save(output / f'confusion_{args.split}.npy', confusion)
        print(f"{args.split}: loss={result['loss']:.4f}, accuracy={result['micro_accuracy']:.2%}, "
              f"macro accuracy={result['macro_accuracy']:.2%}, samples={result['sample_count']}", flush=True)
        print(f"Per-class metrics: {output / f'evaluation_{args.split}.json'}", flush=True)
        return checkpoint
    if args.checkpoint:
        raise ValueError('--checkpoint is evaluation-only; scratch training does not import policy/evaluator weights')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=False)
    start_epoch, global_step = 0, 0
    best_macro, best_loss, best_epoch, best_state = -float('inf'), float('inf'), -1, None
    history = []
    if args.resume:
        state = torch.load(output / 'last_training.pt', map_location=device, weights_only=False)
        if state['config'] != config or state['cache_id'] != audited['manifest']['cache_id'] or state['runtime'] != runtime:
            raise ValueError('Resume requires identical cache, scientific config, device and runtime versions')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, global_step = state['next_epoch'], state['global_step']
        best_macro, best_loss, best_epoch = state['best_macro'], state['best_loss'], state['best_epoch']
        best_state = {key: value.cpu() for key, value in state['best_model'].items()}
        history = state['history']
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch_cpu'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state['rng']['torch_cuda']])
        for name, generator in generators.items():
            generator.set_state(state['rng']['data'][name].cpu())
        # Repair aliases if interruption happened between epoch artifact writes.
        _atomic_torch(output / 'resnet18_best.pt', best_state, torch)
        _atomic_torch(output / 'resnet18_last.pt', _cpu_state(model), torch)
    else:
        output.mkdir(parents=True, exist_ok=False)
    if start_epoch >= args.epochs:
        raise ValueError(f'Resume has already completed {start_epoch} epochs; request a larger --epochs total')

    def provenance(checkpoint, epoch, selection):
        return {
            'schema_version': 1, 'checkpoint_role': 'newly_trained_evaluator_v1',
            'smoke_only': args.smoke_batches is not None,
            'evaluation_only': True, 'extractor_sha256': sha256(checkpoint),
            'label_map_sha256': sha256(root / 'label_map.json'),
            'cache_id': audited['manifest']['cache_id'], 'cache_manifest_sha256': sha256(root / 'manifest.json'),
            'renderer_config': RENDERER_CONFIG, 'source_hashes': audited['manifest'].get('source_hashes', {}),
            'trainer_source_sha256': sha256(__file__), 'config': config, 'runtime': runtime,
            'training_records': {'path': str(root / 'records.jsonl'), 'sha256': sha256(root / 'records.jsonl'),
                                 'split': 'train', 'count': audited['counts']['train']},
            'split_counts': audited['counts'], 'observed_labels': audited['observed_labels'],
            'cache_partition_provenance': {key: value for key, value in audited['manifest'].items() if key not in ('files', 'cache_id')},
            'checkpoint_epoch': epoch, 'selection': selection,
            'test_used_for_checkpoint_selection': False,
            'legacy_comparability': 'same_architecture_and_renderer_new_weights_not_legacy_scores',
        }

    with _wandb_logging(args, output, config, audited, runtime, history) as wandb_run:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            train_loss, train_correct, train_count = 0.0, 0, 0
            for batch_index, (images, labels) in enumerate(loaders['train']):
                if args.smoke_batches is not None and batch_index >= args.smoke_batches:
                    break
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite classifier training loss')
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * len(labels)
                train_correct += int((logits.argmax(dim=-1) == labels).sum().item())
                train_count += len(labels)
                global_step += 1
            validation, confusion = _measure(model, loaders['validation'], device, torch, args.smoke_batches)
            if not np.isfinite(validation['loss']):
                raise ValueError('Nonfinite classifier validation loss')
            current_state = _cpu_state(model)
            if (validation['macro_accuracy'], -validation['loss']) > (best_macro, -best_loss):
                best_macro, best_loss, best_epoch = validation['macro_accuracy'], validation['loss'], epoch + 1
                best_state = current_state
            history.append({'epoch': epoch + 1, 'global_step': global_step,
                            'train': {'loss': train_loss / train_count, 'micro_accuracy': train_correct / train_count, 'sample_count': train_count},
                            'validation': validation, 'best_epoch': best_epoch})
            state = {
                'schema_version': 1, 'config': config, 'runtime': runtime,
                'cache_id': audited['manifest']['cache_id'], 'next_epoch': epoch + 1,
                'global_step': global_step, 'model': current_state, 'optimizer': optimizer.state_dict(),
                'best_model': best_state, 'best_macro': best_macro, 'best_loss': best_loss,
                'best_epoch': best_epoch, 'history': history,
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch_cpu': torch.get_rng_state(),
                        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        'data': {name: generator.get_state() for name, generator in generators.items()}},
            }
            _atomic_torch(output / 'last_training.pt', state, torch)
            _atomic_torch(output / 'resnet18_last.pt', current_state, torch)
            _atomic_torch(output / 'resnet18_best.pt', best_state, torch)
            criterion = {'criterion': 'highest_validation_macro_accuracy_then_lowest_loss',
                         'best_epoch': best_epoch, 'macro_accuracy': best_macro, 'loss': best_loss}
            _atomic_json(output / 'provenance.json', provenance(output / 'resnet18_best.pt', best_epoch, criterion))
            _atomic_json(output / 'provenance_last.json', provenance(output / 'resnet18_last.pt', epoch + 1, {'criterion': 'last_completed_epoch'}))
            np.save(output / f'confusion_validation_epoch{epoch + 1:04d}.npy', confusion)
            temporary = output / 'metrics.jsonl.tmp'
            temporary.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in history))
            temporary.replace(output / 'metrics.jsonl')
            row = history[-1]
            print(f"Epoch {epoch + 1}/{args.epochs} | "
                  f"train loss={row['train']['loss']:.4f}, accuracy={row['train']['micro_accuracy']:.2%} | "
                  f"val loss={validation['loss']:.4f}, accuracy={validation['micro_accuracy']:.2%}, "
                  f"macro accuracy={validation['macro_accuracy']:.2%} | best epoch={best_epoch}", flush=True)
            if wandb_run is not None:
                wandb_run.log(_epoch_metrics(row), step=epoch + 1)
    return output / 'resnet18_best.pt'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--checkpoint')
    parser.add_argument('--smoke-batches', type=int, help='Explicit diagnostic batch cap for train/validation; incompatible with full-run resume')
    parser.add_argument('--wandb-project', help='Enable W&B epoch logging directly during training')
    parser.add_argument('--wandb-name', help='W&B run name; defaults to the output directory name')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'),
                        default=os.environ.get('WANDB_MODE', 'online'))
    print(run(parser.parse_args()))


if __name__ == '__main__':
    main()
