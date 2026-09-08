"""Offline CPU worker for the original renderer and a named frozen evaluator.

Run this file with a separate Python environment containing NumPy, Pillow,
SciPy, torch and torchvision. No JAX, policy, donor loaders or FAISS are imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np


def load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load evaluator source: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--donor-root', required=True)
    parser.add_argument('--extractor-checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--label-map')
    parser.add_argument('--checkpoint-provenance')
    args = parser.parse_args()
    checkpoint, donor_root = Path(args.extractor_checkpoint), Path(args.donor_root)
    if not checkpoint.is_file():
        parser.error(f'QuickDraw evaluator checkpoint is missing: {checkpoint}. Supply a trained grayscale 345-class ResNet18 checkpoint.')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    metric_path, renderer_path = donor_root / 'metrics/resnet18.py', donor_root / 'dataset/rasterize.py'
    for path in (metric_path, renderer_path):
        if not path.is_file():
            parser.error(f'Required donor source is missing: {path}')

    # Direct source loads avoid either repository's top-level package imports.
    metrics = load_source(Path(__file__).with_name('metrics.py'), '_icil_offline_quickdraw_metrics')
    arrays, metadata = metrics.load_trajectories(args.artifact)
    renderer = load_source(renderer_path, '_quickdraw_original_rasterizer')
    config = renderer.RasterizerConfig(**metrics.RENDERER_CONFIG)
    rasters, invalid, point_hashes = [], [], []
    for tokens, mask in zip(arrays['tokens'], arrays['point_mask']):
        points = tokens[np.asarray(mask, dtype=bool), :3]
        failed = not np.all(np.isfinite(points))
        invalid.append(failed)
        point_hashes.append(None if failed else hashlib.sha256(b'float32_le_xy_incomingpen_v1\0' + np.asarray(points, dtype='<f4').tobytes()).hexdigest())
        # Count-preserving failure policy is recorded in metadata and statistics.
        render_points = np.zeros((0, 3), dtype=np.float32) if failed else points
        rasters.append(renderer.rasterize_absolute_points(render_points, config=config))
    if not rasters:
        parser.error('Cannot extract features from an empty sample population')
    rasters = np.stack(rasters)

    try:
        import torch
        import torchvision
    except ImportError as error:
        parser.error(f'Frozen evaluator needs torch and torchvision in this separate metric interpreter: {error}')
    original = load_source(metric_path, '_quickdraw_original_resnet18')
    # Same donor architecture and unchanged tensor values; map saved CUDA tensors
    # to CPU explicitly because the donor constructor omits map_location.
    model = original.resnet18(pretrained=False, num_classes=345)
    model.conv1 = original.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    classifier_weight = model.fc.weight.detach().clone()
    classifier_bias = model.fc.bias.detach().clone()
    model.fc = original.nn.Identity()
    model.eval()
    model.requires_grad_(False)
    features, predictions = [], []
    with torch.inference_mode():
        for start in range(0, len(rasters), args.batch_size):
            batch = torch.from_numpy(rasters[start:start + args.batch_size, None]).to(dtype=torch.float32, device='cpu')
            output = model(batch)
            if output.shape[1] != 512:
                raise ValueError('QuickDraw feature extractor must return 512 channels')
            features.append(output.numpy())
            predictions.append((output @ classifier_weight.T + classifier_bias).argmax(dim=-1).numpy())
    features, predictions = np.concatenate(features), np.concatenate(predictions)
    if not np.all(np.isfinite(features)):
        raise ValueError('Frozen extractor produced nonfinite features')
    extractor_hash = metrics.file_sha256(checkpoint)
    provenance = {'training_ids': 'unknown', 'evaluation_only': True}
    if args.checkpoint_provenance:
        provenance = json.loads(Path(args.checkpoint_provenance).read_text())
        if provenance.get('extractor_sha256') != extractor_hash:
            raise ValueError('Checkpoint provenance hash does not match the frozen evaluator')
    classification = None
    if args.label_map:
        label_path = Path(args.label_map)
        if provenance.get('label_map_sha256') != metrics.file_sha256(label_path):
            raise ValueError('Classifier metrics require checkpoint provenance attesting the exact label-map hash')
        labels = json.loads(label_path.read_text())
        if set(labels.values()) != set(range(345)) or len(labels) != 345:
            raise ValueError('Classifier label map must map 345 categories one-to-one to 0..344')
        target_labels = np.asarray([labels[record['intended_category']] for record in metadata['records']])
        accuracy = predictions == target_labels
        classification = {
            'accuracy': float(accuracy.mean()),
            'per_category_accuracy': {category: float(accuracy[np.asarray([record['intended_category'] for record in metadata['records']]) == category].mean()) for category in sorted(set(record['intended_category'] for record in metadata['records']))},
            'label_map_sha256': metrics.file_sha256(label_path),
            'resource_role': 'evaluation_only',
        }
    output_metadata = {
        'schema_version': metrics.SCHEMA_VERSION,
        'feature_version': metrics.FEATURE_VERSION,
        'feature_dimension': 512, 'feature_normalization': 'none',
        'extractor_sha256': extractor_hash,
        'extractor_training_provenance': provenance,
        'loader_version': 'donor_architecture_cpu_state_dict_v1',
        'renderer_config': metrics.RENDERER_CONFIG,
        'source_hashes': {'rasterizer': metrics.file_sha256(renderer_path), 'resnet18': metrics.file_sha256(metric_path)},
        'artifact_sha256': metrics.file_sha256(Path(args.artifact) / 'trajectories.npz'),
        'artifact_metadata_sha256': metrics.file_sha256(Path(args.artifact) / 'metadata.json'),
        'actual_count': len(features),
        'records': [{**record, 'trajectory_sha256': digest, 'trajectory_valid': not failed}
                    for record, digest, failed in zip(metadata['records'], point_hashes, invalid)],
        'trajectory_hash_convention': 'float32_le_xy_incomingpen_v1; effective points before STOP; no alignment',
        'manifest_id': metadata.get('manifest_id'),
        'experiment': metadata.get('experiment'),
        'evaluation_id': metadata.get('evaluation_id'),
        'reference_half': metadata.get('reference_half'),
        'reference_protocol': metadata.get('reference_protocol'),
        'statistics': metrics.trajectory_statistics(arrays, rasters),
        'invalid_render_policy': 'blank_for_nonfinite_trajectory; retain sample in score',
        'classification': classification,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'device': 'cpu'},
    }
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path / 'features.npz', features=features, classifier_predictions=predictions, invalid=np.asarray(invalid, dtype=bool))
    np.savez_compressed(output_path / 'rasters.npz', rasters=rasters)
    (output_path / 'metadata.json').write_text(json.dumps(output_metadata, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'actual_count': len(features), 'extractor_sha256': extractor_hash, 'output': str(output_path)}))


if __name__ == '__main__':
    main()
