from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import classifier


def _rehash(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest.pop('cache_id', None)
    manifest['files'] = {name: classifier.sha256(root / name)
                         for name in ('images.npy', 'labels.npy', 'splits.npy', 'records.jsonl', 'label_map.json')}
    manifest['cache_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps(manifest))


def _cache(root):
    root.mkdir()
    rng = np.random.default_rng(14)
    np.save(root / 'images.npy', rng.uniform(0, 1, (6, 64, 64)).astype(np.float32))
    np.save(root / 'labels.npy', np.tile(np.arange(2), 3).astype(np.int64))
    np.save(root / 'splits.npy', np.repeat(np.arange(3), 2).astype(np.uint8))
    label_map = {f'fixture_category_{index:03d}': index for index in range(345)}
    (root / 'label_map.json').write_text(json.dumps(label_map))
    records = [{'drawing_id': f'fixture-{index}', 'category': f'fixture_category_{index % 2:03d}',
                'split': ('train', 'validation', 'test')[index // 2], 'duplicate_cluster_id': f'cluster-{index}'} for index in range(6)]
    (root / 'records.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
    (root / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'renderer_config': classifier.RENDERER_CONFIG,
                                                   'source_hashes': {}, 'kind': 'synthetic_classifier_correctness_fixture'}))
    _rehash(root)
    return root


def test_classifier_cache_checks_ids_labels_clusters_and_frozen_rendering(tmp_path):
    root = _cache(tmp_path / 'cache')
    audited = classifier.validate_cache(root)
    assert audited['counts'] == {'train': 2, 'validation': 2, 'test': 2}
    assert audited['observed_labels']['train'] == [0, 1]
    records = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines()]
    records[2]['duplicate_cluster_id'] = records[0]['duplicate_cluster_id']
    (root / 'records.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
    with pytest.raises(ValueError, match='file hash mismatch'):
        classifier.validate_cache(root)
    _rehash(root)
    with pytest.raises(ValueError, match='duplicate cluster crosses'):
        classifier.validate_cache(root)
    records[2]['duplicate_cluster_id'] = 'cluster-2'
    records[2]['category'] = 'fixture_category_001'
    (root / 'records.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
    _rehash(root)
    with pytest.raises(ValueError, match='category/split disagrees'):
        classifier.validate_cache(root)


def test_classifier_import_does_not_import_torch_jax_or_wandb():
    script = ('import sys; import icil_jax_rlbench.quickdraw.classifier; '
              'assert not {"torch", "jax", "wandb"}.intersection(sys.modules)')
    subprocess.run([sys.executable, '-c', script], check=True)


def test_wandb_online_resume_backfills_missing_epochs_and_finishes_on_failure(tmp_path, monkeypatch):
    runs = []

    class Run:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.id = kwargs['id']
            self.entity = kwargs['entity'] or 'fixture-team'
            self.step = 2 if runs else 0
            self.logs = []
            self.finishes = []

        def define_metric(self, *_args, **_kwargs):
            pass

        def log(self, metrics, *, step):
            self.logs.append((step, metrics))
            self.step = step + 1

        def finish(self, *, exit_code):
            self.finishes.append(exit_code)

    def init(**kwargs):
        run = Run(kwargs)
        runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init))
    args = SimpleNamespace(wandb_project='fixture-project', wandb_mode='online')
    audited = {'manifest': {'cache_id': 'fixture-cache'}, 'counts': {'train': 2, 'validation': 2, 'test': 2}}
    history = [{'epoch': epoch, 'global_step': epoch, 'best_epoch': 1,
                'train': {'loss': 2.0 / epoch, 'micro_accuracy': 0.5, 'sample_count': 2},
                'validation': {'loss': 3.0 / epoch, 'micro_accuracy': 0.5, 'macro_accuracy': 0.5, 'sample_count': 2}}
               for epoch in (1, 2)]
    with classifier._wandb_logging(args, tmp_path, {}, audited, {}, history[:1]):
        pass
    with pytest.raises(RuntimeError, match='training interrupted'):
        with classifier._wandb_logging(args, tmp_path, {}, audited, {}, history):
            raise RuntimeError('training interrupted')
    assert runs[0].kwargs['name'] == tmp_path.name
    assert runs[0].kwargs['resume'] == 'never'
    assert runs[1].kwargs['resume'] == 'allow'
    assert runs[1].id == runs[0].id
    assert runs[1].kwargs['entity'] == 'fixture-team'
    assert [step for step, _ in runs[0].logs] == [1]
    assert [step for step, _ in runs[1].logs] == [2]
    assert runs[0].finishes == [0]
    assert runs[1].finishes == [1]
    args.wandb_mode = 'disabled'
    with classifier._wandb_logging(args, tmp_path, {}, audited, {}, history) as run:
        assert run is None
    assert len(runs) == 2


def test_actual_resnet_epoch_resume_and_explicit_test_evaluation(tmp_path):
    python = os.environ.get('QUICKDRAW_METRIC_PYTHON')
    if python is None and importlib.util.find_spec('torch') is not None:
        python = sys.executable
    if python is None:
        pytest.skip('Actual ResNet training/resume needs isolated Torch interpreter via QUICKDRAW_METRIC_PYTHON')
    has_wandb = subprocess.run([python, '-c', 'import importlib.util; raise SystemExit(importlib.util.find_spec("wandb") is None)'])
    if has_wandb.returncode:
        pytest.skip('Actual ResNet logging/resume check needs W&B in the isolated Torch interpreter')
    root = _cache(tmp_path / 'cache')
    script = str(Path(classifier.__file__))
    common = [python, script, '--cache-root', str(root), '--batch-size', '2', '--workers', '0', '--threads', '1', '--device', 'cpu', '--seed', '8']
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    uninterrupted = subprocess.run(common + ['--output', str(full), '--epochs', '2'], check=True, capture_output=True, text=True)
    first_epoch = subprocess.run(common + ['--output', str(resumed), '--epochs', '1'], check=True, capture_output=True, text=True)
    # Logging can be enabled on continuation and must recover the saved first epoch.
    logged = subprocess.run(common + ['--output', str(resumed), '--epochs', '2', '--resume',
                                     '--wandb-project', 'quickdraw-classifier-test', '--wandb-mode', 'offline'],
                            check=True, capture_output=True, text=True)
    for completed in (uninterrupted, first_epoch, logged):
        assert 'per_class_accuracy' not in completed.stdout
    assert not (full / 'wandb_run.json').exists()
    assert (resumed / 'wandb_run.json').exists()
    rows = [json.loads(line) for line in (resumed / 'metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in rows] == [1, 2]
    assert all(set(row['validation']['per_class_accuracy']) == {'0', '1'} for row in rows)
    compare = '''
import json, sys, torch
import numpy as np
from pathlib import Path
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

left = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
right = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
assert left['global_step'] == right['global_step'] == 2
assert left['history'] == right['history']
assert left['next_epoch'] == right['next_epoch'] == 2
for key in left['model']:
    assert torch.equal(left['model'][key], right['model'][key]), key
assert left['rng']['python'] == right['rng']['python']
assert left['rng']['numpy'][0] == right['rng']['numpy'][0]
assert np.array_equal(left['rng']['numpy'][1], right['rng']['numpy'][1])
assert left['rng']['numpy'][2:] == right['rng']['numpy'][2:]
assert torch.equal(left['rng']['torch_cpu'], right['rng']['torch_cpu'])
assert len(left['rng']['torch_cuda']) == len(right['rng']['torch_cuda'])
for left_rng, right_rng in zip(left['rng']['torch_cuda'], right['rng']['torch_cuda']):
    assert torch.equal(left_rng, right_rng)
for key in left['rng']['data']:
    assert torch.equal(left['rng']['data'][key], right['rng']['data'][key]), key
assert left['optimizer']['param_groups'] == right['optimizer']['param_groups']
for key, values in left['optimizer']['state'].items():
    for name, value in values.items():
        assert torch.equal(value, right['optimizer']['state'][key][name]), name

files = list(Path(sys.argv[2]).parent.glob('wandb/offline-run-*/run-*.wandb'))
assert len(files) == 1, files
store = DataStore()
store.open_for_scan(str(files[0]))
logged_history = []
while (data := store.scan_data()) is not None:
    record = wandb_internal_pb2.Record()
    record.ParseFromString(data)
    if record.HasField('history'):
        logged_history.append({item.key or '.'.join(item.nested_key): json.loads(item.value_json)
                               for item in record.history.item})
assert [row['epoch'] for row in logged_history] == [1, 2], logged_history
for saved, logged in zip(right['history'], logged_history):
    assert logged['global_step'] == saved['global_step']
    for split, names in [('train', ['loss', 'micro_accuracy']),
                         ('validation', ['loss', 'micro_accuracy', 'macro_accuracy'])]:
        for name in names:
            assert logged[f'{split}/{name}'] == saved[split][name]
    assert not any('per_class_accuracy' in key for key in logged)
'''
    subprocess.run([python, '-c', compare, str(full / 'last_training.pt'), str(resumed / 'last_training.pt')], check=True)
    assert not (full / 'evaluation_test.json').exists()
    provenance = json.loads((full / 'provenance.json').read_text())
    assert provenance['checkpoint_role'] == 'newly_trained_evaluator_v1'
    assert not provenance['test_used_for_checkpoint_selection']
    assert not provenance['smoke_only']
    assert provenance['extractor_sha256'] == classifier.sha256(full / 'resnet18_best.pt')
    evaluated = subprocess.run(common + ['--output', str(full), '--evaluate-only', '--split', 'test'],
                               check=True, capture_output=True, text=True)
    assert 'per_class_accuracy' not in evaluated.stdout
    assert json.loads((full / 'evaluation_test.json').read_text())['sample_count'] == 2
    # A diagnostic limiter is a scientific configuration change, not continuation.
    failed = subprocess.run(common + ['--output', str(resumed), '--epochs', '3', '--resume', '--smoke-batches', '1'], capture_output=True, text=True)
    assert failed.returncode != 0
    assert 'identical cache, scientific config' in failed.stderr
