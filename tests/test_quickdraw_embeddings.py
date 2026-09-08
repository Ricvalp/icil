from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from icil_jax_rlbench.quickdraw import embeddings
from icil_jax_rlbench.quickdraw.data import SketchRecord, SketchStore


def _python():
    python = os.environ.get('QUICKDRAW_METRIC_PYTHON')
    if python is None and importlib.util.find_spec('faiss') and importlib.util.find_spec('torch'):
        python = sys.executable
    if python is None:
        pytest.skip('Offline extraction/index check needs isolated QUICKDRAW_METRIC_PYTHON with Torch and FAISS.')
    return python


def _features(root):
    root.mkdir()
    raw = np.random.default_rng(1).normal(size=(24, 512)).astype(np.float32)
    # Same declared geometry cluster, with slightly different feature vectors.
    records = [{'row_index': i, 'base_id': str(100 + i), 'category': 'cat',
                'duplicate_cluster_id': f'cluster-{i if i != 1 else 0}'} for i in range(24)]
    raw[1] = raw[0] + 0.001
    np.save(root / 'raw.npy', raw)
    np.save(root / 'cosine.npy', raw / np.linalg.norm(raw, axis=1, keepdims=True))
    (root / 'records.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
    manifest = {'embedding_version': embeddings.EMBEDDING_VERSION,
                'actual_count': len(records), 'cache_identifier': 'fixture-cache',
                'raw_normalization': 'none', 'cosine_normalization': 'l2',
                'extractor_sha256': 'fixture-extractor',
                'files': {name: embeddings.file_sha256(root / name) for name in ('raw.npy', 'cosine.npy', 'records.jsonl')}}
    manifest['identifier'] = embeddings._identifier(manifest)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return embeddings.EmbeddingStore.open(root)


def _manifest(features):
    ids = [record['base_id'] for record in features.records]
    manifest = {'cache_identifier': features.cache_id, 'feature_identifier': features.identifier,
                'a_ids': {'train': {'cat': ids[:16]}, 'development': {'cat': ids[16:20]}, 'test': {'cat': ids[20:]}}}
    manifest['identifier'] = embeddings._identifier(manifest)
    return manifest


def test_feature_store_rejects_hash_corruption_and_mixed_conventions(tmp_path):
    features = _features(tmp_path / 'features')
    assert isinstance(features.raw, np.memmap)
    assert features.cache_id == 'fixture-cache'
    assert features.row('105') == 5
    path = features.root / 'cosine.npy'
    cosine = np.load(path).copy()
    cosine[3] = features.raw[3]
    np.save(path, cosine)
    with pytest.raises(ValueError, match='file hash mismatch'):
        embeddings.EmbeddingStore.open(features.root)
    with pytest.raises(ValueError, match='does not normalize'):
        embeddings.EmbeddingStore.open(features.root, verify_hashes=False)


def test_candidate_pool_checks_split_cluster_and_feature_identity(tmp_path):
    features = _features(tmp_path / 'features')
    manifest = _manifest(features)
    assert embeddings._candidate_pools(features, manifest) == manifest['a_ids']
    bad = deepcopy(manifest)
    bad['a_ids']['test']['cat'].append(bad['a_ids']['train']['cat'].pop(1))
    bad['identifier'] = embeddings._identifier(bad)
    with pytest.raises(ValueError, match='cross split'):
        embeddings._candidate_pools(features, bad)
    bad = deepcopy(manifest)
    bad['feature_identifier'] = 'another-feature-extractor'
    bad['identifier'] = embeddings._identifier(bad)
    with pytest.raises(ValueError, match='different frozen features'):
        embeddings._candidate_pools(features, bad)
    # References are split-local but are still forbidden index candidates.
    bad = deepcopy(manifest)
    bad['split_pools'] = {split: {'cat': {'example_ids': ids['cat'][:], 'reference_ids': [], 'anchor_ids': []}}
                         for split, ids in bad['a_ids'].items()}
    bad['split_pools']['train']['cat']['example_ids'].remove('101')
    bad['a_ids']['train']['cat'].remove('101')
    bad['split_pools']['train']['cat']['reference_ids'].append('115')
    bad['split_pools']['train']['cat']['example_ids'].remove('115')
    bad['identifier'] = embeddings._identifier(bad)
    with pytest.raises(ValueError, match='excluding references and anchors'):
        embeddings._candidate_pools(features, bad)


def test_embedding_reader_does_not_import_policy_torch_or_faiss():
    subprocess.run([sys.executable, '-c', '''
import sys
from icil_jax_rlbench.quickdraw.embeddings import EmbeddingStore
assert not {'torch', 'torchvision', 'faiss', 'jax'} & set(sys.modules)
'''], check=True)


def test_smoke_checkpoint_requires_explicit_fixture_mode(tmp_path):
    checkpoint = tmp_path / 'resnet18_best.pt'
    checkpoint.write_bytes(b'fixture checkpoint, rejected before loading tensors')
    provenance = {'extractor_sha256': embeddings.file_sha256(checkpoint), 'smoke_only': True}
    (tmp_path / 'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='smoke/subset classifier'):
        embeddings._checkpoint_provenance(checkpoint, None, allow_subset_fixture=False)
    actual, digest = embeddings._checkpoint_provenance(checkpoint, None, allow_subset_fixture=True)
    assert actual == provenance and digest


def test_actual_faiss_donor_neighbors_duplicates_stochastic_and_split_exclusion(tmp_path):
    python = _python()
    features = _features(tmp_path / 'features')
    manifest = _manifest(features)
    path = tmp_path / 'task.json'
    path.write_text(json.dumps(manifest))
    donor = Path(os.environ.get('QUICKDRAW_DONOR_ROOT', '/home/rvalperga/quick-robot-draw'))
    if not (donor / 'dataset/episode_builder.py').is_file():
        pytest.skip('Donor source required for retrieval reproduction.')
    script = '''
import importlib.util, json, sys, types
from pathlib import Path
import faiss, numpy as np
from icil_jax_rlbench.quickdraw.embeddings import EmbeddingStore, build_indexes, search_index
root, manifest, output, donor = map(Path, sys.argv[1:])
features = EmbeddingStore.open(root)
build_indexes(root, manifest, output, progress=False)
index_metadata = json.loads((output / 'manifest.json').read_text())
assert index_metadata['indexed_candidate_count'] == 24
# Load only donor preprocessing/builder; avoid its dataset package side effects.
package = types.ModuleType('_quickdraw_retrieval_donor')
package.__path__ = [str(donor / 'dataset')]
sys.modules[package.__name__] = package
for name in ('preprocess', 'episode_builder'):
    fullname = package.__name__ + '.' + name
    spec = importlib.util.spec_from_file_location(fullname, donor / 'dataset' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
pre = sys.modules[package.__name__ + '.preprocess']
builder = sys.modules[package.__name__ + '.episode_builder']
ids = np.arange(100, 116)
index = faiss.IndexFlatL2(512)
index.add(np.asarray(features.cosine[:16]))
def fetch(category, item):
    absolute = np.asarray([[0., 0.], [float(int(item) - 100) / 16, 1.]], np.float32)
    return pre.ProcessedSketch(category, str(item), absolute, absolute, np.asarray([0, 1], np.float32), 2)
old = builder.EpisodeBuilderSimilar(fetch_family=lambda category: ids.tolist(), fetch_sketch=fetch,
    family_ids=['cat'], k_shot=4, faiss_indices={'cat': index}, ids={'cat': ids}, coordinate_mode='absolute')
for seed in range(5):
    episode = old.build_episode(family_id='cat', rng=np.random.RandomState(seed))
    query = str(episode.metadata['query_id'])
    result = search_index(output, features, split='train', category='cat', query_id=query, k=4,
                          exclusion_mode='legacy_exact_id_only')
    assert [row['base_id'] for row in result] == list(map(str, episode.metadata['prompt_ids']))
legacy = search_index(output, features, split='train', category='cat', query_id='100', k=4,
                      exclusion_mode='legacy_exact_id_only')
scientific = search_index(output, features, split='train', category='cat', query_id='100', k=4)
assert legacy[0]['base_id'] == '101'
assert '101' not in [row['base_id'] for row in scientific]
randomized = [search_index(output, features, split='train', category='cat', query_id='100', k=4,
                           selection_mode='sample_from_top_m', top_m=10, seed=seed) for seed in (3, 3, 4)]
assert randomized[0] == randomized[1] and randomized[0] != randomized[2]
assert len({row['duplicate_cluster_id'] for row in randomized[0]}) == 4
assert all(int(row['base_id']) < 116 for row in randomized[0])
try:
    search_index(output, features, split='train', category='cat', query_id='121', k=4)
except ValueError as error:
    assert 'declared index split' in str(error)
else:
    raise AssertionError('Held-out query crossed training index boundary')
print(json.dumps({'donor_episode_neighbor_parity_cases': 5, 'duplicate_exclusion': True,
                  'stochastic_reproducibility': True, 'split_safe': True}))
'''
    completed = subprocess.run([python, '-c', script, str(features.root), str(path),
                                str(tmp_path / 'indexes'), str(donor)], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)['donor_episode_neighbor_parity_cases'] == 5


def test_actual_frozen_extraction_matches_donor_raw_and_normalized_features(tmp_path):
    python = _python()
    donor = Path(os.environ.get('QUICKDRAW_DONOR_ROOT', '/home/rvalperga/quick-robot-draw'))
    if not (donor / 'metrics/resnet18.py').is_file():
        pytest.skip('Donor source required for frozen feature parity.')
    records = [SketchRecord(str(index), 'cat',
                            np.asarray([[-.8, -.7], [.3, -.7 + index * .1], [.7, .7]], np.float32),
                            np.asarray([0, 1, 1], np.float32), provenance={'synthetic_fixture': True}) for index in range(4)]
    cache = tmp_path / 'cache'
    SketchStore.write(cache, records, provenance={'kind': 'synthetic_correctness_fixture'})
    script = '''
import json, sys
from pathlib import Path
import numpy as np, torch, torchvision
from icil_jax_rlbench.quickdraw.embeddings import extract_embeddings, file_sha256
from icil_jax_rlbench.quickdraw.metrics_worker import load_source
from icil_jax_rlbench.quickdraw.metrics import RENDERER_CONFIG
from icil_jax_rlbench.quickdraw.data import SketchStore
cache, output, checkpoint, donor = map(Path, sys.argv[1:])
torch.set_num_threads(1)
torch.manual_seed(92)
model = torchvision.models.resnet18(weights=None, num_classes=345)
model.conv1 = torch.nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
torch.save(model.state_dict(), checkpoint)
checkpoint.with_name('provenance.json').write_text(json.dumps(
    {'extractor_sha256': file_sha256(checkpoint), 'smoke_only': True, 'renderer_config': RENDERER_CONFIG}))
features = extract_embeddings(cache, donor, checkpoint, output, device='cpu', threads=1,
                              batch_size=4, allow_subset_fixture=True, progress=False)
original = load_source(donor / 'metrics/resnet18.py', '_donor_embedding_parity_resnet')
renderer = load_source(donor / 'dataset/rasterize.py', '_donor_embedding_parity_renderer')
extractor = original.ResNet18FeatureExtractor(checkpoint).eval()
store = SketchStore.open(cache)
images = np.stack([renderer.rasterize_absolute_points(np.column_stack((record.absolute, record.incoming_pen)),
    config=renderer.RasterizerConfig(**RENDERER_CONFIG)) for record in store.records])
with torch.inference_mode():
    expected = extractor(torch.from_numpy(images[:, None]))
    normalized = torch.nn.functional.normalize(expected, dim=1).numpy()
np.testing.assert_array_equal(features.raw, expected.numpy())
np.testing.assert_allclose(features.cosine, normalized, atol=3e-8, rtol=3e-6)
assert features.manifest['extractor_training_provenance']['smoke_only']
assert features.manifest['allow_subset_fixture']
assert features.manifest['raw_normalization'] == 'none'
print(json.dumps({'raw_donor_max_abs_error': float(np.max(np.abs(features.raw - expected.numpy()))),
                  'cosine_donor_max_abs_error': float(np.max(np.abs(features.cosine - normalized)))}))
'''
    completed = subprocess.run([python, '-c', script, str(cache), str(tmp_path / 'features'),
                                str(tmp_path / 'fixture.pt'), str(donor)], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)['raw_donor_max_abs_error'] == 0.0
