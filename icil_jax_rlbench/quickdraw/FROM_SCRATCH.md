# Download, prepare, and train a fresh QuickDraw evaluator

Preparation, full classifier training, and policy embedding extraction have
completed on this machine. Continue with [FULL_DATASET.md](FULL_DATASET.md) for
the current supervised Transformer experiments. The steps below document how
to reproduce the original data/classifier preparation.

The user requested a new evaluator rather than historical checkpoint recovery.
This workflow trains one grayscale ResNet18 from random initialization, then
freezes it for the entire policy comparison. It retains the donor architecture
and renderer, but its feature scores define a new checkpoint-specific metric.

All commands assume Bash on this machine. Work from `/home/rvalperga/icil`.
The existing source checkout `/home/rvalperga/quick-robot-draw` supplies the
canonical rasterizer; its legacy dataset loaders and classifier scripts are
not used. No original checkpoint, FAISS index, or precomputed embedding is
needed for these steps. Existing synthetic outputs need not be deleted.

The [neighborhood addendum](../../QUICK_ROBOT_DRAW_ICIL_JAX_NEIGHBORHOOD_CORRECTION.md)
now permits this frozen classifier for offline task construction as well as
evaluation. Reuse the caches prepared below; after full classifier training,
continue with [NEIGHBORHOODS.md](NEIGHBORHOODS.md) for embeddings, split-local
cosine indexes, and A-NN/A-local manifests. A two-batch smoke checkpoint remains
a diagnostic: start full training in a fresh output directory using the same
rasters, without `--resume` or `--smoke-batches`.

Local validation: 128 repository tests passed, with only the optional historical
checkpoint parity test skipped. The new ResNet trainer passed exact CPU resume
checks and an actual GPU train/resume fixture. A trained fixture checkpoint
produced exactly the same 512D features through the donor constructor and the
offline worker (16 sketches, maximum feature difference 0). This validates the
new boundary with identical new weights; it is not a real-data accuracy result.

## 1. Separate evaluator environment

This environment has already been created locally. To reproduce it on a fresh
machine, use the pinned donor-compatible PyTorch pair and dependencies:

```bash
cd /home/rvalperga/icil
test -x .venv-quickdraw-metrics/bin/python || \
  uv venv --python 3.11 .venv-quickdraw-metrics
uv pip install --python .venv-quickdraw-metrics/bin/python \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-quickdraw-metrics/bin/python \
  numpy==2.3.4 pillow==12.0.0 scipy==1.16.3 wandb==0.22.3

.venv-quickdraw-metrics/bin/python -c \
  'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

CUDA availability should be `True`. Do not install these dependencies into the
JAX `.venv`. See [the official wheel matrix](https://pytorch.org/get-started/previous-versions/#v291)
for the supported Torch/torchvision pair.

## 2. Download the official simplified vector drawings

The source is Google's [simplified NDJSON release](https://github.com/googlecreativelab/quickdraw-dataset#get-the-data),
one file for each of 345 categories. Preserve filenames and drawing IDs. The
following resumes partially downloaded category files; it downloads the full
source collection, even though the first prepared cache below is bounded.

```bash
cd /home/rvalperga/icil
mkdir -p datasets/quickdraw_raw
curl -fsSL \
  https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt \
  -o datasets/quickdraw_raw/categories.txt

while IFS= read -r category || [ -n "$category" ]; do
  wget -c \
    "https://storage.googleapis.com/quickdraw_dataset/full/simplified/${category// /%20}.ndjson" \
    -O "datasets/quickdraw_raw/${category}.ndjson" || break
done < datasets/quickdraw_raw/categories.txt

python3 - <<'PY'
from pathlib import Path
root = Path('datasets/quickdraw_raw')
categories = root.joinpath('categories.txt').read_text().splitlines()
assert len(categories) == 345
missing = [c for c in categories if not root.joinpath(c+'.ndjson').is_file()
           or root.joinpath(c+'.ndjson').stat().st_size == 0]
assert not missing, missing
print('All 345 category files are present; confirm wget finished successfully.')
PY
```

If a download fails, rerun the loop; file presence alone does not prove a
partially downloaded file is complete. Use `full/simplified`, not bitmap NPY or
Sketch-RNN NPZ: the new sampler needs original IDs and vector stroke boundaries.

## 3. Build the bounded neutral source cache

```bash
cd /home/rvalperga/icil
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw prepare \
  --raw-root datasets/quickdraw_raw \
  --output datasets/quickdraw_source_cache_v1 \
  --max-points 127 --max-drawings-per-category 20000
```

This inspects at most the first 20,000 source records per category before
eligibility filtering, preserving that prefix choice and its hashes in the
provenance. It is a bounded initial source population, not a uniform sample of
the complete release. Drawings longer than 127 points are excluded and counts
are reported. Recognized and unrecognized drawings are both retained when
eligible. This matches the initial 128-event policy budget including STOP.
Change this setting only when creating a separately named cache/experiment.

## 4. Separate drawing pools and render classifier images

```bash
.venv-quickdraw-metrics/bin/python -m icil_jax_rlbench.quickdraw.classifier_data \
  --cache-root datasets/quickdraw_source_cache_v1 \
  --donor-root /home/rvalperga/quick-robot-draw \
  --output datasets/quickdraw_prepared_v1 \
  --train-per-category 2000 --validation-per-category 200 \
  --test-per-category 200 --policy-per-category 5000 --seed 37
```

Progress is enabled by default: terminal bars show completed counts, rate, and
ETA for duplicate grouping, splitting, and rendering. Stage messages identify
cache loading, compression, disk flushing, and hashing. Redirected logs receive
periodic lines instead of terminal redraws. Use `--no-progress` to silence these
messages; the completed raster-cache path is still printed to stdout.

This freezes four mutually disjoint drawing/duplicate-group pools. The first
three train, select, and independently assess the evaluator. The fourth supplies
the later JAX policies and their real-reference manifests. The evaluator can
learn all category names, but none of its train/validation/test drawing IDs
enters the policy pool. Cross-category duplicate clusters are excluded. A
category with insufficient unique eligible drawings receives proportionally
reduced allocations; inspect the recorded actual counts, especially simple
categories such as line. Preparation fails if a category cannot support the
minimum split sizes.

The requested policy pool is now 5,000 sketches per category, up to 1,725,000
distinct drawings across all 345 categories. The ResNet allocations remain
2,000/200/200, so preparation needs 7,400 unique eligible drawings per category
to fill every cap. The 20,000-record source-prefix cap provides room for
filtering and deduplication; actual counts can still be lower. With all
categories eligible, the default policy split has 243 training categories.
Reserving eight metric-reference drawings per category leaves up to
243 * 4,992 = 1,213,056 policy-training drawings before any F/U run-specific
budget. Set those budgets explicitly; the small F=8/U=4,096 pilot remains small.

If an earlier immutable cache has already been prepared, create a separately
named source/prepared cache (for example `_v2`) with the new settings instead
of overwriting it. Reuse the completed raw downloads.

The resulting layout is:

```text
datasets/quickdraw_prepared_v1/
  rasters/
    images.npy          # float32 [N,64,64], read using memory mapping
    labels.npy          # class indices
    splits.npy          # train / validation / test assignments
    records.jsonl       # original IDs, categories, duplicate groups, split
    label_map.json      # exact 345-class mapping
    manifest.json       # content hashes, counts, renderer, source provenance
  policy_cache/
    records.npz
    index.json
```

With full caps, raster storage is about 13.6 GB in decimal units for 828,000
float32 images. Images are rendered one at a time into memory-mapped arrays.
The input vector cache and record metadata still use host memory.

Rendering is exactly 64x64 grayscale, antialias factor 2, width 2, background 0,
stroke 1, and no extra input normalization. Training uses the same convention
as the offline feature extractor. No ImageNet resizing/normalization or image
augmentation is applied.

## 5. Smoke-test actual training

```bash
OMP_NUM_THREADS=4 .venv-quickdraw-metrics/bin/python \
  icil_jax_rlbench/quickdraw/classifier.py \
  --cache-root datasets/quickdraw_prepared_v1/rasters \
  --output outputs/quickdraw_evaluator/resnet18_smoke \
  --epochs 1 --batch-size 64 --workers 0 --device cuda --smoke-batches 2
```

Check that both training and validation finish with finite losses and that
checkpoints and metrics are written. This intentionally short run is labeled
as a smoke test and must not become the experiment's evaluator.

## 6. Train the fresh evaluator

The classifier logs directly to W&B when `--wandb-project` is provided. Install
the logging dependency in the classifier environment and authenticate once if
needed:

```bash
uv pip install --python .venv-quickdraw-metrics/bin/python wandb==0.22.3
.venv-quickdraw-metrics/bin/wandb login
```

Start a new training run with the command below. On this machine,
`outputs/quickdraw_evaluator/resnet18_v1` already has two completed full epochs:
add `--resume` to this command to continue from epoch 3 and log its saved
history to W&B.

```bash
OMP_NUM_THREADS=4 .venv-quickdraw-metrics/bin/python \
  icil_jax_rlbench/quickdraw/classifier.py \
  --cache-root datasets/quickdraw_prepared_v1/rasters \
  --output outputs/quickdraw_evaluator/resnet18_v1 \
  --epochs 10 --batch-size 256 --lr 0.001 --workers 4 --seed 42 --device cuda \
  --wandb-project icil-quickdraw
```

The run name defaults to the output directory name, `resnet18_v1`. Override it
with `--wandb-name`, or choose a team with `--wandb-entity`. The trainer plots
`train/loss`,
`train/micro_accuracy`, `validation/loss`, `validation/micro_accuracy`, and
`validation/macro_accuracy` against epoch. Accuracy values are fractions from
0 to 1. Metrics are available once per epoch, after validation and checkpoint
saving; there are no batch-level curves.

The console prints a compact epoch summary. The long list previously printed
after each epoch was the per-category accuracy mapping, with one entry for each
of the 345 classes. Detailed per-category results remain in `metrics.jsonl`,
and confusion matrices remain on disk. Use `--wandb-mode offline` to record
metrics locally, or omit `--wandb-project` / use `--wandb-mode disabled` to
disable W&B. Checkpoints and dataset images are not uploaded.

The model uses `resnet18(weights=None)`, one input channel and 345 outputs.
Optimization uses categorical cross-entropy and Adam. No pretrained weights are
downloaded. Ten epochs is an initial training budget, not an accuracy guarantee.
Assess validation loss and macro/per-category accuracy before selecting a
final budget. If validation is still improving, extend the same run explicitly:

```bash
OMP_NUM_THREADS=4 .venv-quickdraw-metrics/bin/python \
  icil_jax_rlbench/quickdraw/classifier.py \
  --cache-root datasets/quickdraw_prepared_v1/rasters \
  --output outputs/quickdraw_evaluator/resnet18_v1 \
  --epochs 20 --batch-size 256 --lr 0.001 --workers 4 --seed 42 --device cuda \
  --resume --wandb-project icil-quickdraw
```

Resume restores the last completed epoch, optimizer and random states; an
interrupted partial epoch is rerun. Training writes plain model state dicts for
the best validation checkpoint and latest checkpoint, plus a separate resumable
training snapshot. Use validation only to choose the checkpoint/training budget.
Logging can be enabled when resuming an existing run that started without W&B:
add `--wandb-project` to its resume command. Saved epochs are restored to W&B,
and later online resumes continue the same W&B run. To resume an interrupted
10-epoch run, keep `--epochs 10`; increasing the value extends the budget.

## 7. Assess once, then freeze the evaluator

After choosing the final model using validation, evaluate the reserved classifier
test drawings:

```bash
OMP_NUM_THREADS=4 .venv-quickdraw-metrics/bin/python \
  icil_jax_rlbench/quickdraw/classifier.py \
  --cache-root datasets/quickdraw_prepared_v1/rasters \
  --output outputs/quickdraw_evaluator/resnet18_v1 \
  --evaluate-only --split test --device cuda --batch-size 256 --workers 4
```

Retain `resnet18_best.pt`, `provenance.json`, the raster manifest/record lists,
and `label_map.json` together. The provenance attests the exact checkpoint and
label-map hashes. If the result reveals insufficient competence, report it;
do not tune repeatedly against this reserved test set.

Use this one frozen checkpoint for every policy architecture, diversity level,
seed, and support control. New policy manifests must be built from
`datasets/quickdraw_prepared_v1/policy_cache`, not the mixed source cache.

The existing feature command can now use:

```text
--python /home/rvalperga/icil/.venv-quickdraw-metrics/bin/python
--extractor-checkpoint /home/rvalperga/icil/outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt
--label-map /home/rvalperga/icil/datasets/quickdraw_prepared_v1/rasters/label_map.json
--checkpoint-provenance /home/rvalperga/icil/outputs/quickdraw_evaluator/resnet18_v1/provenance.json
```

Its 512D pooled output is the embedding. Keep it unnormalized for Sketch-FD and
Sketch-MMD. The new offline embedding exporter also saves a separate normalized
copy for A-NN/A-local cosine retrieval. Their task manifests disclose the fixed
classifier's curation role, including its all-category supervision. The JAX
policy receives trajectories only. Follow [the neighborhood workflow](NEIGHBORHOODS.md).
