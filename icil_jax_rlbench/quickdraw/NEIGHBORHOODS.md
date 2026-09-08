# Neighborhood tasks: continuation from the prepared dataset

For the user's current first experiment, follow [FULL_DATASET.md](FULL_DATASET.md):
supervised autoregressive and diffusion Transformers across all 345 categories,
with no meta-learning. Its compact full-data resource replaces the bounded
manifest/index examples below for those runs. This guide retains the separate
A-local diversity sweeps and TTT controls.

This implements the [neighborhood correction](../../QUICK_ROBOT_DRAW_ICIL_JAX_NEIGHBORHOOD_CORRECTION.md)
on the user's `quick-robot-draw` branch. A-NN is a primary query-centered
similarity protocol, A-local provides neighborhood-count sweeps at fixed coarse
categories, and A-category is the independent same-category ablation.

## What can be reused

Keep the raw NDJSON, `datasets/quickdraw_source_cache_v1`, and
`datasets/quickdraw_prepared_v1`. The completed split has 5,000 policy drawings
per category and 2,000/200/200 classifier train/validation/test drawings per
category. The classifier/policy separation remains valid under the correction.

The selected `outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt` is frozen,
and the full policy embedding export has completed. The current policy workflow
reuses both. See [FROM_SCRATCH.md](FROM_SCRATCH.md#6-train-the-fresh-evaluator) for
classifier training reproduction and resume commands. The separate
`resnet18_smoke` checkpoint is still a
two-update diagnostic and cannot resume as a full run. Reuse the existing
rasters. Select the checkpoint using validation, assess the held-out
classifier test split, then freeze the checkpoint for all policy comparisons.
The correction does not require another ResNet architecture or retraining a
completed competent classifier.

Old immutable caches/checkpoints may say `evaluation_only` in their creation
provenance. New embedding/index/task metadata explicitly disclose the additional
fixed offline curation role. Those old artifacts are not rewritten. Classifier
weights, embeddings, neighbor scores, anchors, and target-derived length remain
outside JAX policy inputs and WRITE targets.

## 1. Export frozen features

Run from `/home/rvalperga/icil` after full classifier training. FAISS is installed
only in the isolated metric environment; to reproduce that dependency:

```bash
uv pip install --python .venv-quickdraw-metrics/bin/python 'faiss-cpu==1.15.0'
```

Export the independent policy pool using the selected checkpoint:

```bash
OMP_NUM_THREADS=4 .venv-quickdraw-metrics/bin/python \
  -m icil_jax_rlbench.quickdraw.embeddings extract \
  --cache-root datasets/quickdraw_prepared_v1/policy_cache \
  --donor-root /home/rvalperga/quick-robot-draw \
  --extractor-checkpoint outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt \
  --checkpoint-provenance outputs/quickdraw_evaluator/resnet18_v1/provenance.json \
  --output datasets/quickdraw_embeddings_v1 \
  --batch-size 256 --device cuda
```

This streams canonical 64x64 raster batches and writes `raw.npy`, `cosine.npy`,
`records.jsonl`, and a hashed `manifest.json`. Raw 512D vectors retain the
Sketch-FD convention. Cosine vectors are a separate L2-normalized copy. At
1,725,000 drawings, the two float32 arrays occupy about 7.07 GB combined, plus
the ID mapping. Progress is enabled by default. Known smoke checkpoints require
an explicit fixture override and must not curate a scientific run.

## 2. Freeze splits and tasks before exporting indexes

The default manifest protocol is A-NN. This bounded example fixes eight coarse
categories, a 4,096-drawing available training reservoir, 128 eligible training
targets, and 32 evaluation targets in each development/test split:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw manifest \
  --cache-root datasets/quickdraw_prepared_v1/policy_cache \
  --embedding-root datasets/quickdraw_embeddings_v1 \
  --output outputs/quickdraw_neighborhoods/a_nn.json \
  --a-protocol a_nn --split-regime familiar_drawings \
  --family-count 8 --neighborhood-count 128 \
  --evaluation-neighborhood-count 32 --unique-budget 4096 \
  --selection-mode exact_top_k --top-m 32 --max-points 127
```

The policy's support count selects K from the stored ranking. `exact_top_k`
takes the closest K eligible drawings; `sample_top_m` samples distinct supports
from the declared top-M pool using the saved sampler RNG. A-NN uses one query
target per task (`query_count=1`). The selected target and its duplicate group
are excluded from supports. Its neighborhood-count budget also limits distinct
training targets; this is disclosed, not treated as an isolated diversity test.

Split regimes are separate experiments:

- `familiar_drawings`: split drawing/duplicate groups within a fixed category
  set, then construct tasks separately inside each pool.
- `heldout_regions`: partition complete deterministic farthest-first cosine
  Voronoi cells, then curate within the resulting pools. `--region-count` fixes
  this approximate geometric construction; separation is measured, not assumed.
- `heldout_categories`: keep complete categories disjoint, then construct
  tasks within them. Evaluation task caps must be divisible by each evaluated
  category count; for the full 345-category data the default 51/51 held-out
  category sets can use `--evaluation-neighborhood-count 102`.

The manifest records IDs, duplicates, references, anchor pools, fixed category
coverage, seeds, ranking/tie rules, feature/index hashes, overlap, dispersion,
and train/held-out proximity. Infeasible quotas or incoherent neighborhoods
fail explicitly. Eligibility beyond an explicit construction cap is unknown;
the manifest separately reports candidate and materialized task counts.

The numerical manifest builder performs exact cosine curation using NumPy.
Export matching split/category FAISS indexes for inspection or offline reuse:

```bash
.venv-quickdraw-metrics/bin/python \
  -m icil_jax_rlbench.quickdraw.embeddings build-index \
  --embedding-root datasets/quickdraw_embeddings_v1 \
  --manifest outputs/quickdraw_neighborhoods/a_nn.json \
  --output datasets/quickdraw_indexes_a_nn_v1
```

Every index has an aligned ID mapping and hashes. Held-out drawings cannot enter
training searches. The JAX learner reads the curated numerical manifest and
trajectories; it imports neither FAISS nor Torch. A named exact-ID-only search
mode is retained for legacy retrieval parity; scientific task manifests exclude
duplicate clusters as well.

## 3. Validate fine-grained evaluation before scientific pilots

Freeze evaluation episodes with `quickdraw episodes`, pointing `--manifest-path`
to the new manifest and using the same support count and maximum length as the
policy:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw episodes \
  --cache-root datasets/quickdraw_prepared_v1/policy_cache \
  --manifest-path outputs/quickdraw_neighborhoods/a_nn.json \
  --output outputs/quickdraw_neighborhoods/a_nn_development \
  --split development --tasks 32 --support-count 4 --query-count 1 \
  --max-steps 128 --seed 0
```

A-NN/A-local also freeze `same_category_wrong_neighborhood` supports,
requiring centroid cosine distance at least 0.05 and member Jaccard at most 0.1,
then approximately matching length and stroke complexity. Increase the declared
evaluation task pool if no qualifying negative exists. Keep any changed
separation threshold explicit and fixed across comparisons.

Export and extract both reference halves. Their IDs are fixed before sampling
supports and globally disjoint:

```bash
for reference_half in real_a real_b; do
  uv run --frozen --group metaworld --extra cuda12 --extra wandb \
    python -m icil_jax_rlbench.quickdraw references \
    --cache-root datasets/quickdraw_prepared_v1/policy_cache \
    --manifest-path outputs/quickdraw_neighborhoods/a_nn.json \
    --episodes outputs/quickdraw_neighborhoods/a_nn_development \
    --output "outputs/quickdraw_neighborhoods/$reference_half" \
    --split development --half "$reference_half" --max-steps 128

  uv run --frozen --group metaworld --extra cuda12 --extra wandb \
    python -m icil_jax_rlbench.quickdraw.metrics extract \
    "outputs/quickdraw_neighborhoods/$reference_half" \
    --output "outputs/quickdraw_neighborhoods/${reference_half}_features" \
    --python /home/rvalperga/icil/.venv-quickdraw-metrics/bin/python \
    --donor-root /home/rvalperga/quick-robot-draw \
    --extractor-checkpoint outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt \
    --label-map datasets/quickdraw_prepared_v1/rasters/label_map.json \
    --checkpoint-provenance outputs/quickdraw_evaluator/resnet18_v1/provenance.json
done
```

Use two sufficiently separated same-category task IDs from the saved episode
metadata (`same_category_wrong_pairings` identifies the paired tasks):

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.metrics swap-check \
  outputs/quickdraw_neighborhoods/real_a_features \
  outputs/quickdraw_neighborhoods/real_b_features \
  --left-task TASK_ID_1 --right-task TASK_ID_2 \
  --output outputs/quickdraw_neighborhoods/swap_check.json
```

The gate checks disjoint real populations, unchanged pooled FD after swapping
the two generated task populations, and worse neighborhood-conditional MMD.
Run it on real drawings with the fully trained frozen representation before
claiming fine-grained support use. Synthetic/undertrained-fixture checks are
software validation only.

Generated artifacts retain stochastic outputs, failures, STOP behavior, actual
counts, original targets, intended neighborhoods, references and paired keys.
`summary.json` includes held-out autoregressive target loss for A-NN. `metrics
score` adds macro neighborhood MMD to pooled FD and category metrics. Confidence
intervals group shared members/references; fully connected overlap provides no
independent neighborhood bootstrap interval. Generated samples from one prompt
are not independent training replicates.

Use the existing population exporter and `metrics copying`, plus the new
`metrics geometry-copying GENERATED SUPPORTS TRAINING --manifest-path MANIFEST
--output JSON`, for non-classifier raw geometry, stroke/length, and copy checks.
`support_copy`, `nearest_training_example`, galleries, and actual-update analysis
remain separate diagnostics. Analysis now reports neighborhood functional
diversity within categories as well as across categories.

## 4. Explicit bounded pilots

After the retrieval/information/conditional metric gates, run the matched A-NN
pilot with TTT, explicit context, and independent no-support. It also trains
the A-category ablation with the same target IDs and reference sets:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw pilot \
  --cache-root datasets/quickdraw_prepared_v1/policy_cache \
  --embedding-root datasets/quickdraw_embeddings_v1 \
  --output outputs/quickdraw_neighborhoods/pilot_nn_v1 \
  --a-protocol a_nn --family-count 8 --levels 128 \
  --evaluation-neighborhood-count 32 --unique-budget 4096 \
  --steps 50 --max-steps 128
```

Then use A-local for the N contrast, retaining the exact category set and all
development/test neighborhoods across the two levels:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw pilot \
  --cache-root datasets/quickdraw_prepared_v1/policy_cache \
  --embedding-root datasets/quickdraw_embeddings_v1 \
  --output outputs/quickdraw_neighborhoods/pilot_local_v1 \
  --a-protocol a_local --family-count 8 --levels 32 256 \
  --neighborhood-size 64 --evaluation-neighborhood-count 32 \
  --unique-budget 4096 --steps 50 --max-steps 128
```

These are explicit initial budgets, not evidence of adequate optimization.
Use `--config icil_jax_rlbench/configs/quickdraw_smoke.py` for a smaller capacity
profile, then choose capacity/budget before a matched scientific comparison.
Pilots cap steps at 1,000. They compare all manifests and frozen arrays before
launching training. Optimizer steps and the available reservoir are matched;
actual valid-token and unique-drawing exposure can differ and are reported.

For the separate diversity-plus-data regime, use A-local with
`--budget-regime per_neighborhood` and omit `--unique-budget`. Each selected
neighborhood still has exactly `--neighborhood-size` members, while the allowed
training examples become the union of those memberships. The manifest records
both N times the allowance and the actual unique union after overlap. Curation
uses the declared full construction pool; it is a separate regime from the
capped fixed-reservoir pilot. This changes the data budget as well as N.

For exposure-only controls, reuse the exact same immutable manifest with
`train`, increasing `num_steps` or resuming the checkpoint. Raising the drawing
budget or changing neighborhood size rebuilds tasks and is not an exposure-only
comparison. The pilots report token mismatches rather than claiming an exact
valid-token-matched intervention.

`train` can consume any new immutable manifest directly. `exposure.json` and
checkpoint metadata record neighborhood/category frequencies, support/query
reuse, support-target pairs, unique drawings, and sampled versus consumed
events. Fixed N with more steps tests exposure effects. Changing neighborhood
size changes difficulty and must be a separately named experiment. Overlap
means N is a count of permitted neighborhoods, not independent latent tasks.
B1/B2 retain their original instance/program splits and control behavior.

## Validation status

Local checks cover exact donor feature/retrieval parity, split-local FAISS,
duplicate exclusions, deterministic top-M sampling, region membership,
fixed-category nested N sets, metadata-only identity invariance, conditional
swap sensitivity, non-classifier copying controls, and real JAX TTT/resume on
explicit synthetic fixtures. The saved offline retrieval audit is
`outputs/quickdraw_smoke/neighborhood_retrieval/report.json`.

The complete CPU test suite passed on 2026-09-08 after the direct classifier
logging update: 158 passed, one optional
original-evaluator checkpoint parity test skipped. Compilation, forbidden-import
checks, and Git whitespace checks passed. Actual CLI manifest/index exports and
both pilot preparation paths were checked on fixtures; the preparation checks
mocked training/generation, while the separate TTT/resume tests ran real JAX.

The real-data classifier has completed two full training epochs in
`outputs/quickdraw_evaluator/resnet18_v1` and can resume from that checkpoint.
Full feature extraction, real-set conditional gate results, and scientific
pilots remain to be run after classifier training and model selection. No
threshold or held-out adaptation claim follows from the software checks.
