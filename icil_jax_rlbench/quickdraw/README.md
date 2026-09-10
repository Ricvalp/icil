# QuickDraw diversity diagnostic

The current first policy experiment is [full-dataset supervised in-context
imitation](FULL_DATASET.md) with autoregressive and diffusion Transformers,
direct demonstration cross-attention, and no meta-learning. Its prepared data
cover all 345 categories. Follow that guide for current launch commands; the
bounded pilots and TTT commands below belong to separate experiments.

Full-data fast-weight Transformer experiments use either [KVB WRITE](KVB_TRANSFORMER.md)
or [support-BC WRITE](SUPPORT_BC_EXPERIMENTS.md), with full second-order gradients.
The support-BC guide provides six H200 jobs comparing WRITE objectives, inner-step
counts, and policy capacity under the same [35-class holdout](CLASS_HOLDOUT.md).

This implementation lives on `quick-robot-draw`, created from the merged main
line. It follows `QUICK_ROBOT_DRAW_ICIL_JAX_IMPLEMENTATION_PLAN.md`; its older
branch name does not apply. [Implementation decisions](IMPLEMENTATION_NOTES.md)
record the donor revision, boundaries, and remaining asset requirements.

The [8 September neighborhood correction](../../QUICK_ROBOT_DRAW_ICIL_JAX_NEIGHBORHOOD_CORRECTION.md)
is now the active A contract. Follow [NEIGHBORHOODS.md](NEIGHBORHOODS.md) for the
primary A-NN protocol, A-local fixed-category neighborhood sweeps, offline
feature/index preparation, and the new conditional controls. Random same-category
A-category is an ablation. The category-only examples below remain explicitly
labeled compatibility examples for that ablation and the unchanged B path.

The user has since chosen to train a fresh ResNet evaluator. Follow the
[from-scratch instructions](FROM_SCRATCH.md) for that workflow, including
separate evaluator/policy drawing pools and an explicitly versioned new
checkpoint. The original-checkpoint instructions below describe legacy
reproduction; the new workflow does not require retrieving those old weights.

The main TTT policy uses the existing functional KVB writer, learned fast
initialization/rates, full second-order gradients, and delta READ. A causal GRU
and autoregressive Gaussian-mixture coordinate distribution provide stochastic
sketch generation. Pen and STOP have separate Bernoulli losses. B1 predicts
absolute waypoints; B2 predicts bounded displacement commands from actual pen
states. Explicit context, independently trained no-support, supervised WRITE,
and FOMAML are named model/checkpoint types. There are no new dependencies in
the JAX environment and no simulator or donor training-loader imports.

## Data and assets

Keep the sibling layout:

```text
/home/rvalperga/
  icil/
    datasets/quickdraw_raw/       # official per-category NDJSON files
    datasets/quickdraw_cache/     # neutral numeric records and provenance
    outputs/quickdraw/            # manifests, runs, trajectories, metrics
  phi-mujoco/                     # existing editable dependency
  quick-robot-draw/               # original renderer/evaluator source
    metrics/checkpoints/resnet18_step40000.pt
```

Paths are configurable. Real QuickDraw data, the selected fresh evaluator,
all 1,725,000 policy embeddings, and the full split-local retrieval resource are
prepared. See [FULL_DATASET.md](FULL_DATASET.md) for their current paths and counts.
Synthetic fixtures exercise software only.
The importer accepts raw NDJSON with `key_id` and `drawing`; donor LMDB/retrieval
caches are not required or imported. Preparation and the neutral store currently
use host memory, so begin with bounded raw inputs when profiling large corpora.

Commands below preserve this machine's existing CUDA and W&B extras. For a CPU
environment omit the CUDA extra consistently. W&B is optional; the sketch runner
writes local JSON logs and provenance without requiring an account.

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw prepare \
  --raw-root datasets/quickdraw_raw --output datasets/quickdraw_cache --max-points 127

uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw manifest \
  --cache-root datasets/quickdraw_cache --output outputs/quickdraw/F8.json \
  --a-protocol a_category \
  --family-count 8 --unique-budget 4096 --reference-per-category 8 --max-points 127
```

Manifests freeze category and program splits, duplicate clusters, eligibility,
reservoir IDs, seeds, normalization, and two independent real-reference halves.
F is category diversity for this A-category ablation; the primary A-local sweep
holds F fixed and varies N. `--program-count` sets B's training-program
diversity. `--unique-budget` fixes A's total available drawings across F;
`--drawings-per-category` is the alternative fixed-per-category budget.
The sampler reports any additional exclusions needed for its static token
budget. Support/query episodes remain separate. B's familiar-category holdout
uses unseen base programs; `--b-partition heldout_category` selects the separate
category holdout. Test exports require an explicit `--allow-test`.

## Training and generation

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw train \
  --config icil_jax_rlbench/configs/quickdraw.py \
  --set cache_root=datasets/quickdraw_cache \
  --set manifest_path=outputs/quickdraw/F8.json \
  --set output_dir=outputs/quickdraw/F8 --set num_steps=100
```

`--set model_type=explicit_context`, `no_support`, `ttt_supervised_write`, or
`ttt_kvb_first_order` selects an explicitly separate comparator. Use
`--set experiment=b1` or `b2` for the motor-program tasks. Model dimensions,
segment length, and update controls are in `model.*`; keep them fixed across
diversity levels. Start with the small `configs/quickdraw_smoke.py` for profiling.
B2 usually needs a larger `model.max_steps` because it includes pen-up travel
and bounded integration steps. Its `model.motion_bound` must match the episode
preparation `--motion-bound` (default 0.1); mismatches are rejected.

The printed run directory contains `last.pkl`, configuration, immutable manifest,
provenance, metrics, and runtime accounting. Resume with the same train command
and `--set resume_path=PATH/last.pkl --set num_steps=200`. Steps are the final
target, not additional steps. Scientific settings, optimizer, JAX key, fixed
batch when applicable, and all sampler streams restore from the checkpoint;
only runtime/path settings can override them. Copied cache/manifest paths are
accepted only when content identifiers match. Task-adapted fast states are never
saved as global model state.

Freeze evaluation inputs before generating:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw episodes \
  --cache-root datasets/quickdraw_cache --manifest-path outputs/quickdraw/F8.json \
  --output outputs/quickdraw/episodes --tasks 8 --support-count 4 --max-steps 128

uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw generate \
  --checkpoint PATH/last.pkl --episodes outputs/quickdraw/episodes \
  --output outputs/quickdraw/generated --samples-per-task 4 --seed 17
```

Choose a task count divisible by the evaluated category count when comparing to
balanced real references. The bounded pilot does this automatically. Inputs and
query keys are paired across no-update, correct, wrong, and norm-matched random
fast-update controls. Support IDs and intended targets are distinct metadata.
Generation accepts no hidden query tokens, target masks, target lengths, or
offline embeddings/anchor identities. The offline A-NN curation step may use a
complete target to choose demonstrations, as explicitly allowed by the addendum.

B1 controls include `transformed_replay`, `untransformed_replay`, `oracle`, and
same-category `wrong_support`. `category_prototype` uses only the permitted
training reservoir and requires a familiar category. B2 adds `open_loop_replay`
and `feedback_replay`; `--perturbation-step 5 --perturbation-xy .1 -.1`,
`--noise-std`, and `--delay-steps` launch separately recorded disturbance runs.
Tracking uses executed positions against an unaligned timed reference. Missing
steps, empty outputs, STOP failures, invalid actions, and out-of-bounds motion
remain in reports. B2 demonstration knot markers identify public authored
waypoints among integration steps; query READ never consumes them.
The optional `stroke_reversal` control is currently available for A/B1 only;
B2 rejects it because reversing an executed program needs a separate valid
re-execution procedure.

## Frozen metrics and analysis

Use a separate Python environment containing the donor evaluator dependencies
(`torch`, `torchvision`, NumPy, SciPy, Pillow). Its CPU worker loads the original
checkpoint and source files directly. Keep those dependencies out of the JAX
training environment. Set `METRIC_PYTHON` to that environment's Python.

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw references \
  --cache-root datasets/quickdraw_cache --manifest-path outputs/quickdraw/F8.json \
  --output outputs/quickdraw/real_a --half real_a

uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.metrics extract \
  outputs/quickdraw/generated/correct_support --output outputs/quickdraw/features/correct_support \
  --python "$METRIC_PYTHON" --donor-root ../quick-robot-draw \
  --extractor-checkpoint ../quick-robot-draw/metrics/checkpoints/resnet18_step40000.pt
```

Export `real_b` with `--half real_b` and extract both reference halves using the
same extractor command. Score with:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.metrics score \
  outputs/quickdraw/features/correct_support outputs/quickdraw/features/real_a \
  --reference-repeat outputs/quickdraw/features/real_b --output outputs/quickdraw/scores.json
```

Raw 512D pooled grayscale ResNet features, the original incoming-pen renderer,
64x64/AA2/line-width-2 inputs, and no feature normalization define Sketch-FD.
The versioned float64 calculation handles rank-deficient small samples; negative
unbiased polynomial MMD estimates are preserved. Reports separate pooled FD,
macro category MMD, real-real calibration, failure counts, and optional classifier
accuracy. Classifier labels require checkpoint provenance attesting the label
map. Unknown evaluator training-ID overlap remains a publication limitation.

For A copying diagnostics, freeze the actual prompt and training populations:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.populations \
  --cache-root datasets/quickdraw_cache --manifest-path outputs/quickdraw/F8.json \
  --episodes outputs/quickdraw/episodes --output outputs/quickdraw/copy_populations \
  --training-count 4096 --seed 0
```

Extract the resulting `supports/` and `training_reservoir/` with the same frozen
evaluator. Then run `python -m icil_jax_rlbench.quickdraw.metrics copying
GENERATED_FEATURES SUPPORT_FEATURES TRAINING_FEATURES --manifest-path MANIFEST
--output COPYING.json` in the locked JAX command environment. The scorer measures
nearest actual-support and permitted-training distances and within-prompt
diversity. It separates feature coincidences from exact point-sequence copies;
near-copy thresholds require development calibration. The population exporter
currently covers canonical A inputs; transformed B support needs separately
identified executed-trajectory feature artifacts.

`analyze --checkpoint ... --episodes ... --output ...` exports raw support,
gradients, actual clipped/rate-scaled updates, final fast deltas, oracle alignment,
and responses to identical public probes. Category probes group reused programs
and duplicates, fit scaling on probe-training folds, and skip final-test fitting.
Participation ratio describes covariance across tasks, not rank within a weight
matrix. No diagnostics feed query labels into adaptation.

`aggregate --specification runs.json --output report.json` consumes a JSON list
of objects containing `generated`, optional `features` keyed by every condition,
`reference`, and `reference_repeats`. Include `family_count`, `model_seed`, and
`subset_seed` as run descriptors. It reports support gain and specificity with
category/base-program bootstrap intervals; generated samples sharing a prompt
are not independent model replicates. Without original features A's metric
status remains pending. Independent seed replication and capacity/convergence
controls are needed before any diversity-threshold claim.

Qualitative galleries have a separate fixed selection step:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.gallery select outputs/quickdraw/generated/correct_support \
  --output outputs/quickdraw/gallery_selection.json --count 16 --seed 0
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw gallery --artifact outputs/quickdraw/generated/correct_support \
  --selection outputs/quickdraw/gallery_selection.json --output outputs/quickdraw/gallery.svg
```

The same selection can render paired controls. These labeled SVGs show pen-up
travel and failures; canonical metric rasters come from the donor worker.

## Bounded validation

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw fixture --output datasets/quickdraw_fixture
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw pilot --a-protocol a_category --cache-root datasets/quickdraw_fixture \
  --output outputs/quickdraw_fixture_pilot --levels 2 8 --steps 2 --unique-budget 64 \
  --max-steps 32 --config icil_jax_rlbench/configs/quickdraw_smoke.py
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw gates --cache-root datasets/quickdraw_fixture \
  --manifest-path outputs/quickdraw_fixture_pilot/F2_manifest.json \
  --output outputs/quickdraw_fixture_gates --steps 50 \
  --config icil_jax_rlbench/configs/quickdraw_smoke.py
```

Gate Q4 fits a fixed meta-batch through the full graph and separately measures
supervised support adaptation in the exact fast subspace at frozen slow weights.
It does not select inner steps using query labels. Negative adaptation results
are retained. The compatibility A-category pilot uses two F levels. The new
A-local pilot uses two N levels at fixed F, and A-NN compares against matched
category-sampled supports. All require explicit bounded launches. C and large
sweeps are disabled.

Correctness tests cover the real sketch model, finite differences, full/FOMAML
paths, loop/vmap/JIT, padding, reset, exact resume, label-independent generation,
split isolation, B replay and feedback recovery, metric fixtures, and diagnostic
rank/probe semantics. Original-checkpoint parity requires
`QUICKDRAW_EVALUATOR_CHECKPOINT` and `QUICKDRAW_METRIC_PYTHON`; it is explicitly
skipped while those assets are absent. See the implementation notes for the
recorded local results and the distinction from real-data acceptance.
