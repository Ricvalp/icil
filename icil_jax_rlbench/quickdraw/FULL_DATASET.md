# Full-dataset supervised in-context imitation

This is the current first policy experiment on `quick-robot-draw`: train an
autoregressive Transformer and a diffusion Transformer on all eligible training
drawings across all 345 categories. Both attend directly to K nearest-neighbor
demonstrations and learn through an ordinary supervised query objective. There
are no inner-loop updates or fast weights. The earlier bounded A-NN pilots and
TTT configurations are separate experiments; their category and target caps do
not apply to this workflow.

All commands below run from `/home/rvalperga/icil`. Dataset construction is
complete locally. Policy training has subsequently been run on the HPC.
For a 12-hour single-H200 job on peano, use the [HPC transfer and Slurm guide](../../hpc/README.md).
For checkpoint context panels, a 100-sample gallery, standalone Sketch-FID and
optional periodic metric logging, use the [checkpoint evaluation guide](CHECKPOINT_EVALUATION.md).

## Prepared data and full retrieval coverage

Reuse `datasets/quickdraw_prepared_v1/policy_cache`, the selected frozen
`outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt`, and
`datasets/quickdraw_embeddings_v1`. Downloading, classifier training, and
rendering/embedding do not need repeating. The classifier's own training,
validation, and test drawings remain separate from these policy drawings.

The verified full resource is `datasets/quickdraw_full_nn_v1`, approximately
7.51 GB. Every category has 5,000 policy-pool drawings, divided as follows:

| Policy split | Targets and retrieval candidates | Reserved metric references |
| --- | ---: | ---: |
| Training | 1,026,375 | 181,125 |
| Development/validation | 220,110 | 38,640 |
| Test | 220,110 | 38,640 |
| Total | 1,466,595 | 258,405 |

All 1,725,000 drawings are accounted for. The initial split fractions are
70%/15%/15%; 15% within each split is reserved for independent distributional
references. Training drawings can serve both as query targets and as another
target's demonstrations. Every eligible training target is visited once per
epoch, in shuffled order. Neither held-out drawings nor reserved references
enter training retrieval.

The resource contains fixed-horizon token arrays, compact integer neighbor
tables, a hashed manifest, and 1,035 exact cosine FAISS indexes: one per category
and split. Each query has 32 ranked candidates from its own split/category.
The query itself and its duplicate group are excluded; the selected neighbors
also have distinct duplicate groups. Drawing boundaries and lengths remain
explicit in arrays. The loader reads stored integer neighbors directly during
training, so no FAISS or classifier inference runs inside the JAX training step.

For reproduction on another machine, this single command builds the full
manifest, tokens, neighbor tables, and corresponding indexes together:

```bash
OMP_NUM_THREADS=4 .venv-quickdraw-metrics/bin/python \
  -m icil_jax_rlbench.quickdraw.full_data build \
  --cache-root datasets/quickdraw_prepared_v1/policy_cache \
  --embedding-root datasets/quickdraw_embeddings_v1 \
  --output datasets/quickdraw_full_nn_v1 \
  --top-m 32 --max-steps 128 \
  --train-fraction 0.7 --development-fraction 0.15 \
  --reference-fraction 0.15 --seed 0 \
  --batch-size 256 --threads 4
```

The existing local output is already complete; skip rebuilding it. A changed
split or retrieval budget belongs in a new output directory. The earlier
`quickdraw manifest` and `embeddings build-index` pilot commands are not needed
for this resource.

## Policies and tokens

Both models consume demonstration tokens shaped `[batch,K,128,4]` and their
padding masks. The four channels are `[x,y,pen_down,STOP]`, using absolute
normalized coordinates. `pen_down=0` at a point marks an incoming pen-up
movement, including the start of a stroke. STOP ends the drawing. This preserves
the donor's geometry and incoming-pen semantics; its separate SEP/RESET
channels are represented here by explicit sketch axes and decoder boundaries.
Up to 127 real points and one STOP fit in the public 128-step horizon.

Each demonstration has its own bidirectional Transformer encoding. The decoder
cross-attends to those encoded demonstration points. Category labels, query
identities, classifier embeddings, similarity scores, and query lengths are
offline metadata and never enter generation.

| Policy | Training objective | Generation |
| --- | --- | --- |
| Autoregressive Transformer | Teacher-forced Gaussian-mixture XY negative log likelihood, plus weighted pen/STOP Bernoulli losses | Sample one point/event at a time, with causal self-attention and cached attention keys/values |
| Diffusion Transformer | Predict Gaussian noise on the full noisy 128-step action sequence | Reverse the 100-step cosine DDPM process, with action self-attention and context cross-attention at every step |

Diffusion maps pen/STOP labels to -1/+1 and trains padding as repeated absorbing
STOP actions. The clean query mask only constructs training labels; the
denoiser receives no target length or padding mask. Generated pen/STOP values
are thresholded, and the first STOP determines the output length. Empty outputs,
missing STOP, and invalid or out-of-range geometry remain evaluation outcomes.
Autoregressive and diffusion training losses have different meanings and should
not be compared numerically to select between architectures.

## Train either architecture

The supplied configurations use K=4 exact nearest neighbors, hidden width 256,
8 attention heads, 4 context layers, 6 decoder layers, and dropout 0.1.
Parameters and optimizer state are float32; activations use bfloat16. An
effective batch of 64 accumulates four microbatches of 16. AdamW uses gradient
clipping, a 2,000-update warmup, and cosine learning-rate decay.

These defaults passed real-data GPU update and validation checks on this
machine: 9.55 million AR parameters and 12.43 million diffusion parameters.
After compilation, resident-batch measurements were approximately 897 and 873
examples/s respectively, with 2,926 MiB peak process GPU memory when JAX
preallocation was disabled. These measurements exclude data loading,
checkpointing, compilation, and full validation; they are not end-to-end epoch
time estimates. Details are saved in
`outputs/quickdraw_smoke/supervised_gpu_validation.json`.

Autoregressive training:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.supervised_train \
  --config icil_jax_rlbench/configs/quickdraw_ar_transformer.py
```

Diffusion training:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.supervised_train \
  --config icil_jax_rlbench/configs/quickdraw_diffusion_transformer.py
```

Their output directories are respectively
`outputs/quickdraw_icil/ar_transformer_v1` and
`outputs/quickdraw_icil/diffusion_transformer_v1`. Launch them as separate runs.
The default is 20 epochs, or 320,760 optimizer updates at batch size 64. Set an
initial budget with `--set epochs=NUMBER`; set it before starting so the
learning-rate schedule reflects the intended run.

Both configs enable direct W&B logging to project `icil-quickdraw`. Training
metrics are logged every 50 updates, with complete epoch aggregates, validation
losses, learning rate, throughput, and exposure counts. Local `metrics.jsonl`
contains the same records. To disable W&B, append `--set wandb_project=null`.

Both trainers also save a small PNG panel every **10,000 optimizer updates**.
The panel shows four fixed development examples, each with its K context
sketches and one generated sketch. The same contexts and generation seeds are
reused so that changes reflect the model's progress. Generation has its own
random stream and does not affect training or validation sampling.

Each interval uploads just one PNG to W&B under `samples/context_and_generated`
and saves it locally as `plots/step_000010000.png` inside the run directory.
The selected drawing IDs are recorded in `plots/selection.json`. Local PNGs are
saved even with W&B disabled. Empty, missing-STOP, and invalid generations remain
visible. No additional figures are emitted at epoch boundaries or shutdown.

Adjust the upload frequency with `--set plot_every=20000`, or disable plots with
`--set plot_every=0`. `--set plot_examples=2` reduces the panel to two examples
(allowed range 1-8). `plot_seed` defaults to 2027. Keep the example count and seed
fixed after the first panel; the frequency can change when resuming.

Validation evaluates all 220,110 development queries at the end of every epoch,
using microbatches that fit the training memory budget. `best.pkl` selects the
lowest validation objective; `last.pkl` saves every 1,000 updates and at epoch
boundaries. The test split does not select checkpoints. For explicitly bounded
validation, `--set validation_examples_per_category=NUMBER` chooses a fixed
subset, recorded in the run configuration; it does not reduce training coverage.

To resume an interrupted run, repeat its original command with `--resume`:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.supervised_train \
  --config icil_jax_rlbench/configs/quickdraw_ar_transformer.py \
  --resume
```

Resume restores the optimizer, RNG, epoch/shuffle position, metrics, and exposure
counts from the original run directory. Model, sampling, optimizer, validation,
source code, dependency versions, backend, and device signature must match.
The known trainer versions immediately preceding periodic plotting and FID can
also resume: these observational upgrades preserve the numerical training path
and write `evaluation_upgrade.json`, while retaining the original provenance.
Increasing `epochs` extends training with the original learning-rate schedule;
it does not restart or stretch that schedule. Repeat any original scientific
`--set` overrides when resuming.

To sample four demonstrations from each target's top 32 neighbors, start a new
run with `--set 'selection_mode="sample_top_m"'` and a new `output_dir`. The
default `exact_top_k` uses the closest four. Change K with
`--set support_count=8`, up to the stored top-M limit of 32. Changing sampling or
K defines a new run. The immutable table can be reused.

Train an independent no-context baseline with the same architecture and target
schedule:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.supervised_train \
  --config icil_jax_rlbench/configs/quickdraw_ar_transformer.py \
  --set condition_on_support=false \
  --set 'output_dir="outputs/quickdraw_icil/ar_no_context_v1"'
```

Use the diffusion config and a distinct output directory for its corresponding
baseline. Masking context only at evaluation is also available, but measures a
different intervention from independently training without demonstrations.

## Freeze evaluation inputs and compare conditions

The balanced development evaluation is already prepared at
`datasets/quickdraw_full_nn_eval_v1`, shared by both checkpoints. All declared
same-category negative-neighborhood separation checks passed on the real data.
For reproduction or a new evaluation directory, the command is:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.supervised_evaluate prepare \
  --dataset-root datasets/quickdraw_full_nn_v1 \
  --embedding-root datasets/quickdraw_embeddings_v1 \
  --output datasets/quickdraw_full_nn_eval_v1 \
  --split development --tasks-per-category 4 \
  --support-count 4 --selection-mode exact_top_k \
  --reference-count 8 --seed 0
```

These 1,380 evaluation targets cover all 345 categories. This generation budget
is independent of the full training and validation populations. The preparation
also saves disjoint local real-reference halves in `real_a` and `real_b` and
explicit same-category wrong-neighborhood pairings. If a category cannot meet
the declared negative-neighborhood separation rule, preparation reports that
failure instead of silently substituting a different control. Freeze these
choices before comparing generated results.

After training, generate four samples per target:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.supervised_evaluate generate \
  --checkpoint outputs/quickdraw_icil/ar_transformer_v1/best.pkl \
  --episodes datasets/quickdraw_full_nn_eval_v1 \
  --output outputs/quickdraw_icil/ar_transformer_v1/evaluation \
  --samples-per-task 4 --batch-size 8 --seed 0
```

Repeat with the diffusion checkpoint and output directory. The default
conditions are `correct_support`, `no_context`,
`same_category_wrong_neighborhood`, `wrong_support` from another category, and
`support_copy`. Query targets, reference identities, and generation keys remain
paired across conditions. To add the full-training nearest-example baseline,
pass all desired conditions explicitly with `--conditions`, including
`nearest_training_example`. Its selection uses shown-support raw geometry.
Summary files also contain paired held-out query likelihood components for AR
or paired held-out denoising errors for diffusion.

Feature extraction uses the same frozen classifier as curation. After generating
the five default conditions, extract their features:

```bash
for quickdraw_condition in correct_support no_context \
  same_category_wrong_neighborhood wrong_support support_copy; do
  uv run --frozen --group metaworld --extra cuda12 --extra wandb \
    python -m icil_jax_rlbench.quickdraw.metrics extract \
    "outputs/quickdraw_icil/ar_transformer_v1/evaluation/$quickdraw_condition" \
    --output "outputs/quickdraw_icil/ar_transformer_v1/features/$quickdraw_condition" \
    --python .venv-quickdraw-metrics/bin/python \
    --donor-root /home/rvalperga/quick-robot-draw \
    --extractor-checkpoint outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt \
    --checkpoint-provenance outputs/quickdraw_evaluator/resnet18_v1/provenance.json
done
```

The frozen reference features are already prepared and verified at
`datasets/quickdraw_full_nn_eval_features_v1/real_a` and `real_b`: 5,520 raw
512-dimensional vectors per half. Both match the fixed classifier and source
artifact hashes. They can be shared across policy runs. Skip the following
reference extraction locally; it documents reproduction for a new output:

```bash
for quickdraw_half in real_a real_b; do
  uv run --frozen --group metaworld --extra cuda12 --extra wandb \
    python -m icil_jax_rlbench.quickdraw.metrics extract \
    "datasets/quickdraw_full_nn_eval_v1/$quickdraw_half" \
    --output "datasets/quickdraw_full_nn_eval_features_v1/$quickdraw_half" \
    --python .venv-quickdraw-metrics/bin/python \
    --donor-root /home/rvalperga/quick-robot-draw \
    --extractor-checkpoint outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt \
    --checkpoint-provenance outputs/quickdraw_evaluator/resnet18_v1/provenance.json
done
```

Score all generated conditions together:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.supervised_evaluate score \
  --generation outputs/quickdraw_icil/ar_transformer_v1/evaluation \
  --feature-root outputs/quickdraw_icil/ar_transformer_v1/features \
  --reference datasets/quickdraw_full_nn_eval_features_v1/real_a \
  --reference-repeat datasets/quickdraw_full_nn_eval_features_v1/real_b \
  --output outputs/quickdraw_icil/ar_transformer_v1/evaluation_scores.json
```

This compares pooled Sketch-FD, raw-feature neighborhood MMD, real-real
reference variability, and paired support controls. Shared neighborhood
members/references are accounted for when constructing uncertainty estimates.
Local references are an offline target-neighborhood distribution proxy, not a
claim that every target defines an independent latent task.

To inspect a fixed random selection of 16 generations, create a qualitative SVG
gallery after generation:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.gallery select \
  outputs/quickdraw_icil/ar_transformer_v1/evaluation/correct_support \
  --output outputs/quickdraw_icil/ar_transformer_v1/gallery_selection.json \
  --count 16 --seed 0

uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.gallery render \
  outputs/quickdraw_icil/ar_transformer_v1/evaluation/correct_support \
  --selection outputs/quickdraw_icil/ar_transformer_v1/gallery_selection.json \
  --output outputs/quickdraw_icil/ar_transformer_v1/correct_support.svg
```

Reuse the selection file for each paired condition, changing the artifact and
SVG output paths. The gallery retains failed and empty samples and marks pen-up
travel. It is a qualitative view; metric extraction uses the canonical renderer.

Check raw geometry against the actual shown supports and all eligible training
drawings as a separate copying diagnostic:

```bash
uv run --frozen --group metaworld --extra cuda12 --extra wandb \
  python -m icil_jax_rlbench.quickdraw.supervised_evaluate geometry-copying \
  --dataset-root datasets/quickdraw_full_nn_v1 \
  --generated outputs/quickdraw_icil/ar_transformer_v1/evaluation/correct_support \
  --output outputs/quickdraw_icil/ar_transformer_v1/geometry_copying.json
```

For final held-out evaluation, prepare a new episode directory with
`--split test --allow-test`, then pass `--allow-test` during generation. Keep
development selection and final test reporting separate. These tools establish
the full-data comparison; evidence for a diversity threshold requires subsequent
controlled diversity/exposure comparisons and multiple seeds.
