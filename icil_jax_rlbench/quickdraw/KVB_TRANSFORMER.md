# Full-dataset KVB Transformer

This is the autoregressive Transformer experiment with full second-order
fast-weight adaptation. The existing ordinary AR and diffusion policies remain
separate baselines. Reuse `quickdraw_full_nn_v1`, its neighbor tables, and the
existing frozen evaluator and FID references. No new download, rendering,
classifier training, or FAISS build is needed.

## Model and objective

For each query centroid, retrieve K=4 distinct demonstrations using the same
split-local, category-local nearest-neighbor protocol as the ordinary policies.
A sequence Transformer encodes the demonstrations once, and produces learned
keys and values for all valid demonstration tokens, including STOP. Pool these
tokens into **one support batch**, and perform **three gradient steps on that
same batch** to fit a small fast MLP, `64 -> 128 -> 64`. This means three updates
per task, not three per demonstration or chunk.

Reset the fast MLP to the meta-learned initialization W0 for every task. The
autoregressive decoder reads the difference between the adapted MLP and W0
before its existing mixture-Gaussian XY and Bernoulli pen/STOP head. Support can
affect the query only through those updates; there is no direct support
cross-attention path. Query decoding sees only the causal action history.

The outer objective is the independent query's teacher-forced action likelihood.
The KVB reconstruction loss is the inner WRITE objective and a diagnostic; it is
not added to the outer loss. Full second-order gradients pass through all three
updates, including learned WRITE representations, W0 and update rates. During
generation, adapt once and freeze that task's fast state for the whole sketch.
Checkpoints store slow parameters, W0 and optimizer state; task-adapted weights
are transient and are never checkpointed or updated by the outer optimizer.

The default configuration uses float32, effective task batch 64 accumulated from
microbatches of 4, and the full training pool for 20 epochs. This is a new model
and run, initialized from scratch. An ordinary AR checkpoint remains usable for
baseline evaluation; it is not a resume checkpoint for the KVB architecture.

## Train on the HPC

With the updated repository and existing environment on peano, run from `~/icil`:

```bash
bash hpc/submit_quickdraw_h200.sh kvb
```

This requests **one H200 for 12 hours** and reads
`/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1`. Outputs go to
`outputs/quickdraw_icil/kvb_transformer_v1`. The existing W&B setup logs training
and validation metrics and four fixed context/generated panels every 10,000
updates. Checkpoints follow the existing validation-loss selection and periodic
`last.pkl` saving. To continue after the time limit, use the same overrides:

```bash
bash hpc/submit_quickdraw_h200.sh kvb --resume
```

Inside an allocated GPU session, the direct command is:

```bash
.venv/bin/python -u -m icil_jax_rlbench.quickdraw.supervised_train \
  --config icil_jax_rlbench/configs/quickdraw_kvb_transformer.py \
  --set 'dataset_root="/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1"'
```

Microbatch size can be reduced with `--set micro_batch_size=2` when starting a
run; effective batch size stays 64. Keep numerical settings unchanged on resume.

For the named FOMAML ablation, use a distinct run directory:

```bash
bash hpc/submit_quickdraw_h200.sh kvb \
  --set model.first_order=true \
  --set 'output_dir="outputs/quickdraw_icil/kvb_first_order_transformer_v1"'
```

This stops differentiation through the inner gradients. It does not change the
forward adaptation procedure and must be reported separately from full KVB.

## Figures and Sketch-FID

The same checkpoint tools restore the KVB model and adapt on its actual context.
For 10 examples with context and a separate 10x10 gallery:

```bash
bash hpc/submit_quickdraw_h200.sh visualize \
  --checkpoint outputs/quickdraw_icil/kvb_transformer_v1/best.pkl \
  --output outputs/quickdraw_icil/kvb_transformer_v1/figures
```

For the existing class-balanced development Sketch-FID protocol:

```bash
bash hpc/submit_quickdraw_h200.sh fid \
  --checkpoint outputs/quickdraw_icil/kvb_transformer_v1/best.pkl \
  --reference datasets/quickdraw_fid_development_v1 \
  --output outputs/quickdraw_icil/kvb_transformer_v1/fid_best
```

The frozen classifier, donor renderer and metric environment are needed for FID,
as documented in [checkpoint evaluation](CHECKPOINT_EVALUATION.md). The prepared
reference selects 34,500 held-out centroids and 34,500 reserved real drawings,
100 of each per category. Query centroids select context and never enter the
generation model. The score uses frozen sketch-ResNet features, not Inception
features, and does not establish absence of copying.

FID generation still defaults to batches of 64. Lower `--batch-size` if needed
for adaptation memory; feature extraction remains in the frozen CPU worker.
The generated population, context selection and per-example seeds stay fixed
across batch sizes.

To log Sketch-FID during a new training run, enable it at submission:

```bash
bash hpc/submit_quickdraw_h200.sh kvb --set fid_enabled=true
```

Its default interval is **10,000 optimizer steps**, independently of figures.
It logs `validation/sketch_fid` to W&B and local metrics, and records evaluation
duration. Adjust `--set fid_every=20000` or `--set fid_batch_size=16` if needed.
The metric is off by default and does not change the outer training objective
or the best-checkpoint rule. Frozen validation populations and independent RNGs
keep periodic evaluation separate from optimization.

The neighborhood-control evaluator also accepts KVB checkpoints. Its
`no_context` condition supplies an empty WRITE mask, leaving W=W0, while wrong
neighborhood/category conditions adapt on the corresponding substituted
demonstrations. Existing support-copy baselines remain available there. No new
copy-rate metric is added to training until its definition is agreed.
