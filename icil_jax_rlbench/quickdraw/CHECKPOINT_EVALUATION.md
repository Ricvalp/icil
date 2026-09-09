# Checkpoint figures and periodic Sketch-FID

Both supervised architectures use these commands. Evaluation restores the
checkpoint's model, K, neighbor selection mode and conditioning setting.
The examples use the AR checkpoint path; substitute the trained diffusion
checkpoint or another saved checkpoint as appropriate. No retraining is needed.

## Sampling and metric

The default frozen development protocol selects **100 distinct query sketches
per category**, giving **34,500 generated sketches across 345 categories**.
Each query is only a retrieval center. Generation receives its actual K nearest
neighbors, sampled exactly as configured during training, and no query actions,
category labels or embeddings. The query's duplicate group is excluded from its
context. Generation is free-running, with one sample per query center.

Compare these generations with **34,500 independently selected reserved real
drawings**, also 100 per category. References cannot be queries or context
drawings. Both populations and sampling seeds are fixed across checkpoints.
Sampling a different query and using it only to select neighbors is the same
conditional input construction used for training, applied to a held-out split.

The score uses the existing frozen sketch-trained ResNet18's **raw 512D
features**, with the existing 64x64 rasterizer. It is named **Sketch-FID**, and
logged as `validation/sketch_fid`; these values are not comparable to ImageNet
Inception FID. Compute one pooled score with equal category weights. Per-class
FID with only 100 samples in 512 dimensions would be poorly estimated.

34,500 is a practical starting budget, not a statistical significance guarantee.
For final comparisons, repeat generation with several declared seeds and check
sample-count stability. Fixed N improves comparability but does not eliminate
model-dependent finite-sample bias ([Chong and Forsyth, CVPR
2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)).

Empty, nonfinite, out-of-bounds and missing-STOP generations remain in the
population. Nonfinite trajectories render as blanks under the existing failure
policy. Failure counts accompany the metric; there is no rejection sampling.
FID alone does not establish neighborhood adherence or absence of copying.
The [copy-metric proposal](COPY_METRIC_PROPOSAL.md) awaits agreement.

## Additional HPC resources

The full dataset already at
`/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1` and the policy checkpoint are
enough for figures. FID additionally needs the frozen classifier, its provenance,
the donor's two evaluator source files, a CPU metric environment, and one compact
prepared reference directory. No raw downloads, full raster cache, embedding
cache or new FAISS build are required.

The following transfers use workstation source paths. Run them from a machine
that has those files and can reach `peano`; SSH from this workstation is known
to be unavailable. Keep the relative paths in each transfer:

```bash
cd /home/rvalperga/icil
rsync -aR --partial --info=progress2 \
  outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt \
  outputs/quickdraw_evaluator/resnet18_v1/provenance.json \
  datasets/quickdraw_fid_development_v1 \
  peano:~/icil/

cd /home/rvalperga/quick-robot-draw
rsync -aR dataset/rasterize.py metrics/resnet18.py peano:~/quick-robot-draw/
```

The default development reference was prepared and checked locally at
`datasets/quickdraw_fid_development_v1`: **55,877,615 bytes**, covering 34,500
unique reserved drawings and 34,500 distinct query centers. Its self-distance
is zero within floating-point tolerance, and 32 sampled feature vectors match
the existing raw embedding cache within `4.8e-6`. The transfers above total
approximately **103 MB**, including the classifier and provenance.

Transfer this bundle and skip preparation on the HPC. Use
the exact `resnet18_best.pt` that was used for reference preparation. Updating
the classifier or renderer requires preparing a new reference directory.

After updating the code on the HPC, install the existing JAX environment as in
the [HPC guide](../../hpc/README.md), then create the additional metric environment
on the login node (skip this if already installed):

```bash
cd ~/icil
uv venv --python 3.12 .venv-quickdraw-metrics
uv pip install --python .venv-quickdraw-metrics/bin/python \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-quickdraw-metrics/bin/python \
  numpy==2.3.4 pillow==12.0.0 scipy==1.16.3
```

The separate process performs frozen CPU feature extraction only. Training and
W&B logging remain in the main JAX process. There is no separate logging script.

## Prepare references once

Run this on an allocated CPU node, or locally with the local dataset path.
It renders and extracts features for the reserved real drawings once. It does
not train the classifier. The output is immutable and portable between machines.

```bash
cd ~/icil
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m icil_jax_rlbench.quickdraw.supervised_fid prepare \
  --dataset-root /hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1 \
  --output datasets/quickdraw_fid_development_v1 \
  --samples-per-category 100 \
  --split development --seed 2029 \
  --python .venv-quickdraw-metrics/bin/python \
  --donor-root ../quick-robot-draw \
  --extractor-checkpoint outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt \
  --checkpoint-provenance outputs/quickdraw_evaluator/resnet18_v1/provenance.json
```

The reference bundle contains selected query/reference row IDs, real features,
hashes and evaluator provenance. It discards temporary real rasters and tokens.
Preparing another sample budget requires a distinct output directory. Insufficient
unique drawings in any category cause an error instead of silent downsampling.

## Generate checkpoint figures

Submit from `~/icil` on the HPC:

```bash
bash hpc/submit_quickdraw_h200.sh visualize \
  --checkpoint outputs/quickdraw_icil/ar_transformer_v1/best.pkl \
  --output outputs/quickdraw_icil/ar_transformer_v1/figures
```

This requests one H200 for up to 12 hours and passes the external dataset path
automatically. Inside an existing GPU allocation, the direct command is:

```bash
.venv/bin/python -u -m icil_jax_rlbench.quickdraw.supervised_visualize \
  --checkpoint outputs/quickdraw_icil/ar_transformer_v1/best.pkl \
  --dataset-root /hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1 \
  --output outputs/quickdraw_icil/ar_transformer_v1/figures
```

Defaults save `context_samples.png`/`context_samples.pdf` showing **10 generated sketches with
their K demonstrations**, and `sample_gallery.png`/`sample_gallery.pdf` containing a **10x10
square of generated sketches**. The gallery visits distinct categories before
repeating one; its first ten drawings also appear in the context panels. PDF
strokes are vector paths; PNGs are 300 dpi. Use `--formats png pdf svg` for SVG,
or `--no-category-labels` for an unlabeled gallery.

`samples.json` records checkpoint/dataset hashes, selected IDs, actual contexts,
seeds and failures. `generated_samples.npz` stores the plotted sequences. Figures
retain every selected sample, including failures. Category captions describe the
intended retrieval category, not an independent classification of the output.
Use a new output directory for each checkpoint or seed.

## Score a checkpoint

```bash
bash hpc/submit_quickdraw_h200.sh fid \
  --checkpoint outputs/quickdraw_icil/ar_transformer_v1/best.pkl \
  --reference datasets/quickdraw_fid_development_v1 \
  --output outputs/quickdraw_icil/ar_transformer_v1/fid_best \
  --checkpoint-provenance outputs/quickdraw_evaluator/resnet18_v1/provenance.json
```

The direct equivalent inside a GPU allocation is:

```bash
.venv/bin/python -u -m icil_jax_rlbench.quickdraw.supervised_fid evaluate \
  --checkpoint outputs/quickdraw_icil/ar_transformer_v1/best.pkl \
  --dataset-root /hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1 \
  --reference datasets/quickdraw_fid_development_v1 \
  --output outputs/quickdraw_icil/ar_transformer_v1/fid_best \
  --checkpoint-provenance outputs/quickdraw_evaluator/resnet18_v1/provenance.json
```

The result is printed and saved in `summary.json`, with sample provenance in
`generation.json`. Full generated rasters, tokens and features are temporary by
default; add `--keep-artifacts` to retain them. Generation defaults to batch size
8, and frozen CPU feature extraction to 64; tune with `--batch-size` and
`--feature-batch-size`. No trained-checkpoint score has been measured merely by
preparing the reference bundle.

For final reporting, prepare a separate reference bundle using `--split test
--allow-test`, then pass `--allow-test` when scoring it. Test references are
rejected by periodic training evaluation. Keep test results out of checkpoint
selection.

## Log Sketch-FID during training

References and evaluator files must exist before submission. For a new AR run:

```bash
bash hpc/submit_quickdraw_h200.sh ar \
  --set fid_enabled=true \
  --set 'output_dir="outputs/quickdraw_icil/ar_transformer_fid_v1"'
```

Use `diffusion` and a corresponding new output directory for that architecture.
The metric is **off by default**; when enabled its interval defaults to
**10,000 optimizer steps**. Change it with `--set fid_every=20000`.
`--set 'fid_reference_root="/absolute/path/to/reference"'` overrides the reference
location. Other resource overrides are `fid_python`, `fid_donor_root`,
`fid_extractor_checkpoint`, and `fid_checkpoint_provenance`.

The existing `icil-quickdraw` W&B run logs `validation/sketch_fid`, generated and
real sample counts, metric duration, and generation failure statistics. Local
`metrics.jsonl` contains the same scalar records. Metric summaries live under
`fid/step_000010000/`; full artifacts remain optional with
`--set fid_keep_artifacts=true`. Existing four-example image logging remains
independent under `samples/context_and_generated`.

Metric RNG is independent of training RNG; the query/reference sets, neighbor
choices and generation seeds remain fixed across evaluations. Once enabled,
changing the reference or `fid_seed` within the same run is rejected. Changing
the interval, batch sizes or file locations is allowed when resuming.

To add FID while resuming an unfinished run, repeat its original configuration
and overrides, adding `--resume --set fid_enabled=true`. Known trainer versions
before the plotting/FID additions can resume with unchanged numerical settings;
an `evaluation_upgrade.json` records this migration. Package/backend/device and
model/data/optimizer checks still apply. A completed training budget must be
explicitly extended to train further. To score an already completed run, use
the standalone checkpoint command; it does not modify that run's W&B history.

The best checkpoint still follows validation loss. Adding FID does not change
the optimization objective, sampling schedule or checkpoint selection rule.
Evaluating 34,500 sketches takes additional time; the logged `validation/fid_seconds`
measures that overhead so the interval can be adjusted to fit the 12-hour job.
