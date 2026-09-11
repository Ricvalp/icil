# QuickDraw on peano: one H200 for 12 hours

The supplied SSH host is `peano` (`rvalperga@10.0.31.201`). These commands use
`~/icil` for the repository and
`/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1` for the training dataset.
SSH from the workstation timed out during preparation,
so the remote filesystem, quota, network policy, and Slurm resource names have
not been verified. Connect to the required network/VPN first. If the cluster
requires a project or scratch directory, substitute that directory consistently
for the remote `~/icil` below; submit from the transferred repository root.

`quickdraw_h200.sbatch` requests partition `gpuq`, **one H200**, one task,
16 CPUs, 64 GB host RAM, and **12:00:00**. The partition comes from
`example_sbatch.sbatch`. `submit_quickdraw_h200.sh` queries Slurm to find the
actual H200 GRES spelling or node feature and overrides the batch file's default
`gpu:h200:1` spelling. Ambiguous or missing H200 resources stop submission.
The job also checks that JAX sees exactly one H200 before training.

## 1. Copy code and the complete training dataset

Run on the workstation:

Skip these transfers if the repository is already cloned and the dataset is
already present at the path above.

```bash
cd /home/rvalperga/icil
ssh peano 'mkdir -p ~/icil/hpc/logs /hpc/home/phi/rvalperga/data && df -h /hpc/home/phi/rvalperga/data'

rsync -a --info=progress2 \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='logs/' \
  pyproject.toml uv.lock README.md AGENTS.md icil_jax_rlbench tests hpc \
  peano:~/icil/

rsync -a --partial --info=progress2 \
  datasets/quickdraw_full_nn_v1/ \
  peano:/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1/
```

The second transfer is **7,513,038,034 bytes (7.51 GB), 2,086 files**. Copy the
whole directory: it contains the tokens, masks' source lengths, category and
split mappings, compact top-32 neighbor tables, manifest, and all **1,035 FAISS
indexes**. Training verifies every manifest-listed file, including the indexes.
Allow additional space for the CUDA environment, checkpoints, plots, and logs.

This copies the current code, including uncommitted/untracked implementation
files. It does not copy workstation virtual environments, raw NDJSON, raster
images, the initial vector cache, classifier training state, or partial policy
run directories. Those are unnecessary for training, full validation, and
periodic context/generated-sketch plots. The source-cache and embedding paths
inside the manifest are provenance; training does not open those paths.

`phi-mujoco`, `quick-robot-draw`, PyTorch, FAISS Python, and the ResNet checkpoint
are not runtime dependencies of these two supervised trainers.

## 2. Create the environment on the HPC login node

```bash
ssh peano
cd ~/icil

uv sync --frozen --no-default-groups --group visualization \
  --extra cuda12 --extra wandb --python 3.12

.venv/bin/wandb login
```

Use the cluster-provided `uv` command/module if needed. The workstation uses
uv 0.11.21. The frozen install was
dry-run tested in an isolated project without a sibling `phi-mujoco` checkout.
The visualization group supplies matplotlib for the periodic PNG panels.
CUDA runtime libraries come from the pinned JAX CUDA extra; the compute node
must supply a compatible NVIDIA driver. The job performs no downloads or
environment installation, and the login node does not need a GPU for setup.
Authenticate W&B interactively on the HPC; do not rsync SSH keys or credentials.

For a compute node without outbound W&B access, set `WANDB_MODE=offline` before
submission. Images and metrics are retained locally for a later `wandb sync`.

## 3. Submit either architecture

On peano, from `~/icil`:

The batch file passes the dataset path above to either trainer. If it moves,
set `QUICKDRAW_DATASET_ROOT` to its new absolute path before submission.

```bash
bash hpc/submit_quickdraw_h200.sh ar
```

Or submit the diffusion Transformer:

```bash
bash hpc/submit_quickdraw_h200.sh diffusion
```

For the full second-order KVB autoregressive Transformer, use:

```bash
bash hpc/submit_quickdraw_h200.sh kvb
```

This uses the same dataset and allocation, with three pooled support updates
per task. Its float32 meta-training configuration uses microbatches of 4 and
saves to `outputs/quickdraw_icil/kvb_transformer_v1`. See the
[KVB Transformer guide](../icil_jax_rlbench/quickdraw/KVB_TRANSFORMER.md) for the
objective, resume, figures, and optional periodic FID commands.

To train with **35 classes held out of policy training**, and use those classes'
development drawings for validation, submit a new run with one of:

```bash
bash hpc/submit_quickdraw_h200.sh ar-heldout
bash hpc/submit_quickdraw_h200.sh diffusion-heldout
bash hpc/submit_quickdraw_h200.sh kvb-heldout
```

These use dedicated ICIL/KVB batch files and the same 12-hour H200 allocation.
All three modes use the same seeded 310/35 class assignment and save to separate
`*_heldout35_v1` directories. They reuse the existing dataset and evaluator.
See the [class holdout guide](../icil_jax_rlbench/quickdraw/CLASS_HOLDOUT.md)
for class names, count/seed overrides, resume, and held-out figures/FID.

Six additional held-out-class experiments use support-BC WRITE or larger
Transformers. Submission modes are `bc-heldout`, `bc1-heldout`, `bc5-heldout`,
`bc-fast128-heldout`, `bc-large-heldout`, and `kvb-large-heldout`, each with its
own 12-hour H200 batch file. See the
[support-BC experiment guide](../icil_jax_rlbench/quickdraw/SUPPORT_BC_EXPERIMENTS.md)
for the comparison table, exact commands, output directories, and resume.

Twelve larger-fast-state experiments cross BC/KVB WRITE, base/large Transformers,
and approximately 500k/1M/3M fast parameters. Their submission modes are
`{bc,kvb}-{base,large}-f{500k,1m,3m}`, for example `bc-base-f500k` or
`kvb-large-f3m`, each with a separate 12-hour, single-H200 batch file. See the
[fast-capacity experiment guide](../icil_jax_rlbench/quickdraw/FAST_CAPACITY_EXPERIMENTS.md)
for dimensions, total parameter counts, and the twelve commands. Only these new
batch files disable CUDA-graph capture by default after the large KVB capture
failure on peano; they print the resulting `XLA_FLAGS` in the job log. Set
`QUICKDRAW_CUDA_GRAPHS=1` before submission to keep inherited XLA flags without
adding the workaround. Existing experiment modes keep their original defaults.

Each command submits a separate 12-hour, single-H200 job. Use only the command
for the experiment you want to run. The ordinary AR/diffusion hyperparameters remain K=4, effective batch
64, microbatch 16, 20 epochs, and four-example plot panels every 10,000 updates.
The H200 does not automatically change the scientific configuration.

To pass ordinary trainer overrides, append them:

```bash
bash hpc/submit_quickdraw_h200.sh ar --set plot_every=20000
```

Logs go to `hpc/logs/quickdraw_ar_JOBID.{out,err}` (or
`quickdraw_diffusion_JOBID.{out,err}`). Policy outputs use the normal
`outputs/quickdraw_icil/ar_transformer_v1` and
`outputs/quickdraw_icil/diffusion_transformer_v1` directories, including
checkpoints, metrics, W&B identity, and `plots/`.

Inspect the queue or logs with:

```bash
squeue -u rvalperga
tail -f hpc/logs/quickdraw_ar_JOBID.out
```

The supplied batch file can also be submitted directly once the site's exact
H200 spelling is known. Create `hpc/logs` first and override `--gres` and, if
necessary, `--constraint` on the `sbatch` command line. Do not use an unqualified
`--gres=gpu:1` unless a verified H200 node constraint accompanies it.

## 4. Resume and retrieve outputs

If the job reaches its time limit before finishing, resubmit the same
architecture and overrides with `--resume`:

```bash
bash hpc/submit_quickdraw_h200.sh ar --resume
```

The trainer saves `last.pkl` every 1,000 updates and at epoch boundaries.
Timeout recovery starts from that checkpoint, not the exact kill instant.
Keep the full run directory and the same code/environment between jobs.
The requested GPU type remains H200. An RTX-workstation checkpoint cannot be
resumed unchanged because the trainer enforces the original device signature;
these commands start fresh H200 runs. No trained policy checkpoint was present
in the workstation's `outputs/quickdraw_icil` when this guide was prepared.

To copy outputs back, run on the workstation; use a separate destination to
preserve any existing workstation attempt:

```bash
mkdir -p /home/rvalperga/icil/outputs/quickdraw_hpc
rsync -a --partial --info=progress2 \
  peano:~/icil/outputs/quickdraw_icil/ \
  /home/rvalperga/icil/outputs/quickdraw_hpc/
```

## Optional: evaluation resources on the HPC

For the newer 10-example context panels, 10x10 galleries, and class-balanced
34,500-sample Sketch-FID (including optional training logging), follow the
[checkpoint evaluation guide](../icil_jax_rlbench/quickdraw/CHECKPOINT_EVALUATION.md).
The following resources are for the earlier neighborhood-control evaluation.

These are not needed to start training or log its plots. To run the existing
frozen development generation/metric protocol on the HPC as well, copy:

```bash
cd /home/rvalperga/icil
rsync -aR --partial --info=progress2 \
  datasets/quickdraw_full_nn_eval_v1 \
  datasets/quickdraw_full_nn_eval_features_v1 \
  outputs/quickdraw_evaluator/resnet18_v1/resnet18_best.pt \
  outputs/quickdraw_evaluator/resnet18_v1/provenance.json \
  peano:~/icil/

ssh peano 'mkdir -p ~/quick-robot-draw'
cd /home/rvalperga/quick-robot-draw
rsync -aR dataset/rasterize.py metrics/resnet18.py peano:~/quick-robot-draw/
```

These resources total approximately **485 MB**, plus the two small donor source
files. The existing CPU metric worker dynamically loads only those donor files.
Before feature extraction, create its isolated environment on the HPC login node:

```bash
cd ~/icil
uv venv --python 3.12 .venv-quickdraw-metrics
uv pip install --python .venv-quickdraw-metrics/bin/python \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-quickdraw-metrics/bin/python \
  numpy==2.3.4 pillow==12.0.0 scipy==1.16.3
```

This worker performs CPU inference, so it does not need another CUDA environment.
Use the transferred frozen classifier instead of retraining it. Update donor paths
to the HPC's `~/quick-robot-draw` when using the
[evaluation commands](../icil_jax_rlbench/quickdraw/FULL_DATASET.md).

`datasets/quickdraw_embeddings_v1` is another 7.36 GB and is needed only to build
new neighbor tables or prepare new evaluation selections (for example, final
test selections). It is not needed for training or the already-frozen evaluation:

```bash
cd /home/rvalperga/icil
ssh peano 'mkdir -p ~/icil/datasets'
rsync -a --partial --info=progress2 \
  datasets/quickdraw_embeddings_v1/ peano:~/icil/datasets/quickdraw_embeddings_v1/
```
