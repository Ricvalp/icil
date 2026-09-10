# Support-BC WRITE experiments

These six new runs compare a supervised support behavior-cloning (BC) WRITE
objective, the number of inner steps, fast-state capacity, and Transformer
capacity. The existing `kvb-heldout` three-step run is the base KVB comparison.
All runs use the same 310 policy-training classes and 35 held-out classes
(class-split seed 37), training seed 0, K=4, effective task batch 64, float32,
and 20 epochs. No dataset, classifier, or nearest-neighbor rebuild is needed.

Support-BC uses the query decoder and probabilistic output head to predict each
demonstration's next action from its causal preceding actions. It pools all
valid events from all K demonstrations, including STOP, into one support loss.
Each inner step uses that same full support batch. Only the small fast MLP is
adapted; the Transformer, head, learned initialization W0, and learned update
rates receive outer gradients. Full second-order differentiation passes through
the inner updates. The outer objective is independent-query likelihood alone.

Reset the fast state to W0 for each task and freeze its adapted value throughout
query generation. Support reaches query predictions only through adapted fast
weights and delta READ. The support-BC model omits the KVB-only support encoder
and K/V projections. It shares the query decoder/head design and fast-state
capacity with its corresponding KVB model, but their total parameter counts
are consequently different.

| New submission mode | WRITE | Inner steps | Fast MLP | Query Transformer width / layers | Microbatch |
| --- | --- | --- | --- | --- | --- |
| `bc-heldout` | Support BC | 3 | 64 / 128 / 64 | 256 / 6 | 4 |
| `bc1-heldout` | Support BC | 1 | 64 / 128 / 64 | 256 / 6 | 4 |
| `bc5-heldout` | Support BC | 5 | 64 / 128 / 64 | 256 / 6 | 4 |
| `bc-fast128-heldout` | Support BC | 3 | 128 / 256 / 128 | 256 / 6 | 4 |
| `bc-large-heldout` | Support BC | 3 | 64 / 128 / 64 | 384 / 8 | 2 |
| `kvb-large-heldout` | KVB | 3 | 64 / 128 / 64 | 384 / 8 | 2 |

All Transformers have eight attention heads. The larger KVB support encoder
uses six layers, compared with four in the existing base KVB configuration.
Support-BC configurations inherit that field for the shared configuration
contract but do not instantiate a KVB support encoder.

Active parameter counts for these configurations are:

| Configuration | Total stored parameters | Query decoder and head | Fast initialization W0 |
| --- | ---: | ---: | ---: |
| Existing base KVB | 8,052,408 | 4,780,340 | 16,576 |
| Support BC, 1 / 3 / 5 steps | 4,830,008 | 4,780,340 | 16,576 |
| Support BC, fast dimension 128 | 4,912,184 | 4,780,340 | 65,920 |
| Large support BC | 14,321,464 | 14,255,284 | 16,576 |
| Large KVB | 25,060,152 | 14,255,284 | 16,576 |

The total includes W0 and four learned per-tensor update rates; adapted task
weights are transient. These counts come from the instantiated parameter
shapes, including the omission of KVB-only components from support BC.

Compare `bc-heldout` with the existing `kvb-heldout` to assess the WRITE
objective; compare `bc1-heldout` and `bc5-heldout` with `bc-heldout` for update
depth; compare `bc-fast128-heldout` with `bc-heldout` for fast-state capacity.
The two large runs allow the WRITE-objective comparison at larger Transformer
capacity, and each can be compared with its own base model.
The KVB reconstruction and support-BC WRITE losses use different units. Compare
methods using held-out query negative log-likelihood and Sketch-FID; raw WRITE
loss magnitudes are not comparable between objectives.

## Copy code and submit

Update the code in `~/icil` on peano using your existing transfer method. If
using rsync, run this from a machine that has the updated repository and can
reach peano; workstation SSH is unavailable:

```bash
rsync -a --exclude='__pycache__/' --exclude='*.pyc' --exclude='logs/' \
  icil_jax_rlbench hpc tests peano:~/icil/
```

The existing environment and dataset at
`/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1` are reused. On peano, from
`~/icil`, submit whichever new experiments you want to run:

```bash
bash hpc/submit_quickdraw_h200.sh bc-heldout
bash hpc/submit_quickdraw_h200.sh bc1-heldout
bash hpc/submit_quickdraw_h200.sh bc5-heldout
bash hpc/submit_quickdraw_h200.sh bc-fast128-heldout
bash hpc/submit_quickdraw_h200.sh bc-large-heldout
bash hpc/submit_quickdraw_h200.sh kvb-large-heldout
```

Each command submits **one H200 for 12 hours**, with 16 CPUs and 64 GB host RAM.
The training budget may require multiple allocations, particularly for the
larger configurations; continue from saved checkpoints with `--resume`.
Each experiment has its own `.sbatch` file: its mode with hyphens replaced by
underscores appears in `hpc/quickdraw_MODE_h200.sbatch`, for example
[`quickdraw_bc_fast128_heldout_h200.sbatch`](../../hpc/quickdraw_bc_fast128_heldout_h200.sbatch).
The wrapper detects the site's H200 resource spelling; all jobs share the
existing GPU checks, environment setup, and trainer.

Output directories under `outputs/quickdraw_icil/` are:

| Mode | Directory |
| --- | --- |
| `bc-heldout` | `support_bc_transformer_heldout35_v1` |
| `bc1-heldout` | `support_bc_transformer_s1_heldout35_v1` |
| `bc5-heldout` | `support_bc_transformer_s5_heldout35_v1` |
| `bc-fast128-heldout` | `support_bc_transformer_fast128_heldout35_v1` |
| `bc-large-heldout` | `support_bc_transformer_large_heldout35_v1` |
| `kvb-large-heldout` | `kvb_transformer_large_heldout35_v1` |

Start these as fresh runs. An existing KVB checkpoint is not a support-BC
resume checkpoint. After a job reaches its time limit, append `--resume` to
the same mode with the original overrides, for example:

```bash
bash hpc/submit_quickdraw_h200.sh bc-large-heldout --resume
```

W&B uses the existing login and remains online by default. Training and
validation metrics are logged, with fixed context/generated panels every
10,000 steps. Validation and its figures use the 35 held-out classes'
development drawings. Test drawings from those same classes remain separate;
see the [class holdout guide](CLASS_HOLDOUT.md).

Optional Sketch-FID is off by default. Once the existing evaluator resources
are available on the HPC, enable it at the default 10,000-step interval:

```bash
bash hpc/submit_quickdraw_h200.sh bc-heldout --set fid_enabled=true
```

The existing checkpoint visualization and FID commands also accept support-BC
checkpoints and use their saved class split. For example:

```bash
bash hpc/submit_quickdraw_h200.sh visualize \
  --checkpoint outputs/quickdraw_icil/support_bc_transformer_heldout35_v1/best.pkl \
  --output outputs/quickdraw_icil/support_bc_transformer_heldout35_v1/figures
```

See [checkpoint evaluation](CHECKPOINT_EVALUATION.md) for reference preparation,
standalone Sketch-FID, and separate test evaluation. The frozen classifier used
for retrieval and Sketch-FID has seen all categories; the 35-class exclusion
applies to policy training.
