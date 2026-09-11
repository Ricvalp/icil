# Fast-memory capacity: twelve experiments

These runs cross two WRITE objectives (support BC and KVB), two Transformer
sizes (base and large), and three fast-parameter budgets (approximately 500k,
1M, and 3M). Each has a separate Python config, sbatch file, and output directory.
They reuse the prepared data and the existing environment.

Every model has one fast MLP, read immediately before the probabilistic output
head. Base means a width-256, six-layer query Transformer; large means width
384 and eight layers. Both have eight attention heads. KVB additionally has a
four-layer support encoder for base and six layers for large. Support BC uses
the query decoder for WRITE and has no separate support encoder.

## Actual parameter counts

Keep `fast_dim=256` across this matrix and vary only `fast_hidden_dim`. For an
MLP `d -> h -> d`, including both biases, the fast count is `(2*d + 1)*h + d`.
Widths are multiples of 16; each actual count is within 0.2% of its run label.
The fixed input/output width also keeps KVB's normalized target dimension and
coordinate-averaging factor the same across all three budgets.

| Run label | Fast MLP | Fast parameters |
| --- | --- | ---: |
| `f500k` | 256 -> 976 -> 256 | 500,944 |
| `f1m` | 256 -> 1952 -> 256 | 1,001,632 |
| `f3m` | 256 -> 5840 -> 256 | 2,996,176 |

Total trainable counts below include the learned fast initialization W0 and
the four learned per-tensor update rates. Task-adapted copies of W0 are temporary
runtime state, not additional trainable models or checkpoint parameters.

| Fast budget | BC base total | BC large total | KVB base total | KVB large total |
| --- | ---: | ---: | ---: | ---: |
| 500k | 5,412,872 | 14,953,480 | 8,733,960 | 25,840,008 |
| 1M | 5,913,560 | 15,454,168 | 9,234,648 | 26,340,696 |
| 3M | 7,908,104 | 17,448,712 | 11,229,192 | 28,335,240 |

Base and large specify the Transformer architecture, not a fixed total across
budgets or objectives. BC and KVB match query-policy and fast-state dimensions;
KVB's extra writer accounts for their different total counts.

## Files and submission modes

All twelve jobs use the same 35 held-out classes (split seed 37), training seed
0, K=4, exact top-K neighbors, float32, and 20 epochs. Each task takes three full
second-order GD steps on the same pooled support batch. The fast state resets
to W0 per task and freezes during query generation; the outer loss is query
likelihood only. Effective task batch size is 64, with microbatches of 4 for
base and 2 for large, as in the existing configurations.

| WRITE | Fast budget | Base sbatch / mode | Large sbatch / mode |
| --- | --- | --- | --- |
| BC | 500k | [bc-base-f500k](../../hpc/quickdraw_bc_base_f500k_h200.sbatch) | [bc-large-f500k](../../hpc/quickdraw_bc_large_f500k_h200.sbatch) |
| BC | 1M | [bc-base-f1m](../../hpc/quickdraw_bc_base_f1m_h200.sbatch) | [bc-large-f1m](../../hpc/quickdraw_bc_large_f1m_h200.sbatch) |
| BC | 3M | [bc-base-f3m](../../hpc/quickdraw_bc_base_f3m_h200.sbatch) | [bc-large-f3m](../../hpc/quickdraw_bc_large_f3m_h200.sbatch) |
| KVB | 500k | [kvb-base-f500k](../../hpc/quickdraw_kvb_base_f500k_h200.sbatch) | [kvb-large-f500k](../../hpc/quickdraw_kvb_large_f500k_h200.sbatch) |
| KVB | 1M | [kvb-base-f1m](../../hpc/quickdraw_kvb_base_f1m_h200.sbatch) | [kvb-large-f1m](../../hpc/quickdraw_kvb_large_f1m_h200.sbatch) |
| KVB | 3M | [kvb-base-f3m](../../hpc/quickdraw_kvb_base_f3m_h200.sbatch) | [kvb-large-f3m](../../hpc/quickdraw_kvb_large_f3m_h200.sbatch) |

Configs live under `icil_jax_rlbench/configs/` and follow the pattern
`quickdraw_{support_bc,kvb}_transformer_{base,large}_fast{500k,1m,3m}_heldout.py`.
Outputs live under `outputs/quickdraw_icil/` with directory names
`{support_bc,kvb}_transformer_{base,large}_fast{500k,1m,3m}_heldout35_v1`.

Transfer the updated code to `~/icil` on peano using the existing transfer
method. The dataset remains at
`/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1`; no data/index rebuild or
classifier retraining is needed. From `~/icil` on peano, submit any one mode:

```bash
bash hpc/submit_quickdraw_h200.sh bc-base-f500k
```

Each file requests one H200 for 12 hours, 16 CPUs, and 64 GB host RAM. To submit
the complete matrix, run these twelve commands on peano:

```bash
bash hpc/submit_quickdraw_h200.sh bc-base-f500k
bash hpc/submit_quickdraw_h200.sh bc-base-f1m
bash hpc/submit_quickdraw_h200.sh bc-base-f3m
bash hpc/submit_quickdraw_h200.sh bc-large-f500k
bash hpc/submit_quickdraw_h200.sh bc-large-f1m
bash hpc/submit_quickdraw_h200.sh bc-large-f3m
bash hpc/submit_quickdraw_h200.sh kvb-base-f500k
bash hpc/submit_quickdraw_h200.sh kvb-base-f1m
bash hpc/submit_quickdraw_h200.sh kvb-base-f3m
bash hpc/submit_quickdraw_h200.sh kvb-large-f500k
bash hpc/submit_quickdraw_h200.sh kvb-large-f1m
bash hpc/submit_quickdraw_h200.sh kvb-large-f3m
```

Start each size as a new run. To continue a saved run, use the same mode and
overrides with `--resume`. Larger fast memories increase compute and may need
more than one 12-hour allocation. If changing microbatch size, start a new run;
it is part of the runner's immutable resume configuration.

W&B metrics and fixed context/generated-sketch panels remain enabled, with
panels every 10,000 steps. Optional development Sketch-FID is off by default;
enable it with `--set fid_enabled=true` when its evaluator resources are
available. Figures and FID use the same held-out class split. Existing standalone
checkpoint visualization and FID commands work for every config.

## CUDA graph workaround

The twelve new sbatch files default to `--xla_gpu_enable_command_buffer=` to
disable CUDA graph capture after the reported HPC capture failure. This is a
runtime workaround, not a confirmed diagnosis of its original cause. Full
second-order differentiation remains enabled. The resolved `XLA_FLAGS` are
printed in the job log; other inherited flags are preserved.

To leave inherited graph settings unchanged, explicitly set
`QUICKDRAW_CUDA_GRAPHS=1` before submitting. Use the same runtime flags when
resuming; the current checkpoint fingerprint does not record `XLA_FLAGS`.

See [support-BC experiments](SUPPORT_BC_EXPERIMENTS.md) for the WRITE objective
and [checkpoint evaluation](CHECKPOINT_EVALUATION.md) for figures and Sketch-FID.
