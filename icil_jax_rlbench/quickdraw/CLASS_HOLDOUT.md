# Hold out classes from policy training

The optional class split applies to ordinary autoregressive ICIL, ordinary
diffusion ICIL, and the full second-order KVB autoregressive Transformer. The
supplied configurations choose **35 of 345 classes**, using class-split seed
**37**, and train on the other **310 classes**. The same count and split seed
select the same classes across methods, independently of the training RNG seed.
The existing configurations still train on all classes by default.

Training targets and their nearest-neighbor demonstrations come only from the
310 training classes. Validation loss, best-checkpoint selection, periodic
sample panels, and optional validation Sketch-FID use development drawings
from the 35 held-out classes. Later test evaluation uses the same 35 classes
with separate test drawings. The held-out classes are therefore used during
model selection; the test drawings remain separate from validation drawings.
On the current full cache this gives 922,250 training targets, 22,330
development targets, and 22,330 test targets.

These classes are held out of **policy training**. The existing frozen ResNet
used for retrieval and Sketch-FID was trained across all categories. Reusing
that evaluator does not make the entire pipeline unseen-class training.

Reuse the current dataset and its category-local, split-local neighbor tables.
No new download, rendering, classifier training, or FAISS build is needed.
Start fresh policy runs: an existing all-category policy checkpoint has already
seen the held-out classes and cannot serve as this experiment's initialization
or resume checkpoint.

## Submit on the HPC

After updating the code on peano, run from `~/icil`. Ordinary AR ICIL:

```bash
bash hpc/submit_quickdraw_h200.sh ar-heldout
```

Ordinary diffusion ICIL:

```bash
bash hpc/submit_quickdraw_h200.sh diffusion-heldout
```

Full second-order KVB with three pooled support updates:

```bash
bash hpc/submit_quickdraw_h200.sh kvb-heldout
```

Each command submits its own **12-hour, single-H200** job with 16 CPUs and
64 GB host RAM. The dedicated batch files are
[`quickdraw_icil_heldout_h200.sbatch`](../../hpc/quickdraw_icil_heldout_h200.sbatch)
and [`quickdraw_kvb_heldout_h200.sbatch`](../../hpc/quickdraw_kvb_heldout_h200.sbatch).
They reuse the shared training setup and default to
`/hpc/home/phi/rvalperga/data/quickdraw_full_nn_v1`. W&B remains enabled through
the existing authentication and `WANDB_MODE` setting. Metric logging and sample
panels every 10,000 steps follow the existing configuration.

Outputs are separate for each experiment:

| Submission mode | Output directory |
| --- | --- |
| `ar-heldout` | `outputs/quickdraw_icil/ar_transformer_heldout35_v1` |
| `diffusion-heldout` | `outputs/quickdraw_icil/diffusion_transformer_heldout35_v1` |
| `kvb-heldout` | `outputs/quickdraw_icil/kvb_transformer_heldout35_v1` |

After a time limit, use the same mode and overrides with `--resume`:

```bash
bash hpc/submit_quickdraw_h200.sh ar-heldout --resume
bash hpc/submit_quickdraw_h200.sh kvb-heldout --resume
```

The dataset and class assignment are checked on resume. Keep the count, split
seed, and other scientific settings unchanged. To change the holdout count or
seed for a new experiment, use a new output directory:

```bash
bash hpc/submit_quickdraw_h200.sh ar-heldout \
  --set heldout_category_count=20 \
  --set heldout_category_seed=41 \
  --set 'output_dir="outputs/quickdraw_icil/ar_transformer_heldout20_seed41_v1"'
```

The trainer writes the full assignment to `class_split.json` in the run
directory and records its identity in checkpoints. To print the 35 names:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path('outputs/quickdraw_icil/ar_transformer_heldout35_v1/class_split.json')
split = json.loads(path.read_text())
print('\n'.join(split['heldout_categories']))
PY
```

## Figures and Sketch-FID

Checkpoint tools automatically use the saved held-out class assignment. For
development figures with their actual in-context demonstrations:

```bash
bash hpc/submit_quickdraw_h200.sh visualize \
  --checkpoint outputs/quickdraw_icil/ar_transformer_heldout35_v1/best.pkl \
  --output outputs/quickdraw_icil/ar_transformer_heldout35_v1/figures
```

Use `--split test --allow-test` and a separate output directory when producing final test
figures. Substitute the diffusion or KVB checkpoint to evaluate those policies.
For an explicit comparison on familiar classes, pass `--category-scope seen`;
`--category-scope all` selects all classes and labels that scope in the output.

To enable periodic Sketch-FID during training, once the existing reference and
evaluator resources are available on the HPC:

```bash
bash hpc/submit_quickdraw_h200.sh ar-heldout --set fid_enabled=true
bash hpc/submit_quickdraw_h200.sh kvb-heldout --set fid_enabled=true
```

The interval defaults to 10,000 steps. The existing all-category development
reference is filtered to the held-out classes and its real feature statistics
are recomputed for that subset. At 100 samples per class, this evaluates
**3,500 generated sketches against 3,500 real sketches**. Scores and sample
budgets therefore differ from the existing 34,500-sample all-category protocol.
Use the same class split and sample budget for comparisons between policies.

For standalone development FID:

```bash
bash hpc/submit_quickdraw_h200.sh fid \
  --checkpoint outputs/quickdraw_icil/ar_transformer_heldout35_v1/best.pkl \
  --reference datasets/quickdraw_fid_development_v1 \
  --output outputs/quickdraw_icil/ar_transformer_heldout35_v1/fid_best
```

Final test FID needs a separate test reference prepared with `--split test
--allow-test`; pass `--allow-test` when evaluating that reference. Preparation
and evaluator resources are described in [checkpoint evaluation](CHECKPOINT_EVALUATION.md).
Test references cannot be used for periodic training evaluation.

The older neighborhood-control evaluator requires its frozen episode manifest
to match the evaluation classes. Its existing all-category manifest is rejected
by default for a holdout checkpoint. `--category-scope all` allows that manifest
for explicitly labeled all-category diagnostics. The figure and FID commands
above filter the existing resources automatically.
