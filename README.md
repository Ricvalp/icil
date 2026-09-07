# Fast-Weight Test-Time Training for Robot Imitation

This branch implements the adaptation-only fast-weight TTT program in
`ICIL_TTT_IMPLEMENTATION_PLAN.md`. The former direct-regression pretraining,
parameter-MAML, and memory-MAML paths have been removed deliberately.

The policy experiments use hidden-goal MetaWorld ML1 Reach, ML1 Push, ML10, and
ML45 through the independent `phi-mujoco` interoperability layer. A self-contained synthetic
hidden-goal benchmark remains available for fast mechanism and autodiff tests.
End-to-end RLBench TTT remains gated on successful held-out-latent state
experiments.

## Setup

The project uses Python 3.11 and [`uv`](https://docs.astral.sh/uv/). The
synthetic-only environment is:

```bash
uv sync
```

The MetaWorld path uses the independent `phi-mujoco` checkout as an editable
sibling rather than copying or nesting that repository:

```text
Robotics/
  icil-jax-rlbench/
  phi-mujoco/
```

Clone both repositories with that layout before syncing this environment. The
experiments summarized below used `phi-mujoco` commit `e59d9b0`:

```bash
git clone git@github.com:PHI-Lab-AI4I/phi-mujoco.git ../phi-mujoco
git -C ../phi-mujoco checkout e59d9b0
```

Install that local integration and optional W&B support with:

```bash
uv sync --group metaworld --extra wandb
uv run --group metaworld pytest -q
```

This installs the `metaworld_ml1_reach`, `metaworld_ml1_push`, `metaworld_ml10`,
and `metaworld_ml45` data/evaluation adapters. ICIL owns the task-aware
support/query sampler, train-task normalization, JAX policy, adaptation, and
experimental controls; simulator and collection logic remains in `phi-mujoco`.

On a CUDA 12 machine, add the JAX CUDA wheels:

```bash
uv sync --group metaworld --extra cuda12 --extra wandb
```

`uv` environment sync is exact: repeat `--extra cuda12` on every later
`uv sync` or `uv run` invocation that can synchronize this environment, or
`uv` will remove the optional CUDA plugin and JAX will fall back to CPU. Verify
the backend with:

```bash
uv run --frozen python -c "import jax; print(jax.default_backend(), jax.devices())"
```

All commands below can then run through `uv run`; the package is installed
editable in its own environment.

## Core Invariant

Support demonstrations may affect a query prediction only by updating an
explicit transient fast state. The prediction API receives:

```text
(slow parameters, adapted fast state, query observation) -> query action
```

It has no support argument, task label, language input, or query-action input.
Fast state is reset to meta-learned `W0` at each task boundary, adapted on
support, and then frozen while evaluating independent query episodes.

## Current Components

- `phi-mujoco` ML1 Reach and ML1 Push cache loading with declared 40/10/50
  task splits.
- Deconfounded ML10 loading with explicit family/instance hierarchy, confirmed
  development holdouts, horizon bucketing, and untouched official test families.
- Train-task-only observation and action normalization stored in checkpoints.
- Query-only MetaWorld behavior cloning with 39D hidden-goal state and 4D
  continuous actions.
- Separate ML45 family-conditioned and family-plus-task-latent oracle behavior
  cloning controls for policy-capacity diagnosis.
- Ordinary support-BC Gate 1 adaptation and matched fresh closed-loop rollouts.
- Full-second-order MetaWorld KVB meta-training initialized from the query-only
  policy, with matched FOMAML and support-BC WRITE configs.
- Held-out MetaWorld support-count and support-information controls with offline
  and fresh closed-loop evaluation.
- Synthetic 2D hidden-goal reach-and-grasp benchmark with disjoint train,
  validation, and test goals.
- Query-only base policy for the ordinary-adaptation upper bound.
- Full-second-order key-value binding (KVB) WRITE objective.
- Matched FOMAML and support action-BC WRITE ablations.
- Closed-loop no/correct/wrong/shuffled-support controls.
- Fixed-meta-batch gradient and update-direction diagnostics.
- RLBench delta-translation, 6D-rotation, geodesic, and gripper losses.
- RLBench space-time supernodes and small event registers for the later visual
  phase.

## ML1 Reach Workflow

Create a balanced ML1 Reach cache with `phi-mujoco`. This example collects 24
independent demonstrations for each of the 100 declared tasks:

```bash
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw="$PWD/datasets/raw/metaworld_ml1_reach-$stamp"
processed="$PWD/datasets/processed/metaworld_ml1_reach-$stamp"

uv run --frozen --group metaworld phi-mujoco collect metaworld_ml1_reach \
  --output-directory "$raw" \
  --episodes 2400 \
  --seed 0 \
  --quiet --progress

uv run --frozen --group metaworld phi-mujoco convert \
  metaworld_ml1_reach "$raw" \
  --output-directory "$processed"

uv run --frozen --group metaworld phi-mujoco validate-cache "$processed"
export PHI_MUJOCO_ML1_REACH_CACHE="$processed"
```

Train the support-free query-only policy over the 40 training goals:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_query_only \
  --config=icil_jax_rlbench/configs/metaworld_ml1_reach_query_only.py
```

Run Gate 1 on the 10 validation goals. Each condition uses identical fresh
query starts, while offline support and query demonstrations are disjoint:

```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.eval_metaworld_ml1_reach_gate1 \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml1_reach_gate1.py \
  --config.checkpoint_path=/path/to/query_only/last.pkl
```

The default Gate 1 run adapts all parameters for 100 SGD steps from four
support demonstrations and compares no update, correct support, wrong-task
support, shuffled actions, and observations-only controls. Override
`inner_steps`, `inner_lr`, and `adapt_subset` for the planned 50--200 step and
parameter-subset sweeps.

Once Gate 1 establishes that support is informative, initialize MetaWorld KVB
meta-training from the competent query-only checkpoint. This starts a new TTT
step counter; it does not reuse the query-only optimizer:

```bash
export ICIL_ML1_REACH_QUERY_CHECKPOINT=/absolute/path/to/query_only/last.pkl

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_ttt \
  --config=icil_jax_rlbench/configs/metaworld_ml1_reach_kvb.py
```

The default meta-batch contains four training goals. Each goal contributes two
support and two different query demonstrations with padded shapes
`[task, demonstration, time, feature]`. The KVB writer processes support in
16-transition segments. The outer objective is action imitation only on the
query demonstrations, and full second-order gradients pass through every
WRITE update.

Checkpoints are written below
`outputs/metaworld_ml1_reach_ttt/<run-id>/`. Evaluate the 10 validation goals
before touching the 50-task test split:

```bash
CKPT=/absolute/path/to/outputs/metaworld_ml1_reach_ttt/<run-id>/last.pkl

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.eval_metaworld_ml1_reach_ttt \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml1_reach_ttt.py \
  --config.checkpoint_path="$CKPT"
```

The default evaluation sweeps one, two, and four support demonstrations over
all support controls. Every condition uses the same fresh closed-loop seeds;
fast state resets to `W0` independently for each goal and condition, then stays
frozen across that condition's query rollouts. Results are written to
`eval_outputs/metaworld_ml1_reach_ttt/<checkpoint>_<timestamp>/summary.json`.
After selecting all settings on validation, run the untouched test split once:

```bash
--config.split=test
```

To record a small qualitative sample without changing evaluator code, add:

```bash
--config.max_tasks=3 \
--config.support_counts='(2,)' \
--config.conditions='("no_update","correct_support","wrong_task_support")' \
--config.save_rollout_artifacts=True \
--config.record_video=True
```

The matched training ablations are:

```text
icil_jax_rlbench/configs/metaworld_ml1_reach_fomaml.py
icil_jax_rlbench/configs/metaworld_ml1_reach_action_bc.py
```

## ML1 Push Workflow

ML1 Push is the next controlled benchmark after Reach. The hidden task latent
is the fixed planar target for the puck. The 39D policy observation contains
the hand and puck state but its three goal slots are always zero. Every episode
gets independently sampled hand and puck starts, so support and query episodes
share only the target. The catalog uses 40 training targets, 10 validation
targets, and the 50 official ML1 test targets.

Collect and convert a balanced cache. As with Reach, 2,400 episodes gives 24
successful demonstrations per target:

```bash
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw="$PWD/datasets/raw/metaworld_ml1_push-$stamp"
processed="$PWD/datasets/processed/metaworld_ml1_push-$stamp"

uv run --frozen --group metaworld phi-mujoco collect metaworld_ml1_push \
  --output-directory "$raw" \
  --episodes 2400 \
  --seed 0 \
  --quiet --progress

uv run --frozen --group metaworld phi-mujoco convert \
  metaworld_ml1_push "$raw" \
  --output-directory "$processed"

uv run --frozen --group metaworld phi-mujoco validate-cache "$processed"
export PHI_MUJOCO_ML1_PUSH_CACHE="$processed"
```

Train the support-free behavior-cloning policy and run the ordinary-adaptation
upper bound on the 10 validation targets:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_query_only \
  --config=icil_jax_rlbench/configs/metaworld_ml1_push_query_only.py

QUERY_CKPT=/absolute/path/to/outputs/metaworld_ml1_push_query_only/<run-id>/last.pkl

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.eval_metaworld_ml1_push_gate1 \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml1_push_gate1.py \
  --config.checkpoint_path="$QUERY_CKPT"
```

Initialize KVB meta-training from that competent query-only policy, then run
the matched held-out support controls:

```bash
export ICIL_ML1_PUSH_QUERY_CHECKPOINT="$QUERY_CKPT"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_ttt \
  --config=icil_jax_rlbench/configs/metaworld_ml1_push_kvb.py

TTT_CKPT=/absolute/path/to/outputs/metaworld_ml1_push_ttt/<run-id>/last.pkl

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.eval_metaworld_ml1_push_ttt \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml1_push_ttt.py \
  --config.checkpoint_path="$TTT_CKPT"
```

Use validation for model and support-count choices. Run the untouched 50-target
test split only after those choices are fixed by adding `--config.split=test`.
The matched Push ablations are:

```text
icil_jax_rlbench/configs/metaworld_ml1_push_fomaml.py
icil_jax_rlbench/configs/metaworld_ml1_push_action_bc.py
```

## ML10 Workflow

ML10 is the first cross-task-family experiment. The development protocol trains
on 40 native instances from each of eight official train families. Instances
40--49 from those families form latent validation; all 50 `pick-place-v3` and
`door-open-v3` instances form family validation. The five official ML10 test
families remain untouched until settings are fixed. Family and instance IDs are
cache metadata only and never enter the model.

Collect eight successful demonstrations for every one of the 750 instances:

```bash
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw="$PWD/datasets/raw/metaworld_ml10-$stamp"
processed="$PWD/datasets/processed/metaworld_ml10-$stamp"

uv run --frozen --group metaworld phi-mujoco collect metaworld_ml10 \
  --output-directory "$raw" --episodes 6000 --seed 0 --quiet --progress
uv run --frozen --group metaworld phi-mujoco convert metaworld_ml10 "$raw" \
  --output-directory "$processed"
uv run --frozen --group metaworld phi-mujoco validate-cache "$processed"
export PHI_MUJOCO_ML10_CACHE="$processed"
```

Collection is balanced by construction and stores only successful rollouts.
Each required family/instance/episode slot has a 100-attempt retry budget because
some official scripted experts, especially `door-open-v3`, succeed only on a
subset of the deconfounded episode starts.

Train the development query-only policy and test the ordinary-adaptation upper
bound on held-out families:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_query_only \
  --config=icil_jax_rlbench/configs/metaworld_ml10_query_only.py

QUERY_CKPT=/absolute/path/to/outputs/metaworld_ml10_query_only/<run-id>/last.pkl

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.eval_metaworld_ml10_gate1 \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml10_gate1.py \
  --config.checkpoint_path="$QUERY_CKPT"
```

Then train the original gated-READ full second-order KVB and its ablations:

```bash
export ICIL_ML10_QUERY_CHECKPOINT="$QUERY_CKPT"

for config in metaworld_ml10_kvb metaworld_ml10_fomaml metaworld_ml10_action_bc; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --frozen --group metaworld python \
    -m icil_jax_rlbench.train_metaworld_ttt \
    --config="icil_jax_rlbench/configs/${config}.py"
done
```

The support-BC delta-READ experiment avoids the zero-gate collapse of the
original action-BC path. Delta READ injects the projected difference between
the adapted fast model and `W0`; it is therefore an exact query-only no-op at
`W0` while retaining a nonzero derivative with respect to the fast state. Use
the same READ parameterization for KVB and support-BC so the WRITE objectives
remain matched:

```bash
export ICIL_ML10_QUERY_CHECKPOINT="$QUERY_CKPT"

for config in metaworld_ml10_kvb_delta_read metaworld_ml10_action_bc_delta_read; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --frozen --group metaworld python \
    -m icil_jax_rlbench.train_metaworld_ttt \
    --config="icil_jax_rlbench/configs/${config}.py"
done
```

Small eager configs are available for stepping through either full
second-order objective. They use one task, one support episode, one query
episode, the first 32 temporal slots, and two outer steps. With the default
segment size, each support batch contains two WRITE segments. These configs are
for debugging, not measurements:

```bash
for config in metaworld_ml10_kvb_debug metaworld_ml10_action_bc_debug; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --frozen --group metaworld python \
    -m icil_jax_rlbench.train_metaworld_ttt \
    --config="icil_jax_rlbench/configs/${config}.py"
done
```

Evaluate a checkpoint on family validation. The default screen uses ten task
instances and matched fresh rollout seeds; remove `max_tasks` for the complete
100-instance validation run.

```bash
TTT_CKPT=/absolute/path/to/outputs/metaworld_ml10_ttt/<run-id>/last.pkl

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.eval_metaworld_ml10_ttt \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml10_ttt.py \
  --config.checkpoint_path="$TTT_CKPT"
```

This evaluation distinguishes same-family wrong-instance support from support
drawn from a different family. Videos remain isolated in the evaluation path;
add `--config.record_video=True --config.save_rollout_artifacts=True`.

Inspect what task information the learned WRITE update retains with the
standalone update-information diagnostic. It extracts full vectors without a
random projection and writes `features.npz`, `summary.json`, `report.md`, and a
probe plot:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.analyze_metaworld_ml10_information \
  --config=icil_jax_rlbench/configs/analyze_metaworld_ml10_information.py \
  --config.checkpoint_path="$TTT_CKPT"
```

Run this once for KVB and once for support-BC. Frozen probes train on instances
000--039 and evaluate on instances 040--049. They report family classification,
within-family latent regression, update cosine geometry, support-control
direction changes, query gain, and oracle query-gradient alignment. The oracle
gradient is diagnostic-only and never enters WRITE or READ.

After all settings are frozen, retrain query-only and KVB with
`metaworld_ml10_query_only_final.py` and `metaworld_ml10_kvb_final.py`. Evaluate
that final KVB checkpoint once with `--config.split=test --config.max_tasks=0`.

## ML45 Workflow

ML45 tests whether the ML10 failure is caused by an easy family-labeling
minimum. The development protocol trains on instances 000--039 from 40
families, uses instances 040--049 for latent validation, and holds out five
compositionally related families for family validation. The five native ML45
test families remain untouched. Family/composition labels are cache metadata
and never enter the model.

Collect at least four successful episodes for each of the 2,500 tasks. This is
the minimum for the default two-support/two-query meta-batch; collect more when
the offline evaluator needs additional independent query episodes:

```bash
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw="$PWD/datasets/raw/metaworld_ml45-$stamp"
processed="$PWD/datasets/processed/metaworld_ml45-$stamp"

uv run --frozen --group metaworld phi-mujoco collect metaworld_ml45 \
  --output-directory "$raw" --episodes 10000 --seed 0 --quiet --progress
uv run --frozen --group metaworld phi-mujoco convert metaworld_ml45 "$raw" \
  --output-directory "$processed"
uv run --frozen --group metaworld phi-mujoco validate-cache "$processed"
export PHI_MUJOCO_ML45_CACHE="$processed"
```

Train query-only BC, then initialize full-second-order KVB with delta READ:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_query_only \
  --config=icil_jax_rlbench/configs/metaworld_ml45_query_only.py

QUERY_CKPT=/absolute/path/to/outputs/metaworld_ml45_query_only/<run-id>/last.pkl
export ICIL_ML45_QUERY_CHECKPOINT="$QUERY_CKPT"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_ttt \
  --config=icil_jax_rlbench/configs/metaworld_ml45_kvb.py
```

Two direct-conditioning controls test whether the query policy can represent
the expert behavior when task information is supplied explicitly. They are
separate from the TTT path and their checkpoints cannot initialize KVB:

```bash
# Discrete family only: 39D state + 50D family one-hot.
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_conditioned_query \
  --config=icil_jax_rlbench/configs/metaworld_ml45_family_conditioned_query_only.py

# Oracle control: state + family + normalized declared task latent + mask.
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.train_metaworld_conditioned_query \
  --config=icil_jax_rlbench/configs/metaworld_ml45_oracle_conditioned_query_only.py
```

The oracle control reads only reset-vector fields that the public
`phi_mujoco` family contract marks as `task_latent`. It excludes task IDs,
instance indices, hashes, and episode nuisance. ML45 has at most three such
continuous values, so the oracle model input is 95D: 39 state + 50 family + 3
latent + 3 validity mask. Task-latent normalization is fitted on development
training tasks only.

Evaluate either conditioned checkpoint on unseen instances of the 40 training
families:

```bash
CONDITIONED_CKPT=/absolute/path/to/conditioned_query_only/<run-id>/last.pkl

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.eval_metaworld_ml45_conditioned_query \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml45_conditioned_query.py \
  --config.checkpoint_path="$CONDITIONED_CKPT"
```

Add `--config.max_tasks=3 --config.closed_loop_episodes=3
--config.save_rollout_artifacts=True --config.record_video=True` for a small
video run. The default `latent_validation` split is intentional. Evaluation on
family holdouts is rejected unless `allow_unseen_families=True`, because their
one-hot coordinates were never optimized and therefore do not provide a fair
family-conditioned capacity control.

Evaluate the compositional family holdouts with matched support controls:

```bash
TTT_CKPT=/absolute/path/to/outputs/metaworld_ml45_ttt/<run-id>/last.pkl

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld python \
  -m icil_jax_rlbench.eval_metaworld_ml45_ttt \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml45_ttt.py \
  --config.checkpoint_path="$TTT_CKPT"
```

Expand the family-validation screen to ten latent instances from each of the
five held-out families, with one matched rollout per condition:

```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld --extra cuda12 python \
  -m icil_jax_rlbench.eval_metaworld_ml45_ttt \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml45_ttt.py \
  --config.checkpoint_path="$TTT_CKPT" \
  --config.output_dir=eval_outputs/metaworld_ml45_ttt/family_validation_expanded \
  --config.split=family_validation \
  --config.max_tasks=50 \
  --config.support_counts='(2,)' \
  --config.conditions='("no_update","correct_support","same_family_wrong_instance","different_family_support","shuffled_actions","random_update_matched_norm")' \
  --config.closed_loop_episodes=1
```

Analyze what task information ML45 WRITE updates contain and generate
interactive update embeddings:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld --extra cuda12 python \
  -m icil_jax_rlbench.analyze_metaworld_ml45_information \
  --config=icil_jax_rlbench/configs/analyze_metaworld_ml45_information.py \
  --config.checkpoint_path="$TTT_CKPT"
```

The diagnostic saves all unprojected representations in `features.npz`, frozen
probe and support-control results in `summary.json`, and interactive plots under
`tsne/`. Each t-SNE uses row-normalized updates followed by PCA, never a random
projection. Family plots distinguish splits by marker shape. Held-out-focus
plots color the five family-validation families, retain familiar-family points
as gray context, and connect the two support samples belonging to each task
instance. Original-space cosine and nearest-neighbor measurements are recorded
in `tsne/manifest.json`; use those for conclusions and t-SNE for exploration.

Run the ordinary-adaptation upper bound on the same 50 family-validation
instances when distinguishing unavailable support information from a learned
fast-WRITE failure:

```bash
QUERY_CKPT=/absolute/path/to/outputs/metaworld_ml45_query_only/<run-id>/last.pkl

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld --extra cuda12 python \
  -m icil_jax_rlbench.eval_metaworld_ml45_gate1 \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml45_gate1.py \
  --config.checkpoint_path="$QUERY_CKPT" \
  --config.output_dir=eval_outputs/metaworld_ml45_gate1/family_validation_expanded \
  --config.split=family_validation \
  --config.max_tasks=50 \
  --config.support_episodes=2 \
  --config.offline_query_episodes=2 \
  --config.inner_steps=100 \
  --config.inner_lr=0.01 \
  --config.adapt_subset=all \
  --config.closed_loop_episodes=1 \
  --config.closed_loop_base_seed=3000000
```

### Current ML45 result

As of 2026-09-03, one seed of query-only BC and full-second-order delta-READ KVB
has completed 100,000 steps. On 120 unseen latent instances from the 40 familiar
families, KVB correct-support adaptation reached 80.0% success versus 47.5% with
no update, but same-family wrong-instance and shuffled-action support still
reached 70.0% and 74.2%. On 50 instances from the five held-out development
families, correct KVB updates reduced success from 66% to 40%.

Ordinary support-BC adaptation of the query encoder and action heads on those
same 50 held-out-family instances improved success from 40% to 58%; shuffled
actions and observations-only controls reached 2% and 0%. The support therefore
contains a useful update direction, but current KVB does not transfer it through
the small fast state. The update-information diagnostic found highly
family-separable, low-effective-rank fast deltas that were almost unchanged by
time shuffling and had near-zero held-out query-gradient alignment.

The next unresolved comparison is action-head-only ordinary adaptation followed
by the matched full-second-order delta-READ support-BC WRITE run. These separate
fast-module placement/capacity from KVB-objective alignment. The official ML45
test split remains untouched. Exact checkpoints, metrics, artifact paths, and
the continuation decision tree are recorded in `SUMMARY.md`.

Use `metaworld_ml45_fomaml.py` and `metaworld_ml45_action_bc.py` for the matched
ablations. Add `--config.record_video=True
--config.save_rollout_artifacts=True` for qualitative rollouts. Do not switch
to `--config.split=test` until all choices are fixed on latent and family
validation.

The synthetic benchmark remains useful for Gate 2 and fast-weight
implementation checks:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -m icil_jax_rlbench.eval.ttt_state_gate2 \
  --config=icil_jax_rlbench/configs/ttt_state_gate2.py
```

## Synthetic Diagnostic

The controlled synthetic hidden-goal mechanism experiment remains the fast
autodiff and implementation diagnostic. MetaWorld is the policy experiment;
the synthetic result is not a substitute for held-out MetaWorld adaptation.

Train the main KVB method with at least three independent optimization seeds.
Keep `benchmark.seed` fixed so every seed uses the same disjoint task-latent
split:

```bash
for seed in 0 1 2; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --frozen python -m icil_jax_rlbench.train_ttt_state \
    --config=icil_jax_rlbench/configs/ttt_state_kvb.py \
    --config.benchmark.seed=0 \
    --config.train.seed="$seed"
done
```

Each run writes `last.pkl` below `outputs/ttt_state/<run-id>/`. Select
hyperparameters using only the 12 validation goals:

```bash
CKPT=/absolute/path/to/outputs/ttt_state/<run-id>/last.pkl

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen python -m icil_jax_rlbench.eval.ttt_state \
  --config=icil_jax_rlbench/configs/eval_ttt_state.py \
  --config.checkpoint_path="$CKPT" \
  --config.split=validation
```

The default sweep evaluates 1, 2, and 4 support demonstrations, 20 matched
query episodes per latent, and all support controls. It resets to `W0` for
every task and condition, then freezes the adapted fast state across that
task's query episodes. Results are written below
`eval_outputs/ttt_state/<checkpoint>_<timestamp>/summary.json`. Once choices
are fixed, run the same command once with `--config.split=test`.

Train and evaluate the matched FOMAML, action-BC WRITE, and no-WRITE/no-READ
ablations by replacing the training config with, respectively:

```text
icil_jax_rlbench/configs/ttt_state_fomaml.py
icil_jax_rlbench/configs/ttt_state_action_bc.py
icil_jax_rlbench/configs/ttt_state_query_only.py
```

### Gate 3 Visualization

Visualization is a separate replay pipeline so plotting and video code does
not enter the trainer or quantitative evaluator. Install its direct
dependencies on a CUDA machine with
`uv sync --group visualization --extra cuda12`, then render a few selected
validation tasks using the same checkpoint. CPU-only installations can omit
`--extra cuda12`.

```bash
SUMMARY=/absolute/path/to/eval_outputs/ttt_state/<evaluation>/summary.json

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group visualization python \
  -m icil_jax_rlbench.visualize_ttt_state \
  --config=icil_jax_rlbench/configs/visualize_ttt_state.py \
  --config.checkpoint_path="$CKPT" \
  --config.evaluation_summary_path="$SUMMARY" \
  --config.task_ids='(0,1,2)' \
  --config.support_count=2 \
  --config.query_episodes=3
```

Each task directory contains matched trajectory overlays and MP4 animation,
support trajectories and goals, action changes at identical query
observations, per-WRITE-step diagnostics, per-tensor fast-weight deltas, policy
vector fields, and the underlying arrays in `trajectory_data.npz`. Goals are
read from privileged benchmark metadata only for rendering and never enter the
policy. The top-level `evaluation_summary.png` summarizes the quantitative
support-count sweep.

## Acceptance Signal

The intended Gate 3 result is:

```text
correct support > no update approximately wrong or shuffled support
```

Before moving to RLBench, require either at least 15 points of absolute
closed-loop success improvement or at least a 30% relative query-loss reduction,
with confidence intervals excluding zero and substantially smaller gains from
wrong or shuffled support.

## Outputs and Resume

Each MetaWorld query-only or KVB run creates a unique directory under its
`train.output_dir` and writes:

- `resolved_config.json`;
- `provenance.json`;
- `task_splits.json`;
- `normalization.json`;
- `dataset_integrity.json`;
- `training_contract.json` for KVB runs;
- periodic `step_XXXXXXX.pkl` checkpoints and `last.pkl`.

Checkpoints contain slow parameters, meta-learned `W0`, optimizer state, RNG,
configuration, and provenance. They never contain task-specific adapted fast
state. Resume with:

```bash
--config.train.resume_path=/path/to/checkpoint.pkl \
--config.train.num_steps=40000
```

`train.num_steps` is the final target step, not an additional-step count.
KVB resume restores model, adaptation, meta-batch, and optimizer
hyperparameters from the checkpoint. It also restores optimizer state, JAX
RNG, and train/validation sampler RNG states. Only runtime controls such as the
new target step, logging intervals, output directory, cache location, and W&B
settings are taken from the resume command.

## Repository Layout

- `icil_jax_rlbench/data/metaworld_hidden_goal.py`: shared task-aware phi cache,
  train-task normalization, and support/query sampler for Reach and Push.
- `icil_jax_rlbench/data/metaworld_ml1_{reach,push}.py`: benchmark-specific
  dataset entrypoints.
- `icil_jax_rlbench/train/metaworld_query_runner.py`: query-only MetaWorld BC.
- `icil_jax_rlbench/train/metaworld_ttt_runner.py`: MetaWorld KVB meta-training,
  exact resume, ledger, and checkpoints.
- `icil_jax_rlbench/eval/metaworld_hidden_goal_gate1.py`: ordinary-adaptation
  upper bound and closed-loop controls shared by Reach and Push.
- `icil_jax_rlbench/eval/metaworld_hidden_goal_ttt.py`: held-out KVB support
  controls and closed-loop evaluation shared by Reach and Push.
- `icil_jax_rlbench/eval/metaworld_policy.py`: phi policy-interface adapter.
- `icil_jax_rlbench/eval/support_controls.py`: shared support perturbations and
  confidence intervals.
- `icil_jax_rlbench/data/hidden_goal.py`: synthetic controlled benchmark.
- `icil_jax_rlbench/models/fast_weight_ttt.py`: WRITE, READ, and fast-state logic.
- `icil_jax_rlbench/train/ttt_step.py`: meta-objective and JIT/pmap steps.
- `icil_jax_rlbench/train/ttt_runner.py`: training, validation, ledger, and resume.
- `icil_jax_rlbench/eval/ttt_state*.py`: Gates 1-3.
- `icil_jax_rlbench/visualization/`: deterministic Gate 3 rollout capture,
  plots, and animations.
- `icil_jax_rlbench/models/ttt_supernode.py`: visual event-register bridge.
- `icil_jax_rlbench/models/robotics_actions.py`: RLBench action geometry/losses.
- `icil_jax_rlbench/data/h5_cache.py`: retained dense RLBench H5 reader.

## RLBench Cache Contract

The retained reader expects `CACHE_ROOT/task_name/variationN.h5`, with episode
groups containing `xyz`, `valid`, `state`, and `action`; `rgb` and `mask_id` are
optional. Shapes must be inferred from data rather than hard-coded.

There is currently no RLBench TTT sampler, trainer, or online evaluator. Those
belong to the visual phase after Gate 3.

## Verification

```bash
uv run --frozen --group metaworld python -m compileall -q icil_jax_rlbench tests
uv run --frozen --group metaworld pytest -q
rg -n "^(from|import) (icil|icil_jax_query_memory|diagnostics|metaworld)(\\.|\\s|$)" \
  icil_jax_rlbench tests
```

See `IMPLEMENTATION_SUMMARY.md` for implemented gradient semantics and known
limits. See `SUMMARY.md` for the current experiment history, artifact map, and
continuation state.
