# Codex Handoff: Fast-Weight TTT for Robot Imitation

This document captures the repository and conversation state as of 2026-09-03.
It is written for a new Codex agent continuing the project on another machine.
It records what exists, what was run, what the results mean, and what was being
considered next. The authoritative behavioral constraints remain in
[`AGENTS.md`](AGENTS.md); the original research sequence is in
[`ICIL_TTT_IMPLEMENTATION_PLAN.md`](ICIL_TTT_IMPLEMENTATION_PLAN.md).

## 1. Snapshot

- Repository: `git@github.com:Ricvalp/icil-jax-rlbench.git`
- Active branch: `fast-weight-ttt`
- Branch HEAD before this handoff edit: `f63b448`
- Remote branch at the same commit: `origin/fast-weight-ttt`
- `master` and `origin/master`: `187fa8b`
- The worktree was clean before `SUMMARY.md` and the accompanying documentation
  updates were created.
- The branch is intentionally not backward compatible with the old RLBench
  direct-regression pretraining, parameter-MAML, or memory-MAML programs.
- Generated data, checkpoints, evaluations, videos, and W&B state are ignored by
  Git. They will not appear after cloning and are catalogued below.
- The current scientific decision point is the failure of the small KVB
  fast-weight mechanism to transfer to five held-out ML45 task families, despite
  successful same-family latent adaptation and a positive ordinary full-policy
  adaptation upper bound.

The branch commits after `master`, in order, are:

```text
08a83c1 Add TTT training step implementation and associated tests
8cbf923 Remove deprecated sbatch scripts and update test for TTT supernode encoder
57a5ccd Refactor code structure and remove redundant code blocks for readability
f7c6f9d Implement MetaWorld fast-weight TTT experiments
b5a0cd9 Add MetaWorld ML45 configurations and analysis scripts
b9ad658 Document phi-mujoco ML45 audit and wall-button correction
ae3d21e Document wall-button and stick-pull validation details
51ad597 Add conditioned query evaluation and training for MetaWorld ML45
f63b448 Add MetaWorld ML45 update analysis and visualization
```

## 2. User Intent and Working Preferences

The research objective is test-time imitation from one or a few demonstrations:

```text
support demonstrations of a hidden task
  -> gradient-based WRITE into a small transient fast state
  -> READ from that state while acting on independent query rollouts
```

The desired eventual interpretation is that slow parameters contain reusable
motion primitives such as reach, grasp, push, lift, and place, while fast
parameters select or compose those primitives for a demonstrated task.

Relevant preferences expressed during the conversation:

- Favor scientific clarity, simplicity, and readable code over preserving old
  features or backward compatibility on this branch.
- Keep simulator/data interoperability in `phi-mujoco` and policy-specific
  sampling, learning, adaptation, controls, and analysis in this repository.
- Use only public `phi_mujoco` contracts from the policy project.
- Keep visualization in dedicated modules rather than making training and
  evaluation entry points difficult to read.
- Provide extensive qualitative visualization for evaluations, especially
  matched trajectories before and after fast updates.
- Distinguish task family, task instance/latent, demonstration episode, and
  rollout start precisely when explaining experiments.
- Do not edit code in response to a question that only asks for an explanation
  or a command. Earlier in the conversation an unsolicited edit caused explicit
  user concern.
- W&B logging is useful for long runs, including preserving the same W&B run and
  step axis on checkpoint resume.
- Multiple-seed robustness is acknowledged as necessary for final evidence, but
  the user explicitly deferred it while investigating mechanism failures.

## 3. Repository Purpose

This is now a focused JAX research project for adaptation-only fast-weight TTT.
It has three layers:

1. A synthetic low-dimensional hidden-goal benchmark for fast autodiff,
   implementation, and visualization checks.
2. State-based MetaWorld policy experiments using ML1 Reach, ML1 Push, ML10,
   and ML45 through `phi-mujoco`.
3. Retained RLBench visual prerequisites: dense H5 loading, robotics-aware
   actions, and a self-contained space-time supernode/register encoder. There is
   no current RLBench TTT sampler, trainer, or online evaluator.

The main claim requires support to influence query predictions only through a
fast-state update. Direct support cross-attention, support FiLM, task labels,
language, and support-token initialization of READ memory are absent from the
main path.

The original implementation plan defines these gates:

- Gate 1: ordinary gradient adaptation from support can improve independent
  held-out query episodes.
- Gate 2: full-second-order implementation passes fixed-meta-batch, gradient,
  finite-difference, reset/carry, mask, sign, and JIT checks.
- Gate 3: meta-learned fast updates show support-specific held-out adaptation
  under wrong/shuffled/random controls and multiple seeds.
- RLBench follows only after a convincing state-based Gate 3.

## 4. Hard Contracts

The complete instructions are in [`AGENTS.md`](AGENTS.md). The most consequential
ones for this branch are:

- Do not import `icil`, `icil_jax_query_memory`, `diagnostics`, or upstream
  `metaworld`; use public `phi_mujoco` contracts.
- Do not restore removed legacy pretraining/MAML paths unless explicitly asked.
- Support can affect query inference only through explicit fast-state updates.
- Query prediction receives slow parameters, fast state, and query observation;
  it never receives query actions or support tensors.
- Support and query are separate episodes with different starts but the same
  hidden task latent.
- Fast state starts from meta-learned `W0` at every task/condition boundary and
  is frozen and reused across declared query rollouts after adaptation.
- WRITE loss creates the update but is not added to the outer objective.
- Full second order is the main method. `first_order=True` is a visible FOMAML
  ablation that stops computed inner gradients.
- A transient task-adapted fast state is never checkpointed or passed to the
  slow optimizer.

## 5. External `phi-mujoco` Dependency

`phi-mujoco` is an independent simulator interoperability repository, not a
subdirectory of this project. The expected layout is:

```text
Robotics/
  icil-jax-rlbench/
  phi-mujoco/
```

[`pyproject.toml`](pyproject.toml) declares:

```toml
[tool.uv.sources]
phi-mujoco = { path = "../phi-mujoco", editable = true }
```

The experiments in this handoff used:

- repository: `git@github.com:PHI-Lab-AI4I/phi-mujoco.git`
- branch: `main`
- commit: `e59d9b0`
- worktree state at handoff: clean and equal to `origin/main`

That repository provides:

- public integration specifications and task-bound environment factories;
- expert demonstration collection;
- integration-specific raw data and common processed caches;
- public complete-episode loading and task indices;
- task/family/reset contracts and the ML10/ML45 development protocols;
- the `EvaluationRunner`, policy interface, action projection, and optional video
  recording.

This policy repository provides:

- train-task-only normalization;
- explicit task/support-demo/support-time/query-demo/query-time sampling;
- query-only and meta-learning models;
- WRITE/READ logic and full second-order optimization;
- support perturbation controls, aggregation, and statistical summaries;
- policy adapters and policy-specific visualizations.

The relevant `phi-mujoco` integrations are:

- `metaworld_ml1_reach`
- `metaworld_ml1_push`
- `metaworld_ml10`
- `metaworld_ml45`

The ML45 integration was audited across all 50 MetaWorld families. Three
notable expert/reset corrections are recorded in `phi-mujoco` cache provenance:

- a geometry-aware wall-button controller for native starts where the upstream
  policy collides with the wall;
- restoration of the canonical disassemble nut quaternion after a MetaWorld
  3.1.1 double-reset artifact;
- a latched terminal stick-pull correction around a brittle success boundary.

## 6. Environment and Setup

The project uses Python 3.11 and `uv`. Core versions are pinned in
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock): JAX 0.10.0, Flax
0.12.7, Optax 0.2.8, NumPy 2.3.4, and `ml-collections` 1.1.0. `phi-mujoco`
pins MetaWorld 3.1.1 and MuJoCo 3.3.0.

Typical setup on the destination CUDA 12 machine is:

```bash
git clone git@github.com:PHI-Lab-AI4I/phi-mujoco.git ../phi-mujoco
git -C ../phi-mujoco checkout e59d9b0

uv sync --group metaworld --extra cuda12 --extra wandb
uv run --frozen --group metaworld --extra cuda12 python -c \
  "import jax; print(jax.default_backend(), jax.devices())"
```

Important `uv` behavior observed during the project: synchronization is exact.
Running `uv sync --group visualization` after installing CUDA and MetaWorld
removed the MetaWorld packages and JAX CUDA plugin because those options were
not repeated. Commands capable of synchronizing the environment should keep
`--group metaworld --extra cuda12` when those packages are required.

Other environment observations:

- `MUJOCO_GL=egl` is used for headless/offscreen simulation and videos.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` avoids JAX reserving all GPU memory.
- `Unable to initialize backend 'tpu': ... libtpu.so` is harmless on this
  machine.
- `No OpenGL_accelerate module loaded` is also only an optional acceleration
  warning.
- A warning that CUDA hardware exists but CUDA-enabled `jaxlib` is absent means
  the `cuda12` extra was omitted or removed, not that the GPU is unusable.
- Video writing dependencies are included by the visualization/MetaWorld groups.
  The earlier `imageio` MP4 backend error was addressed through the `uv` groups,
  not by ad hoc installation into the environment.

## 7. Terminology

- **Family:** a semantic MetaWorld environment class such as `push-v3`,
  `door-open-v3`, or `pick-place-wall-v3`.
- **Task instance / task latent:** one fixed native task configuration inside a
  family, usually indexed `000` through `049`. It includes goal/object/reset
  quantities designated as task latent by the public family contract.
- **Episode / demonstration:** one rollout for a fixed task instance with a
  separately sampled episode start/nuisance and seed.
- **Support:** expert episodes used to compute WRITE gradients.
- **Offline query:** different cached expert episodes from the same task used for
  the outer imitation loss or offline evaluation.
- **Closed-loop query:** fresh simulator rollouts from starts not present in the
  cache.
- **No update:** inference with the meta-learned initial fast state `W0`.
- **Correct support:** support episodes from the evaluated task instance.
- **Same-family wrong instance:** support from another latent in the same family.
- **Different-family support:** support from a different semantic family.
- **Native random vector:** MetaWorld's serialized per-task reset vector. Public
  `phi_mujoco` family contracts divide its fields into task latent and episode
  nuisance. It is provenance, not main-policy input.
- **`set_task()` / `for_task()`:** binding an integration/environment to one
  serialized task instance before reset and rollout. Policy code uses public
  `IntegrationSpec.for_task(task_id)` rather than upstream MetaWorld APIs.

## 8. MetaWorld Observation and Action Contract

Every current MetaWorld benchmark exposes:

- observation: 39D float state;
- action: 4D continuous Sawyer control `[dx, dy, dz, gripper]`;
- one action predicted and executed at every policy step;
- no wrist orientation action. ML45 remains 4D even for tasks involving object
  orientation because MetaWorld's Sawyer XYZ interface fixes end-effector
  orientation and scripted behavior accomplishes those tasks through Cartesian
  motion and gripper control.

Goal slots in the 39D state are zeroed for the hidden-goal integrations. Goals,
family names, task IDs, instance IDs, hashes, and composition metadata are not
fed to the main query-only or TTT policy.

Observations and actions are standardized using statistics fitted exclusively
from training-task episodes. The evaluator restores the checkpoint's exact
normalizer and validates cache SHA, normalizer ID, model config, and adaptation
config before running.

## 9. Task Splits and Local Caches

### ML1 Reach and ML1 Push

Each has 100 fixed hidden-goal tasks:

- 40 training goals;
- 10 validation goals;
- 50 official ML1 test goals.

Support and query share the fixed hidden goal but use different starts. For Push,
both hand and puck starts vary independently from the target.

### ML10 development protocol

The cache contains 750 native instances: 15 families times 50 instances.

- train: instances `000..039` from eight train families, 320 tasks;
- latent validation: `040..049` from those eight families, 80 tasks;
- family validation: all 50 instances from held-out `pick-place-v3` and
  `door-open-v3`, 100 tasks;
- untouched official test: five test families, 250 tasks.

The eight familiar development families are basketball, button-press-topdown,
drawer-close, peg-insert-side, push, reach, sweep, and window-open. The native
test families are door-close, drawer-open, lever-pull, shelf-place, and
sweep-into.

### ML45 development protocol

The cache contains 2,500 native instances: 50 families times 50 instances.

- train: instances `000..039` from 40 familiar families, 1,600 tasks;
- latent validation: `040..049` from those 40 families, 400 tasks;
- family validation: all 50 instances from five compositional family holdouts,
  250 tasks;
- untouched native test: all 50 instances from five official test families,
  250 tasks.

The five family-validation holdouts are:

```text
button-press-topdown-wall-v3
coffee-pull-v3
pick-place-wall-v3
plate-slide-back-side-v3
stick-push-v3
```

The untouched test families are:

```text
bin-picking-v3
box-close-v3
door-lock-v3
door-unlock-v3
hand-insert-v3
```

The official test split has not been evaluated. All current ML45 choices and
diagnostics use development train, latent-validation, or family-validation
splits.

### Local processed caches at handoff

These paths are ignored by Git and must be transferred or regenerated:

| cache | episodes | total transitions | SHA-256 | approximate size |
| --- | ---: | ---: | --- | ---: |
| `datasets/processed/metaworld_ml1_push-20260831T142652Z` | 2,400 | 161,738 | `079e0edc65f21f1b9a388c498f47915106774d196dfb4f242808f4b79be6e8ad` | 57 MB |
| `datasets/processed/metaworld_ml10-20260831T172940Z` | 6,000 | 461,444 | `c9879715a42dda7236bcbe971b9ec008aa0d8029a87b06515ff36795927ea67e` | 142 MB |
| `datasets/processed/metaworld_ml45-20260902T102333Z` | 10,000 | 763,523 | `dcffe0b2f2c2bc6dcf0b58c9ff630073ac01883524191cacf6b89ddbeba3a753` | 237 MB |

The ML1 Reach cache used by the checkpoints is outside this repository:

```text
/mnt/external_storage/robotics/metaworld_ml1_reach_gate1/processed/
  metaworld_ml1_reach-20260828T141247Z
```

Raw local collections also exist for Push, ML10, and ML45, but the processed
cache is sufficient for policy training/evaluation.

## 10. Data Loading and Batch Semantics

The policy-side loader is
[`icil_jax_rlbench/data/metaworld_hidden_goal.py`](icil_jax_rlbench/data/metaworld_hidden_goal.py).
It validates a public `phi_mujoco` collection bundle, loads the integration's
task index, builds task-disjoint episode splits, fits/restores normalization,
and supplies `MetaWorldTaskDataset` and `MetaWorldTaskSampler`.

A meta-batch has explicit axes:

```text
support.observation      [task, support_demo, time, 39]
support.action           [task, support_demo, time, 4]
support.next_observation [task, support_demo, time, 39]
support.write_mask       [task, support_demo, time]
support.outer_loss_mask  [task, support_demo, time]  # always zero

query.observation        [task, query_demo, time, 39]
query.action             [task, query_demo, time, 4]
query.outer_loss_mask    [task, query_demo, time]
```

Tasks are sampled family-uniformly and then instance-uniformly within family.
For each selected task, support and query episode indices are drawn together
without replacement. A common static horizon is selected for the batch. Real
timesteps occupy the beginning and padding is zero with masks false.

ML10/ML45 use horizon buckets `(64, 128, 256, 512)` to avoid compiling one shape
per exact episode length while avoiding universal 500-step padding. Padding is
not data and contributes neither WRITE nor READ loss.

The default ML45 meta-batch uses:

- `train.batch_size=8`: eight independently sampled task instances;
- two support episodes per task;
- two query episodes per task;
- four cached episodes consumed per selected task, with support/query disjoint.

This is not a flat batch of 32 episodes. The task and role axes remain explicit
because adaptation occurs separately for each task.

## 11. Query-Only Policy

The main implementation is in
[`models/fast_weight_ttt.py`](icil_jax_rlbench/models/fast_weight_ttt.py), with
query-only training in
[`train/metaworld_query_runner.py`](icil_jax_rlbench/train/metaworld_query_runner.py)
and [`train/query_only_step.py`](icil_jax_rlbench/train/query_only_step.py).

The unconditioned policy is a small MLP:

```text
39D normalized observation
  -> query encoder, two GELU MLP layers, hidden width 128
  -> 3D translation linear head
  -> 1D continuous gripper linear head
  -> standardized 4D action
```

The loss uses separate Huber terms:

```text
translation weight = 1.0
continuous gripper weight = 0.1
```

The policy predicts one action at a time. At simulation boundaries,
`MetaWorldJaxPolicy` denormalizes it and asks the bound integration to project it
to a valid simulator action.

The checkpoint parameter tree also contains dormant fast-WRITE/READ parameters
so a query-only checkpoint can initialize TTT. During query-only training READ
is disabled, so only the query encoder and action heads receive useful gradients.

## 12. Fast-Weight Architecture

Core code:

- [`models/fast_weight_ttt.py`](icil_jax_rlbench/models/fast_weight_ttt.py)
- [`train/ttt_step.py`](icil_jax_rlbench/train/ttt_step.py)
- [`train/metaworld_ttt_runner.py`](icil_jax_rlbench/train/metaworld_ttt_runner.py)

Default dimensions:

```text
observation_dim = 39
action_dim = 4
slow hidden_dim = 128
fast key/query/value dim = 32
fast hidden dim = 64
fast model = two-layer GELU MLP
```

The transient fast MLP contains 4,192 scalar coordinates:

```text
fc1 kernel [32, 64] + bias [64]
fc2 kernel [64, 32] + bias [32]
```

Its initial weights `W0` are slow/meta-learned parameters stored under
`fast_init`. Positive learned update rates are represented under
`inner_lr_raw`, at least one per fast tensor. Weight decay excludes `fast_init`,
`inner_lr_raw`, and the legacy absolute READ gate.

### KVB WRITE

At every valid support timestep:

```text
evidence = concat(observation[39], action[4], next_observation-observation[39])
evidence [82] -> support encoder -> normalized key K[32], value V[32]
loss = masked mean squared error(fast_model_W(K), V)
```

Support demonstrations are flattened in demo-major temporal order, padded to a
multiple of `write_segment_size=16`, and processed sequentially. The default is
one gradient update per 16-slot segment. A two-episode task can therefore make
many inner updates: the count is `ceil(2 * padded_horizon / 16)`. Segments with
only invalid padding generate zero update. This is full-second-order unrolling
in the main method.

### Support action-BC WRITE

The matched ablation replaces reconstruction with the ordinary robotics action
loss on support observations/actions while updating the same small fast state.
It does not adapt the full slow policy during deployment.

### READ

The query path is:

```text
query observation -> query encoder hidden h
h -> normalized query projection Q[32]
Q -> fast_model_W -> read projection -> residual in hidden space
hidden + residual -> translation/gripper heads
```

Two READ modes exist:

- `absolute_gated`: inject `tanh(gamma) * P_O(f_W(Q))`. The gate initializes
  near zero. This worked for KVB on ML1 but caused the original support-BC path
  to collapse because support-action gradients into fast state were suppressed.
- `delta`: inject `P_O(f_W(Q)) - P_O(f_W0(Q))`. At `W=W0` the READ contribution
  is exactly zero, preserving the query-only base policy, while its derivative
  with respect to adapted fast weights is nonzero. Delta READ is scientifically
  acceptable as a residual fast-adapter architecture and became the matched
  default for ML10 and ML45.

`predict_action()` has no support argument. Adaptation happens first through
`adapt_fast_state()`, after which the resulting fast tree is passed into query
prediction.

### Full second order and FOMAML

The main outer gradient differentiates query imitation loss through all
sequential WRITE gradients. This allows query loss to train the support encoder,
key/value projections, `W0`, learned rates, query projection, and READ path.

The FOMAML ablation applies `stop_gradient` to each computed inner gradient.
That removes or changes the expected meta-gradient paths into components that
act only through how WRITE gradients are formed. FOMAML is an ablation, not the
preferred way to learn a new WRITE objective.

## 13. Training, Checkpoints, and Resume

Entrypoints:

```text
icil_jax_rlbench.train_metaworld_query_only
icil_jax_rlbench.train_metaworld_conditioned_query
icil_jax_rlbench.train_metaworld_ttt
icil_jax_rlbench.train_ttt_state
```

Every run writes a unique run directory containing resolved config, provenance,
task splits, normalization, integrity report, periodic checkpoints, and
`last.pkl`. TTT adds a training contract.

A checkpoint stores:

- slow parameters, including meta-learned `W0` and learned inner rates;
- AdamW optimizer state;
- JAX RNG and current step;
- resolved config and provenance;
- train and validation sampler RNG states for MetaWorld TTT;
- cache hash, normalizer, model/adaptation config, and W&B run ID.

It explicitly does not store transient per-task adapted fast state.

TTT resume restores scientific configuration, optimizer state, RNGs, and
meta-batch settings from the checkpoint. The requested command can override
runtime controls such as final target step, logging/checkpoint intervals, output
directory, cache location, and W&B configuration. `train.num_steps` is the final
step, not a number of additional steps.

When a saved W&B run ID exists, resume uses that ID with `resume='must'`, and
metric logging explicitly uses the restored training step.

## 14. Evaluation Semantics

Ordinary Gate 1 is implemented in
[`eval/metaworld_hidden_goal_gate1.py`](icil_jax_rlbench/eval/metaworld_hidden_goal_gate1.py).
It copies a query-only policy and performs 50--200 ordinary SGD steps on support
action imitation, then evaluates different cached query episodes and fresh
closed-loop starts. `adapt_subset` can be:

- `action_heads`: translation and gripper heads;
- `query_policy`: query encoder and both heads;
- `all`: every parameter group is selected, although with READ disabled the
  support encoder, fast/KVB, and READ-only groups receive zero gradient. Thus
  `all` is effectively the same as `query_policy` in current Gate 1.

Fast-weight Gate 3 evaluation is in
[`eval/metaworld_hidden_goal_ttt.py`](icil_jax_rlbench/eval/metaworld_hidden_goal_ttt.py).
It always restores the checkpoint's adaptation semantics unless a declared
evaluation override is present. For each task and condition it resets to `W0`,
adapts on selected support, freezes fast state, and runs matched fresh simulator
episodes.

Implemented support conditions are centralized in
[`eval/support_controls.py`](icil_jax_rlbench/eval/support_controls.py):

- no update;
- correct support;
- wrong task / same-family wrong instance;
- different-family support;
- shuffled actions;
- shuffled time;
- observations only;
- actions only;
- duplicated support;
- random fast update matched to the correct update norm.

Summaries report offline losses, update norms, closed-loop success, pooled Wilson
intervals, and task-paired bootstrap intervals. Family-aware evaluations also
aggregate by family and declared motion phase. Matched conditions use identical
fresh rollout seeds.

## 15. Visualization and Information Diagnostics

Visualization is deliberately separate from quantitative training/evaluation.

Synthetic rollout visualization lives in
[`visualization/ttt_state.py`](icil_jax_rlbench/visualization/ttt_state.py),
[`visualization/state_rollouts.py`](icil_jax_rlbench/visualization/state_rollouts.py),
and [`visualization/state_plots.py`](icil_jax_rlbench/visualization/state_plots.py).
It can render trajectory overlays, videos, support and goals, action changes at
identical observations, vector fields, WRITE traces, and per-tensor fast deltas.

MetaWorld simulator videos are emitted by `phi_mujoco.evaluation` when an eval
config sets both `save_rollout_artifacts=True` and `record_video=True`.

The update-information diagnostic is implemented in
[`analysis/metaworld_update_information.py`](icil_jax_rlbench/analysis/metaworld_update_information.py)
with plotting in
[`visualization/metaworld_update_information.py`](icil_jax_rlbench/visualization/metaworld_update_information.py).
It extracts, without random projection:

- first WRITE gradient, 4,192D;
- final fast delta, 4,192D;
- oracle query gradient, 4,192D, diagnostic only;
- functional READ/action delta over fixed probe observations, 256D;
- raw support statistics, 494D.

It trains frozen family classifiers and within-family task-latent regressors,
computes update/query-gradient alignment and support-control cosine geometry,
and writes `features.npz`, `records.json`, `summary.json`, `report.md`, static
probe plots, and interactive PCA+t-SNE plots. t-SNE is exploratory; original-
space cosine and nearest-neighbor results are the quantitative evidence.

## 16. Synthetic Benchmark Results

The synthetic environment in
[`data/hidden_goal.py`](icil_jax_rlbench/data/hidden_goal.py) hides a 2D goal
from observations `[x, y, gripper, phase]`. It uses normalized planar delta plus
binary gripper actions and disjoint train/validation/test goals.

Gate 2 passed before MetaWorld scaling:

- fixed-meta-batch loss decreased from approximately `0.508807` to `0.002243`,
  a 99.56% reduction;
- forward WRITE improved query loss and reversed WRITE worsened it;
- full/FOMAML pathway, finite-difference, reset/carry, mask, eager/JIT, and pmap
  checks are covered by tests.

Two optimization seeds were trained and evaluated on validation. With two
supports, both reached 100% correct-support success from 0% no-update success;
wrong-task and random matched-norm updates remained at 0%. Shuffled-time support
was often still effective, which later reappeared as an important warning in
MetaWorld.

The user viewed generated trajectory videos and considered the synthetic
mechanism qualitatively functional. Seed robustness beyond these two runs was
deferred.

## 17. ML1 Reach Results

Local checkpoints, ignored by Git:

```text
outputs/metaworld_ml1_reach_query_only/ml1_reach_query_only_20260828-142159/last.pkl
outputs/metaworld_ml1_reach_ttt/ml1_reach_kvb_full_20260831-094006/last.pkl
outputs/metaworld_ml1_reach_ttt/ml1_reach_kvb_fomaml_20260831-104305/last.pkl
outputs/metaworld_ml1_reach_ttt/ml1_reach_action_bc_full_20260831-104535/last.pkl
```

Gate 1 on ten validation goals with 20 rollouts per goal:

- no update: 21.5% success;
- correct support: 66.5%;
- gain: +45 points;
- wrong-task support: 20%;
- shuffled actions: 1.5%;
- observations only: 1%.

The full-second-order KVB checkpoint was then evaluated on all 50 untouched ML1
test goals with two supports and 20 rollouts per goal:

- no update: 9.0%;
- correct support: 97.6%;
- wrong support: 10.0%;
- correct-support gain: +88.6 points.

Matched test evaluations also found:

- FOMAML KVB: 23.7% no update, 70.3% correct, 14.9% wrong;
- support-BC WRITE: 12.4% no update, 100% correct, 11.9% wrong.

This is the clearest successful held-out-goal result. It is a single semantic
family and therefore does not establish held-out-family composition.

## 18. ML1 Push Results

Local checkpoints:

```text
outputs/metaworld_ml1_push_query_only/ml1_push_query_only_20260831-143949/last.pkl
outputs/metaworld_ml1_push_ttt/ml1_push_kvb_full_20260831-145159/last.pkl
```

The full-second-order KVB validation evaluation used ten goals and 20 rollouts
per goal. Key results:

| supports | no update | correct | wrong task | shuffled actions | shuffled time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40.0% | 40.0% | 30.5% | 40.0% | 42.5% |
| 2 | 40.0% | 96.0% | 41.5% | 50.0% | 92.5% |
| 4 | 40.0% | 93.0% | 63.0% | 86.5% | 79.5% |

Two supports provide a strong adaptation gain, but shuffled-time support is
almost equally effective and shuffled actions become effective with four
supports. The mechanism therefore works behaviorally on Push but does not yet
demonstrate strong sensitivity to temporal/action pairing.

## 19. ML10 Results and Delta READ

Local checkpoints:

```text
outputs/metaworld_ml10_query_only/ml10_query_only_20260901-070031/last.pkl
outputs/metaworld_ml10_ttt/ml10_kvb_full_20260901-080757/last.pkl
outputs/metaworld_ml10_action_bc/ml10_action_bc_full_20260901-084741/last.pkl
outputs/metaworld_ml10_ttt/ml10_action_bc_delta_read_full_20260901-111205/last.pkl
outputs/metaworld_ml10_ttt/ml10_kvb_delta_read_full_20260901-124523/last.pkl
```

The query-only Gate 1 screen on ten balanced family-validation tasks used two
supports and five rollouts per task:

- no update: 4%;
- correct full query-policy adaptation: 30%;
- same-family wrong-instance support: 14%;
- shuffled/observations-only: 0%.

The original absolute-gated KVB showed only a small latent-validation advantage:
58.3% correct versus 50% no update, with shuffled actions also 58.3%. The
original absolute-gated support-BC model was completely collapsed: every
condition produced identical predictions, losses, and success. This motivated
delta READ.

With delta READ and one closed-loop episode per selected task:

- support-BC latent screen: 25% no update, 87.5% correct, 62.5% same-family
  wrong, 25% shuffled actions;
- KVB latent screen: 50% no update, 100% correct, 62.5% same-family wrong,
  62.5% shuffled actions;
- KVB family-validation screen: 0% no update, 20% correct, 10% same-family
  wrong, 0% shuffled/different/random.

These were small screens, not final estimates. They suggested strong adaptation
on familiar families but weak transfer to held-out families. The user suspected
that ten ML10 train families permit an easy family-labeling/memorization
solution, motivating ML45.

The ML10 update-information diagnostic compared delta KVB and delta support-BC.
For both, final fast deltas made familiar family identity perfectly linearly
decodable and retained within-family latent information. KVB's final delta was
very similar under correct and shuffled-time support (cosine 0.990) and under
correct and shuffled-action support (0.906). Support-BC was more action-pairing
sensitive (shuffled-action final-delta cosine 0.392), but still almost invariant
to time shuffling (0.983). This supported the interpretation that updates encode
task/family statistics more readily than causal trajectory structure.

## 20. ML45 Training and Results

### Trained checkpoints

```text
outputs/metaworld_ml45_query_only/
  ml45_query_only_20260902-124305/last.pkl

outputs/metaworld_ml45_ttt/
  ml45_kvb_delta_read_full_kdx104kg/last.pkl
```

Both are step 100,000, seed 0. The KVB checkpoint was initialized from the
query-only checkpoint. Its W&B run ID is `kdx104kg` in project
`ricvalp/icil-metaworld`.

The direct family-conditioned and family-plus-task-latent oracle baselines are
implemented but were not trained before this handoff. They are intentionally
separate checkpoint types and cannot initialize KVB. Family-validation is not a
fair default for the family-one-hot model because held-out family coordinates
were never optimized.

During KVB training, `write_loss` and `query_loss_before` rose. This was not by
itself a training bug: WRITE reconstruction is not directly minimized by the
outer objective, and meta-training can sacrifice the no-update `W0` behavior to
improve post-update behavior. Evaluation, rather than those curves alone,
determined whether that trade was useful.

### Familiar-family latent validation

The 40-family confirmatory evaluation used instance `040` from each familiar
family with three fresh rollouts per condition:

- no update: 45.0%;
- correct support: 76.7%;
- same-family wrong instance: 66.7%;
- shuffled actions: 70.8%;
- shuffled time: 70.8%;
- different-family support: 45.8%;
- random matched-norm update: 36.7%.

An expanded latent sweep evaluated three unseen latent instances per familiar
family, 120 task instances total, with one rollout each:

- no update: 47.5%;
- correct support: 80.0%;
- same-family wrong instance: 70.0%;
- shuffled actions: 74.2%;
- different-family support: 44.2%;
- random matched-norm update: 42.5%.

Thus KVB has strong positive adaptation within familiar meta-training families,
but much of the gain survives wrong-instance and shuffled-action controls. The
fast update appears dominated by family-level behavior selection rather than
precise latent inference.

### Held-out-family validation

The first ten-task/three-rollout screen gave 60% no-update and 40% correct
support. The expanded run evaluated ten latent instances from each of the five
held-out families, 50 task instances, one matched fresh rollout each:

| condition | success |
| --- | ---: |
| no update | 66% (33/50) |
| correct support | 40% (20/50) |
| same-family wrong instance | 42% (21/50) |
| different-family support | 54% (27/50) |
| shuffled actions | 50% (25/50) |
| random matched-norm update | 46% (23/50) |

Correct support reduced offline imitation loss slightly but reduced closed-loop
success by 26 points, with a paired bootstrap interval of roughly `[-38,-14]`
points. The checkpoint provenance was checked and is the intended
`ml45_kvb_delta_read_full_kdx104kg/last.pkl` at step 100,000.

Family-level KVB behavior was highly uneven:

- button-press-topdown-wall: 100% no update and 100% correct;
- coffee-pull: 10% no update and 0% correct;
- pick-place-wall: 30% no update and 0% correct;
- plate-slide-back-side: 90% no update and 0% correct;
- stick-push: 100% no update and 100% correct.

This is a held-out-family Gate 3 failure. The high no-update score reflects the
meta-trained slow policy and easy/similar families; it is not evidence that
adaptation works.

### Expanded ordinary Gate 1 upper bound

The expanded ordinary-adaptation diagnostic used the step-100,000 query-only
checkpoint, the same 50 held-out family-validation instances and rollout seed,
two support episodes, 100 SGD steps at learning rate 0.01, gradient clipping at
1.0, and `adapt_subset=all` (effectively query encoder plus action heads):

| condition | success | offline query loss |
| --- | ---: | ---: |
| no update | 40% (20/50) | 0.48885 |
| correct support | 58% (29/50) | 0.06115 |
| same-family wrong instance | 44% (22/50) | 0.08793 |
| shuffled actions | 2% (1/50) | 0.33739 |
| observations only | 0% (0/50) | 0.42918 |

Correct support gained 18 points over no update. The reported paired 95% CI was
`[+7.2,+28.8]` points. It rescued nine task instances and broke none. Wrong-
instance support gained four points with a CI spanning zero. Update norms were
similar across correct/wrong/control conditions, so magnitude alone does not
explain behavior.

The gain was concentrated by family:

| held-out family | no update | correct | wrong instance |
| --- | ---: | ---: | ---: |
| button-press-topdown-wall | 100% | 100% | 30% |
| coffee-pull | 0% | 10% | 10% |
| pick-place-wall | 0% | 0% | 0% |
| plate-slide-back-side | 0% | 80% | 80% |
| stick-push | 100% | 100% | 100% |

The result shows that support action gradients contain behaviorally useful
information for held-out families. It does not show broad instance-specific
adaptation: plate-slide supplies most gains and works equally with another
instance from the same family. With only five held-out families, a task-level
confidence interval is optimistic for claims about family-level generalization.

## 21. ML45 Update-Information Diagnostic

The completed diagnostic used the KVB checkpoint and selected:

- 400 train tasks;
- 400 latent-validation tasks;
- 50 family-validation tasks;
- two independent support samples per task;
- two support episodes per sample;
- seven support conditions;
- 11,900 feature rows;
- full vectors with no random projection.

The ignored local output is:

```text
eval_outputs/metaworld_ml45_update_information/last_20260902-170215/
```

Key quantitative findings:

- Correct-support final fast deltas classify familiar family at 100% and
  regress familiar within-family latent with normalized RMSE 0.263.
- READ action deltas similarly classify family at 99.9% and have latent nRMSE
  0.332.
- In original fast-delta space, same-family/different-instance cosine is about
  0.907 overall and 0.951 for held-out families; different-family cosine is
  about 0.210 overall and 0.104 for held-out families.
- Family 1-nearest-neighbor accuracy excluding the same instance is 100% in
  train, latent-validation, and family-validation splits.
- Final fast-delta effective rank is about 2.98 on familiar train/latent tasks
  and 2.54 on family holdouts, despite 4,192 coordinates.
- Correct and same-family-wrong held-out updates are almost parallel: final-
  delta cosine 0.955.
- Correct and shuffled-time held-out updates are nearly identical: cosine 0.994.
- Correct and shuffled-action held-out updates also remain similar: cosine
  0.851.
- On held-out families the mean offline query gain is slightly negative
  (`-0.00796`) and update direction has near-zero/slightly wrong oracle-query
  alignment (`update_query_gradient_cosine=-0.0156`).
- On familiar train/latent tasks, mean offline query gain is positive but the
  update/query-gradient diagnostic is still weak.

The diagnostic therefore does not show an information bottleneck in the narrow
sense: family and latent information are recoverable from the final update.
It shows an **actionability/alignment problem**. KVB updates form compact family
signatures, are insufficiently sensitive to temporal/action pairing, and fail
to map novel family structure into a query-improving control residual.

Interactive t-SNE plots visually show clear family clusters and place held-out
families near familiar relatives. These plots are useful for exploration but
the conclusions above come from original-space probes/cosines. The full feature
archive is approximately 385 MB and the complete diagnostic directory about
424 MB.

## 22. Current Scientific Interpretation

The evidence at hand separates several hypotheses:

1. **The implementation can meta-learn through WRITE.** Synthetic Gate 2 and
   ML1 Reach rule out a globally broken second-order implementation.
2. **A small fast state can solve hidden latents within one family.** ML1 Reach
   and Push demonstrate this strongly.
3. **The ML45 support demonstrations contain usable information.** Ordinary
   support-BC adaptation of the query encoder and heads improves held-out-family
   success from 40% to 58% under matched controls.
4. **Current KVB does not turn that information into transferable held-out-
   family control updates.** It improves familiar-family latents but hurts the
   expanded family holdout from 66% to 40%.
5. **The learned update is heavily family-coded.** Family identity is almost
   perfectly recoverable, updates within family are nearly parallel, and the
   functional delta has low effective rank.
6. **Generic reconstruction is weakly tied to causal trajectory structure.**
   Shuffling time barely changes final updates; shuffling action pairing changes
   them more, but often not enough to remove familiar-family gains.
7. **Simply adding more task families did not by itself fix transfer.** ML45 is
   substantially more diverse than ML10, yet the familiar/held-out-family gap
   remains. More data may eventually help, but the current result points first
   to WRITE alignment and/or fast-module placement/capacity.

The branch has not passed broad held-out-family Gate 3. Moving to end-to-end
RLBench now would mix this mechanism failure with visual representation and
robotics-action confounders.

## 23. Last Discussed Next Experiments

The immediate diagnostic sequence proposed immediately before this handoff was:

### A. Action-head-only ordinary Gate 1

The completed ML45 Gate 1 adapted the query encoder and action heads. A cheap
remaining run adapts only `translation_head` and `gripper_head` on the same 50
tasks and matched rollout seed. Re-running `query_policy` is unnecessary because
current `all` is effectively identical while READ is disabled.

This separates whether conventional useful adaptation requires changing the
observation representation or can be expressed entirely at the output heads.

The exact proposed command was:

```bash
export PHI_MUJOCO_ML45_CACHE="$PWD/datasets/processed/metaworld_ml45-20260902T102333Z"
QUERY_CKPT="$PWD/outputs/metaworld_ml45_query_only/ml45_query_only_20260902-124305/last.pkl"

MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld --extra cuda12 python \
  -m icil_jax_rlbench.eval_metaworld_ml45_gate1 \
  --config=icil_jax_rlbench/configs/eval_metaworld_ml45_gate1.py \
  --config.checkpoint_path="$QUERY_CKPT" \
  --config.output_dir=eval_outputs/metaworld_ml45_gate1/action_heads_expanded \
  --config.split=family_validation \
  --config.max_tasks=50 \
  --config.support_episodes=2 \
  --config.offline_query_episodes=2 \
  --config.inner_steps=100 \
  --config.inner_lr=0.01 \
  --config.adapt_subset=action_heads \
  --config.closed_loop_episodes=1 \
  --config.closed_loop_base_seed=3000000
```

This run had not started at handoff.

### B. ML45 support-action-BC WRITE with delta READ

[`configs/metaworld_ml45_action_bc.py`](icil_jax_rlbench/configs/metaworld_ml45_action_bc.py)
already supplies a matched full-second-order ablation. It uses the same small
fast state and delta READ as ML45 KVB, changing only WRITE from reconstruction
to support action imitation.

The proposed training command was:

```bash
export PHI_MUJOCO_ML45_CACHE="$PWD/datasets/processed/metaworld_ml45-20260902T102333Z"
export ICIL_ML45_QUERY_CHECKPOINT="$PWD/outputs/metaworld_ml45_query_only/ml45_query_only_20260902-124305/last.pkl"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --frozen --group metaworld --extra cuda12 python \
  -m icil_jax_rlbench.train_metaworld_ttt \
  --config=icil_jax_rlbench/configs/metaworld_ml45_action_bc.py \
  --config.wandb.enable=True \
  --config.wandb.project=icil-metaworld \
  --config.wandb.name=ml45_action_bc_delta_read_full
```

This run had also not started at handoff.

The intended interpretation matrix was:

- If support-BC fast WRITE transfers while KVB does not, generic KVB
  reconstruction is the main bottleneck.
- If output-head-only ordinary adaptation works but support-BC fast WRITE does
  not, the learned meta-optimization/update transfer is the main suspect rather
  than raw representational capacity.
- If output-head-only adaptation fails but query-encoder-plus-head adaptation
  works, fast weights likely need better placement or capacity inside the query
  encoder rather than only the current isolated branch.
- If support-BC fast WRITE also fails across held-out families, changing only
  the WRITE target is unlikely to be sufficient.

### C. Likely next new WRITE objective

If the ablations isolate KVB alignment as the bottleneck, the original plan's
next objective is one causally aligned future-effect target rather than a large
mixture of auxiliary losses. Candidate progression discussed:

1. KVB plus future normalized state delta;
2. future end-effector or object displacement where family contracts expose a
   semantically consistent quantity;
3. terminal relation / sparse phase anchors;
4. cross-demonstration masked trajectory reconstruction;
5. autoregressive next-state/next-action WRITE as a low-dimensional alternative.

The reason is diagnostic, not aesthetic: current KVB writes remain almost
unchanged when temporal order is shuffled. A future-effect target should force
the update to represent action-conditioned consequences shared across episodes.

No future-effect or trajectory-reconstruction objective has been implemented
yet.

## 24. Explicit Conditioning Controls

Two ML45 query-only capacity controls were added in
[`data/metaworld_conditioning.py`](icil_jax_rlbench/data/metaworld_conditioning.py):

- `family`: appends a 50-way family one-hot;
- `family_task_latent`: appends family one-hot, up to three normalized task-
  latent values declared by the public reset contract, and a three-value mask.

The oracle input is therefore 95D: 39 state + 50 family + 3 latent + 3 mask.
Latent normalization uses training tasks only. It excludes task IDs, instance
indices, hashes, and episode nuisance.

These controls answer whether the small MLP can represent the behavior when
task information is explicit on familiar families. They do not directly solve
held-out-family adaptation: family one-hot coordinates for the five held-out
families are untrained. The evaluator rejects unseen-family use unless
`allow_unseen_families=True` is explicitly acknowledged.

At handoff, implementation/tests existed but no conditioned checkpoint had been
trained.

## 25. Retained RLBench Work

The old end-to-end RLBench direct-regression and MAML runners were deliberately
removed from this branch. Retained building blocks are:

- [`data/h5_cache.py`](icil_jax_rlbench/data/h5_cache.py): dense RLBench H5
  reader for episode groups with `xyz`, `valid`, `state`, `action`, and optional
  `rgb`/`mask_id`;
- [`models/ttt_supernode.py`](icil_jax_rlbench/models/ttt_supernode.py):
  self-contained space-time supernodes and event registers;
- [`models/robotics_actions.py`](icil_jax_rlbench/models/robotics_actions.py):
  delta translation, continuous 6D rotation, geodesic rotation loss, and gripper
  BCE components;
- tests for visual register occupancy, query-action isolation, and action
  geometry.

The later intended visual path is:

```text
local point/state/action/time segment
  -> space-time supernodes and small event registers
  -> support K/V and sequential WRITE into small fast state
  -> query scene/state registers and fast READ
  -> translation + 6D rotation + gripper heads
```

There is no runnable RLBench TTT experiment on the current branch.

## 26. Historical Context Before This Branch

The repository originally contained RLBench direct-regression pretraining and
parameter-/memory-MAML. Early in the conversation the user evaluated and
resumed several legacy checkpoints, including:

- pretrained supernode space-time checkpoint `kz71poab/step_0300000` trained
  with the contextual dense-v4 cache;
- parameter-MAML checkpoints `2a239p79/step_0040000`,
  `zxc4paha/step_0045000`, and later `kps4bfa9` checkpoints;
- Snellius jobs on four H100 GPUs.

Legacy resume logic was investigated so scientific hyperparameters, optimizer
state, RNG, and W&B step/run identity would continue from checkpoints. Those
legacy paths were later intentionally deleted from `fast-weight-ttt`; this
history does not imply they should be restored here.

The old parameter-MAML inner gradients were extracted and visualized with
t-SNE. Batch-size-4 and batch-size-64 gradient datasets showed strong family or
task clusters and overlap between behaviorally related tasks. A 64-example
inner batch under full second-order MAML exceeded H100 memory (an allocation of
roughly 72.8 GiB per GPU failed). Rematerialization/checkpointing and memory-
lighter trials could fit some settings but were slow. These experiments helped
motivate a small explicit fast state and the hypothesis that short action-BC
WRITE gradients may identify a task label without learning a transferable
adaptation rule.

The user then supplied `ICIL_TTT_IMPLEMENTATION_PLAN.md`, requested a clean
branch, and explicitly approved removal of the old paths. The current branch is
the result.

## 27. Tracked File Map

Primary documentation:

- [`README.md`](README.md): setup and runnable workflows.
- [`ICIL_TTT_IMPLEMENTATION_PLAN.md`](ICIL_TTT_IMPLEMENTATION_PLAN.md): original
  phased scientific plan and acceptance gates.
- [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md): implementation-level
  design summary; this handoff supersedes its older experiment-status section.
- [`AGENTS.md`](AGENTS.md): branch-specific coding/scientific constraints.

Core data/model/training/evaluation:

- [`data/metaworld_hidden_goal.py`](icil_jax_rlbench/data/metaworld_hidden_goal.py)
- [`models/fast_weight_ttt.py`](icil_jax_rlbench/models/fast_weight_ttt.py)
- [`train/metaworld_query_runner.py`](icil_jax_rlbench/train/metaworld_query_runner.py)
- [`train/metaworld_ttt_runner.py`](icil_jax_rlbench/train/metaworld_ttt_runner.py)
- [`train/ttt_step.py`](icil_jax_rlbench/train/ttt_step.py)
- [`eval/metaworld_hidden_goal_gate1.py`](icil_jax_rlbench/eval/metaworld_hidden_goal_gate1.py)
- [`eval/metaworld_hidden_goal_ttt.py`](icil_jax_rlbench/eval/metaworld_hidden_goal_ttt.py)
- [`eval/metaworld_policy.py`](icil_jax_rlbench/eval/metaworld_policy.py)
- [`eval/support_controls.py`](icil_jax_rlbench/eval/support_controls.py)

Analysis and visualization:

- [`analysis/metaworld_update_information.py`](icil_jax_rlbench/analysis/metaworld_update_information.py)
- [`visualization/metaworld_update_information.py`](icil_jax_rlbench/visualization/metaworld_update_information.py)
- [`visualization/ttt_state.py`](icil_jax_rlbench/visualization/ttt_state.py)

Important ML45 configs:

- [`configs/metaworld_ml45_query_only.py`](icil_jax_rlbench/configs/metaworld_ml45_query_only.py)
- [`configs/metaworld_ml45_ttt_base.py`](icil_jax_rlbench/configs/metaworld_ml45_ttt_base.py)
- [`configs/metaworld_ml45_kvb.py`](icil_jax_rlbench/configs/metaworld_ml45_kvb.py)
- [`configs/metaworld_ml45_action_bc.py`](icil_jax_rlbench/configs/metaworld_ml45_action_bc.py)
- [`configs/metaworld_ml45_fomaml.py`](icil_jax_rlbench/configs/metaworld_ml45_fomaml.py)
- [`configs/eval_metaworld_ml45_gate1.py`](icil_jax_rlbench/configs/eval_metaworld_ml45_gate1.py)
- [`configs/eval_metaworld_ml45_ttt.py`](icil_jax_rlbench/configs/eval_metaworld_ml45_ttt.py)
- [`configs/analyze_metaworld_ml45_information.py`](icil_jax_rlbench/configs/analyze_metaworld_ml45_information.py)

Tests:

- `tests/test_fast_weight_ttt.py`
- `tests/test_hidden_goal.py`
- `tests/test_metaworld_policy_path.py`
- `tests/test_metaworld_push_policy_path.py`
- `tests/test_metaworld_ml10_policy_path.py`
- `tests/test_metaworld_ml45_policy_path.py`
- `tests/test_metaworld_update_information.py`
- `tests/test_ttt_checkpoint.py`
- `tests/test_ttt_supernode.py`
- `tests/test_robotics_actions.py`
- `tests/test_ttt_visualization.py`

## 28. Generated Artifacts and Machine Transfer

`.gitignore` excludes `datasets/`, `outputs/`, `eval_outputs/`, `wandb/`, and
other run products. At handoff their approximate local sizes were:

```text
datasets/     751 MB  # raw + processed Push/ML10/ML45
outputs/      145 MB  # checkpoints and run metadata
eval_outputs/ 1.9 GB  # summaries, videos, diagnostic feature archives
wandb/         14 MB  # local W&B files
```

The minimal artifact set for continuing the immediate ML45 experiments is:

```text
datasets/processed/metaworld_ml45-20260902T102333Z/
outputs/metaworld_ml45_query_only/ml45_query_only_20260902-124305/last.pkl
outputs/metaworld_ml45_ttt/ml45_kvb_delta_read_full_kdx104kg/last.pkl
```

The corresponding resolved configs/provenance in those run directories are
small and should travel with the checkpoints. The two `last.pkl` files are each
about 850 KB; the processed ML45 cache is about 237 MB.

For preserving all quantitative evidence without the large t-SNE feature
archive, also transfer the `summary.json` and `resolved_eval_config.json` files
from:

```text
eval_outputs/metaworld_ml45_ttt/latent_validation_kdx104kg_confirmatory/
eval_outputs/metaworld_ml45_ttt/latent_validation_kdx104kg_latent_sweep/
eval_outputs/metaworld_ml45_ttt/family_validation_kdx104kg_expanded/
eval_outputs/metaworld_ml45_gate1/family_validation_expanded/
eval_outputs/metaworld_ml45_update_information/last_20260902-170215/
```

The ML45 information diagnostic's `features.npz` is 385 MB. It is needed to
recompute new probes/embeddings without rerunning model extraction, but not to
understand the already copied summary/report.

## 29. Known Gotchas

- Top-level executable modules are named, for example,
  `icil_jax_rlbench.eval_metaworld_ml1_reach_ttt`, not
  `icil_jax_rlbench.eval.ttt_metaworld_gate1`. Earlier commands using the latter
  produced `No module named ...`.
- A config's default `max_tasks` is a balanced total task limit, not a count per
  family. `max_tasks=50` on ML45 family validation selects ten instances from
  each of five families.
- `closed_loop_episodes` is per selected task and per condition. Ten tasks, six
  conditions, and three episodes means 180 simulator episodes.
- Evaluation progress text such as `task 1/10: validation-000` refers to a fixed
  hidden task latent/goal, not merely an arbitrary rollout.
- Support trajectories are padded to a static horizon for JAX; masks ensure
  padding does not train the model. The number of WRITE updates is determined by
  padded support slots and segment size, not a single global `inner_steps` field.
- Full second-order cost scales with the number of sequential WRITE segments.
- `shuffled_time` permutes complete support transitions in time. KVB's
  per-transition reconstruction is structurally close to order-invariant, so
  weak sensitivity is not surprising; this is part of the motivation for a
  future-effect objective.
- `random_update_matched_norm` applies a random fast-state displacement with the
  same total norm as the correct-support update. It tests whether merely moving
  away from `W0` explains gains.
- Offline action loss and closed-loop success can disagree sharply. On ML45
  held-out families KVB slightly improved offline loss while destroying
  plate-slide closed-loop success.
- Results with one rollout per task are screens. Task-paired statistics are
  useful, but five held-out families remain only five semantic generalization
  units.
- The ML45 query-only and KVB results are seed 0 only.
- Do not interpret t-SNE separation as standalone evidence. Use full-space
  probes, cosine geometry, paired controls, and closed-loop outcomes.

## 30. Verification State

The standard repository checks are:

```bash
uv run --frozen --group metaworld --extra cuda12 python \
  -m compileall -q icil_jax_rlbench tests

uv run --frozen --group metaworld --extra cuda12 pytest -q

rg -n "^(from|import) (icil|icil_jax_query_memory|diagnostics|metaworld)(\\.|\\s|$)" \
  icil_jax_rlbench tests
```

The final `rg` should return no matches. The test suite contains 11 test modules
and 51 explicitly named tests, including parameterized cases. After the handoff
documentation edits, compilation succeeded, the forbidden-import scan returned
no matches, and pytest reported `52 passed in 163.51s`.

## 31. Short Continuation State

For a new Codex agent that needs only the immediate context:

- `fast-weight-ttt` at `f63b448` is the intended branch.
- Clone `phi-mujoco` beside it at `e59d9b0`.
- Transfer the ML45 processed cache and query-only/KVB checkpoints because Git
  ignores them.
- ML1 hidden-goal adaptation works; ML45 familiar-family adaptation works but is
  weakly support-specific.
- Expanded ML45 held-out-family KVB fails: 66% no update versus 40% correct.
- Expanded ordinary query-policy adaptation passes: 40% no update versus 58%
  correct, with shuffled-action and observations-only controls at 2% and 0%.
- ML45 update vectors strongly encode family/latent but are low-rank,
  order-insensitive, and misaligned with held-out query improvement.
- The immediate pending experiments are action-head-only Gate 1 and matched
  full-second-order support-BC fast WRITE on ML45.
- Their outcomes determine whether to change fast-weight placement/capacity or
  implement a causal future-effect WRITE objective.
- The official ML45 test split and RLBench TTT remain untouched.
