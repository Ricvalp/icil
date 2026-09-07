# Implementation Plan: Fast-Weight Test-Time Training for In-Context Robot Imitation

> **Branch decision (2026-08-27):** after the initial implementation, the user
> explicitly requested removal of the legacy direct-regression pretraining,
> parameter-MAML, and memory-MAML paths for cleanliness. That decision supersedes
> this document's legacy-preservation requirements on `fast-weight-ttt`.

**Audience:** Codex working directly in [`Ricvalp/icil-jax-rlbench`](https://github.com/Ricvalp/icil-jax-rlbench)  
**Repository branch inspected:** public `master`, as available on 2026-08-26  
**Primary objective:** make gradient-based test-time adaptation from one or a few demonstrations work for in-context robot imitation, first in a controlled state-based benchmark and then in RLBench with the space-time supernode encoder.  
**Status of prior experimental claims:** do not use old reported numbers as evidence. Treat every baseline and adaptation result as unverified until reproduced under the instrumentation specified below.

---

## 1. Mission and scientific claim

The project should continue to target **imitation from in-context demonstrations**. Long demonstrations and long-horizon execution are possible extensions and motivations, not replacements for imitation learning.

The intended deployment process is:

\[
\text{demonstrations of an unseen task}
\xrightarrow{\text{gradient-based WRITE}}
\text{adapted fast weights}
\xrightarrow{\text{READ}}
\text{closed-loop imitation on a new rollout}.
\]

The core claim to establish is:

> A robot policy can meta-learn a small fast-weight state that is updated from one or a few demonstrations at deployment and that enables closed-loop imitation of an unseen task or task variation, without relying on a massive synthetic ICIL corpus or direct retention of the entire demonstration context.

The first paper-worthy mechanism result must demonstrate that:

1. the support demonstration changes a small set of effective policy weights;
2. correct-support updates improve a different query rollout;
3. wrong or shuffled support does not produce the same improvement;
4. support information cannot bypass the update through direct cross-attention or support-summary conditioning;
5. the result holds on held-out task latents, not merely held-out trajectories from a memorized latent.

The long-context extension should be pursued only after this mechanism works. Its role is then:

> Fast weights compress long demonstrations or rollout histories into a constant-size adapted state, while the task remains imitation of the demonstrated behavior.

---

## 2. Design decisions that should remain fixed initially

Codex should preserve the existing methods as baselines, but implement the new method through a separate, clearly named execution path. Do not silently mutate the semantics of the existing parameter-MAML or memory-MAML modes.

The following decisions define the initial restart:

- **Use a small fast-weight module**, not full-policy MAML.
- **Use full second-order differentiation** through the fast update for the main method.
- **Disable direct support conditioning** in the adaptation-only mechanism experiment.
- **Use a learned reconstruction-style WRITE objective first**, specifically key-value reconstruction.
- **Keep outer imitation loss as the READ objective.**
- **Start with state-based MetaWorld or an equivalent controlled low-dimensional environment.**
- **Do not introduce point clouds, diffusion, quaternion prediction, or long contexts until the diagnostic gates pass.**
- **Retain the space-time supernode encoder for the RLBench phase.**
- **Treat FOMAML, support-conditioned ICIL, action-BC WRITE, and trajectory reconstruction as baselines or ablations.**
- **Make support and query separate episodes with different initial states.**
- **Reset fast weights exactly at task boundaries and carry them from support into query.**
- **Evaluate closed-loop success, not only offline action loss.**

Codex may choose the cleanest low-level implementation after inspecting the repository, but these scientific semantics should not be changed without documenting the reason.

---

## 3. Current repository audit and likely failure modes

Before editing, Codex should inspect at least:

- `icil_jax_rlbench/models/direct_regression_policy.py`
- `icil_jax_rlbench/models/encoders.py`
- `icil_jax_rlbench/models/config.py`
- `icil_jax_rlbench/train/step.py`
- `icil_jax_rlbench/train/runner.py`
- `icil_jax_rlbench/data/sampler.py`
- `icil_jax_rlbench/data/action_representation.py`
- `icil_jax_rlbench/configs/base.py`
- the existing space-time-supernode and support-summary-FiLM configs
- the existing evaluation scripts and checkpoint-loading paths

The public repository already contains direct-regression ICIL, parameter MAML/FOMAML, memory MAML/FOMAML, Perceiver and supernode encoders, RLBench H5 loading, distributed training, checkpointing, and evaluation. Preserve all of these as reproducible baselines.

### 3.1 Direct support-conditioning bypass

The current policy can expose support information to the action decoder through several direct paths:

- support-token cross-attention;
- pooled support-summary FiLM/AdaLN modulation;
- separate support trajectory/action tokens;
- combinations such as `query_film_support` and `traj_and_memory`.

The trajectory tokenizer is intentionally designed to provide direct action-shape evidence. This is useful for feed-forward ICIL, but it invalidates a mechanism experiment intended to prove that a gradient update is necessary. The outer loss can simply learn to read the demonstrations directly and ignore the adaptation update.

**Required change:** add a strict adaptation-only conditioning mode in which the query policy has no access to support tokens, support summaries, support trajectories, task IDs, or language. The only object transferred from support to query is the adapted fast-weight state.

Keep the existing direct-conditioning modes unchanged as baselines.

### 3.2 Same short-horizon action loss for WRITE and READ

The current parameter-MAML and memory-MAML paths largely use the same global action regression loss for both the inner and outer objectives. This gives the inner update local information about reproducing a few support actions, but it may not identify the task-level latent shared with a different query rollout.

**Required change:** separate the concepts and configuration of:

- `write_objective`: the loss used only to update fast weights from support;
- `read_objective`: the imitation loss used to train and evaluate query behavior;
- optional auxiliary slow losses, which must be explicit rather than implicitly mixed into WRITE.

### 3.3 Brittle update parameterization

The current implementation exposes scalar inner learning rates, global clipping, and broad parameter masks. Different tensors and action components can have radically different gradient scales. One global step size and one global clip coefficient can make the useful update nearly zero or let one component dominate.

**Required change:** the main fast-weight module should have:

- a very small parameter tree;
- a learned positive step size, at least per fast layer and optionally per tensor;
- per-fast-tensor or fast-tree clipping, separate from slow-model clipping;
- logged update norms and effective learning rates;
- no accidental weight decay or optimizer state carried into test-time adaptation unless intentionally designed.

### 3.4 Correlated support/query sampling

Sampling several nearby windows from one support episode gives a low-diversity gradient. Sampling several outer windows from one query episode can make the meta-gradient episode-specific. The model may learn to correct a particular trajectory rather than infer a latent task.

**Required change:** aggregate support evidence across demonstrations and phases, and compute the outer loss across multiple independent query episodes whenever possible.

### 3.5 FOMAML cannot properly learn a new WRITE mechanism

For a learned WRITE rule,

\[
W' = W - \eta\nabla_W L_{\text{write}}(S;W,\psi),
\]

where \(\psi\) includes learned key/value projections or a reconstruction target, the outer objective must differentiate through how \(\psi\) changes the inner gradient. Applying `stop_gradient` to the inner gradient removes this route.

FOMAML remains a useful approximation for learning an initialization, but it is not the appropriate default when the central contribution is **learning what to write**.

**Required change:** implement full second-order differentiation through the small fast module. Keep FOMAML as an explicit ablation using the same architecture and data.

### 3.6 Robotics-unaware action objective

A global MSE over translation, quaternion components, and gripper mixes unrelated units and geometries. This is especially damaging in MAML because the inner loss defines the update direction itself.

**Required change:** use component-specific action heads, normalizers, losses, weights, and gradient diagnostics. Details are specified in Phase 8.

---

## 4. Target architecture

### 4.1 Slow policy

The slow policy should contain:

1. a current-observation encoder;
2. a query/action decoder;
3. one or more insertion points for a small fast-weight memory branch;
4. a near-zero residual gate controlling the contribution of the fast branch.

The slow parameters are optimized across meta-training tasks and are not modified at deployment. Depending on the phase, they include:

- state or visual encoders;
- query decoder;
- action heads;
- key, value, and query projections for WRITE/READ;
- the initial fast weights;
- learned inner learning rates;
- the residual gate;
- optional future-effect heads.

### 4.2 Fast model \(f_W\)

\(f_W\) is a small neural network whose parameters \(W\) act as the writable memory. Start with the simplest useful choices:

- a linear map; or
- a two-layer MLP with width 128–256 and a smooth nonlinearity.

Do not initially distribute LoRA adapters throughout the whole transformer. The fast model should be small enough that full second-order unrolling is cheap and easy to inspect.

The initial fast weights \(W_0\) are meta-learned. At the beginning of every new task, set \(W \leftarrow W_0\). Carry \(W\) through all support demonstrations and into all query rollouts belonging to that task, according to the evaluation protocol. Reset before a different task.

### 4.3 Input representation \(x_t\)

\(x_t\) denotes the encoded evidence available at a support timestep or support segment.

For the state-based benchmark, it can contain:

- low-dimensional environment state;
- proprioception;
- demonstrated action;
- optional normalized time or phase;
- optional transition information such as \(s_{t+1}-s_t\).

For RLBench, it should contain a small number of tokens produced from:

- space-time supernode or register tokens;
- a separate proprioception/end-effector token;
- a demonstrated action token when actions are available;
- optional contact, gripper-transition, or phase tokens.

The WRITE module should not receive thousands of raw point tokens at every update. The supernode encoder should first compress a local temporal segment into a small register set.

### 4.4 Learned keys, values, and queries

Construct:

\[
K_t=P_K(\operatorname{LN}(x_t)),\qquad
V_t=P_V(\operatorname{LN}(x_t)),\qquad
Q_t=P_Q(\operatorname{LN}(x_t)),
\]

where \(P_K,P_V,P_Q\) are slow, meta-learned projections.

The projections may be token-type-aware. For example, support actions may contribute to values but not to query keys. Query READ tokens must never include the unknown query action.

Separate projections are important: an identity autoencoder would otherwise make WRITE trivial and unrelated to downstream imitation.

### 4.5 WRITE update

The initial WRITE objective is:

\[
L_{\text{KVB},t}(W)
=
\frac{1}{N_t d_v}
\left\|f_W(K_t)-V_t\right\|_2^2.
\]

Update:

\[
W_{t+1}
=
W_t-\eta_t\nabla_{W_t}L_{\text{KVB},t}(W_t).
\]

\(\eta_t\) should be positive by construction and meta-learned. Initially use one value per fast layer or parameter group. Add more granularity only if diagnostics justify it.

The reconstruction loss primarily defines the fast update. By default, do **not** add it directly to the slow outer objective. The query imitation loss should teach the projections what information is useful to write. A directly weighted slow reconstruction loss may encourage trivial reconstruction rather than useful task adaptation; if added for stability, it must be an explicit ablation.

### 4.6 READ operation

After processing support evidence, query tokens produce:

\[
m_t=f_{W_S}(Q_t),
\]

where \(W_S\) is the fast state after all support updates.

Inject this memory into the upper policy through a gated residual:

\[
h'_t=h_t+\tanh(\gamma)\odot P_O(m_t).
\]

Initialize \(\gamma\) near zero so that adding the new branch does not destroy a competent base policy. Log gate magnitudes during training. A permanently near-zero gate means the model is ignoring TTT; a rapidly saturated gate may indicate instability.

### 4.7 Outer READ objective

The outer objective remains imitation on separate query episodes:

\[
L_{\text{read}}
=
\mathbb{E}_{(o^Q,a^Q)}
L_{\text{act}}
\left(
\pi_{\theta,W_S}(o^Q),a^Q
\right).
\]

Support timesteps may update \(W\), but their outer action losses should be masked in the clean asymmetric experiment. Query losses train the slow policy and the WRITE mechanism through the unrolled updates.

---

# 5. Twelve-phase implementation and validation plan

Each phase has a purpose, repository work, experiments, acceptance criteria, and a stop/go decision. Codex should not proceed merely because the implementation runs. The gates are intended to localize failure causes.

---

## Phase 1 — Repository stabilization and experiment ledger

### Purpose

Create a reproducible base before changing model semantics.

### Repository work

- Record the exact commit, dependency environment, JAX/Flax/Optax versions, CUDA setup, dataset paths, and existing checkpoint formats.
- Read any repository-local Codex or agent instructions before editing.
- Preserve existing pretraining, parameter-MAML, memory-MAML, and evaluation entry points.
- Add a clear experiment identifier that records:
  - config contents after resolution;
  - git commit and dirty state;
  - random seeds;
  - dataset/task splits;
  - normalization statistics identifier;
  - checkpoint parent, if any;
  - adaptation mode and reset policy.
- Add a concise implementation summary document when the work is complete, including changed files, new configs, commands, tests, and known limitations.

### Required baseline smoke tests

Run short deterministic jobs that establish only that the old paths still execute:

- direct-regression pretraining;
- direct support-conditioned evaluation;
- parameter-MAML step;
- memory-MAML step;
- checkpoint save/restore;
- one closed-loop evaluation episode if currently supported.

Do not treat the resulting numbers as reproduced scientific results.

### Acceptance criteria

- Old modes still instantiate and run.
- A resolved config and exact experiment provenance are stored with every run.
- New work can be developed without overwriting old semantics.

### Stop/go

Do not continue if the current branch cannot be reproduced sufficiently to distinguish a regression from a new-method failure.

---

## Phase 2 — Instrument support paths and create an adaptation-only policy mode

### Purpose

Ensure the experiment can prove that support information acts through fast-weight updates rather than feed-forward conditioning.

### Repository work

Add independent switches for:

- support-token cross-attention;
- support-summary FiLM/AdaLN;
- support trajectory/action tokens;
- support memory initialization;
- fast-weight WRITE;
- fast-weight READ;
- query-history conditioning.

The new adaptation-only mode must satisfy:

- no support token reaches the query decoder;
- no pooled support summary reaches the query decoder;
- no support action token reaches the query decoder;
- support affects the query only by changing \(W\);
- query observations are processed normally;
- the same base architecture can be run with no fast updates for a matched control.

Add a runtime assertion or test that changing support tensors while freezing the resulting fast weights leaves query outputs unchanged. Conversely, changing fast weights should change query outputs.

### Recommended mode taxonomy

Preserve and clearly label at least:

1. `direct_icil`: existing feed-forward support conditioning, no test-time update;
2. `param_maml_legacy`: existing parameter-MAML behavior;
3. `memory_maml_legacy`: existing memory-token adaptation;
4. `ttt_adaptation_only`: support reaches query only through \(W\);
5. `ttt_hybrid`: fast weights plus direct conditioning, introduced only after the adaptation-only result works.

### Acceptance criteria

- Automated tests verify no direct support bypass in `ttt_adaptation_only`.
- Correct reset and carry semantics are testable independently of the full policy.
- Existing direct-conditioning baselines are unchanged.

### Stop/go

Do not interpret any MAML/TTT result until this isolation is verified.

---

## Phase 3 — Construct the controlled MetaWorld imitation benchmark

### Purpose

Create a task where adaptation is necessary, support demonstrations identify the latent task, and query observations alone do not.

The first benchmark should not be broad ML10 unseen-task-family generalization. Start with an ML1-like hidden-goal task variation.

### Recommended first task

Use a simple state-based reach or push family with a hidden latent \(c_\tau\), such as a target position. Requirements:

- support and query episodes share \(c_\tau\);
- support and query initial states differ;
- the query observation does not expose \(c_\tau\);
- the support trajectory identifies \(c_\tau\);
- success requires acting according to \(c_\tau\);
- task IDs and one-hot labels are absent;
- the expert is reliable enough that imitation noise is not the dominant problem.

MetaWorld’s ML1 benchmark is explicitly intended for few-shot adaptation to goal variations within one task, while its meta-learning environments are partially observable. Use the official split where suitable, or construct an explicit custom split with the same semantics.

### Data split

Use disjoint task-latent splits, for example:

- training goals/latents;
- validation goals/latents for hyperparameter selection;
- test goals/latents never used in slow training.

The exact count can be adjusted, but start with enough task variation for meta-learning without creating a massive dataset. A reasonable initial order is tens to low hundreds of training latents, with multiple trajectories per latent.

### Episode construction

For each meta-task:

- sample \(K\) support episodes from distinct initial states;
- sample at least two independent query episodes from other initial states;
- preserve full trajectories for sequential WRITE;
- store task latent only for evaluation and debugging, never as model input;
- store expert success and terminal state;
- make wrong-support episodes easy to generate by selecting another latent from the same family.

### Action representation

For the initial state benchmark, use the environment’s native low-dimensional action, typically Cartesian delta plus gripper. Do not introduce orientation prediction.

### Dataset integrity checks

Codex should implement diagnostics that test:

- whether a query-only model can infer the latent from the current observation;
- whether a simple classifier can infer the latent from support trajectories;
- whether support and query initial-state distributions overlap appropriately;
- whether success is possible from the retained observation channels;
- whether train/test latent splits are truly disjoint;
- whether support and query episodes accidentally share identical trajectories or seeds.

### Acceptance criteria

The benchmark must exhibit all three:

1. an oracle policy with the latent succeeds;
2. a query-only policy is materially limited;
3. a support-conditioned or ordinarily fine-tuned policy can improve using support.

### Stop/go

If query observations reveal the latent, the benchmark cannot establish adaptation. If support does not identify the latent, no adaptation method can solve it. Fix the benchmark before touching MAML hyperparameters.

---

## Phase 4 — Diagnostic Gate 1: ordinary adaptation upper bound

### Purpose

Test whether support gradients contain information that generalizes to different query episodes before attempting meta-learning.

### Procedure

Train a competent query-only base policy over the training task distribution. For held-out latents, perform ordinary gradient-based adaptation on support demonstrations for 50–200 steps, then evaluate different query episodes.

Test a small set of candidate adaptable subsets:

- last action head only;
- top FiLM/AdaLN or adapter block;
- a small standalone fast MLP;
- as an upper bound only, all parameters of a very small state policy.

Use support action imitation for this gate because it is directly available and requires no learned WRITE projections. The purpose is not to choose the final WRITE objective; it is to establish that the task, data, parameterization, and support labels permit cross-episode adaptation.

### Conditions

Evaluate the same checkpoint with:

- no update;
- correct support;
- wrong-task support;
- shuffled support actions;
- support observations with actions removed, when applicable.

### Metrics

- query action loss before and after adaptation;
- closed-loop query success;
- correct-support adaptation gain;
- wrong-support adaptation gain;
- update norm per tensor;
- support loss versus query loss over adaptation steps;
- overfitting curve: support improves while query eventually degrades;
- sensitivity to number of support demonstrations.

### Acceptance criteria

Proceed only if at least one small adaptable subset shows repeatable improvement on held-out query episodes from correct support. A useful preliminary gate is:

- statistically positive correct-support gain across seeds;
- wrong/shuffled support provides a much smaller gain or causes degradation;
- the gain survives different query starts;
- improvement occurs before severe support overfitting.

Do not require one-step adaptation yet. This gate establishes the existence of a useful update direction.

### Failure interpretation

If ordinary adaptation cannot improve query behavior, the likely problem is one or more of:

- task latent not identifiable from support;
- support/query mismatch;
- action representation;
- adapted parameter subset;
- support objective;
- insufficient base-policy competence;
- expert or normalization errors.

MAML will not fix these. Stop and repair them.

---

## Phase 5 — Diagnostic Gate 2: full-second-order correctness on one meta-batch

### Purpose

Verify the meta-learning implementation independently of generalization.

### Minimal setup

Use:

- a tiny state encoder and action policy;
- a small number of fixed task latents;
- one or two support episodes and query episodes per latent;
- a tiny fast model;
- one or two WRITE steps;
- one device, without `pmap`, initially.

### Required checks

1. **Overfit one fixed meta-batch.** Post-update query loss should approach the achievable minimum.
2. **Check update sign.** Reversing the inner update should worsen results.
3. **Check reset semantics.** Fast weights must be identical at the start of every task.
4. **Check carry semantics.** Fast weights after support must be exactly those used for query.
5. **Check support outer-loss mask.** Support labels must not accidentally contribute to the outer imitation objective in the asymmetric experiment.
6. **Check second-order paths.** The outer loss must produce nonzero gradients for:
   - key projection;
   - value projection;
   - query projection;
   - initial fast weights;
   - learned inner rate;
   - residual gate.
7. **Finite-difference check.** On a tiny network, compare selected meta-gradients with finite differences.
8. **FOMAML contrast.** Intentionally stop the inner gradient and verify that parameters acting only through WRITE lose or substantially change their gradient signal.
9. **Numerical stability.** Detect NaNs, exploding higher-order gradients, and silent zero gradients.
10. **JIT consistency.** Compare eager/non-jitted and jitted results on a tiny deterministic batch.

### Instrumentation

Log for every inner step:

- WRITE loss;
- gradient norm by fast tensor;
- update norm by fast tensor;
- \(\|W_t-W_0\|\);
- learned step sizes;
- gate values;
- query loss before and after each update;
- meta-gradient norm for every slow WRITE parameter.

### Acceptance criteria

- The fixed meta-batch can be overfit.
- Meta-gradients are numerically plausible and nonzero where expected.
- Full second-order and FOMAML differ in the expected pathways.
- Reset/carry/masking tests pass.

### Stop/go

Do not start expensive generalization training if this phase fails. A failure here is implementation or optimization, not a scientific negative result.

---

## Phase 6 — Implement the fast-weight KVB WRITE mechanism

### Purpose

Replace broad policy adaptation and short-horizon BC WRITE with a small, learned fast-weight memory.

### Repository structure

Codex should choose names consistent with the repository, but a clean separation would be:

- a dedicated fast-weight memory module under `models/`;
- a TTT policy wrapper or mode that composes the base policy and fast module;
- a dedicated TTT training step under `train/` rather than adding more branching to the legacy step indefinitely;
- TTT-specific samplers or batch structures under `data/`;
- TTT-specific configs under the existing config convention;
- separate evaluation entry points for state and RLBench adaptation.

Possible conceptual modules are:

- fast-state initialization;
- support token construction;
- key/value/query projections;
- one pure WRITE update;
- sequential support scan;
- query READ;
- residual integration;
- action objective;
- metrics.

### Sequential support processing

Process demonstrations in temporal order. The first implementation may update once per:

- timestep in the state benchmark; or
- short segment/register set in RLBench.

Do not collapse the entire support set into one pooled vector before WRITE. The method should be able to accumulate evidence over time and over demonstrations.

### Full second order

The outer gradient must differentiate through the WRITE updates. In JAX terms, avoid `stop_gradient` inside the unrolled update for the main mode. Keep the fast state small, use rematerialization/checkpointing where necessary, and delay distributed execution until the single-device gradients are verified.

### Stabilization choices

Start with:

- one or two WRITE steps per segment;
- learned positive rate initialized conservatively;
- LayerNorm before key/value/query projections;
- normalized reconstruction targets;
- fast-gradient clipping independent of slow-gradient clipping;
- near-zero residual gate;
- a small \(\|W-W_0\|^2\) regularizer only if drift is excessive;
- no weight decay on the transient fast state unless explicitly intended.

### Acceptance criteria

The module must:

- update from support without direct support conditioning;
- preserve differentiability through the update;
- produce a measurable query-output change;
- run both with full MAML and an otherwise matched FOMAML ablation;
- expose all diagnostics specified above.

---

## Phase 7 — Diagnostic Gate 3: adaptation-only generalization

### Purpose

Demonstrate real test-time learning on held-out task latents before adding point clouds.

### Main experiment

Meta-train the state policy and KVB writer on training latents. Evaluate held-out latents using the same checkpoint under:

1. no fast updates;
2. correct support updates;
3. wrong-latent support updates;
4. shuffled support actions;
5. temporally shuffled support;
6. support observations without actions;
7. random fast updates matched in norm;
8. direct feed-forward support-conditioning baseline;
9. a recurrent-memory baseline with a matched state dimension;
10. legacy parameter MAML and memory MAML where feasible.

### Required evaluation protocol

- Support and query starts must differ.
- Use multiple independent query episodes per latent.
- Report multiple random training seeds.
- Evaluate support counts \(K\in\{1,2,4\}\), or the closest feasible set.
- Evaluate zero, one, and multiple inner updates.
- Reset fast weights between task latents.
- Preserve the adapted state across all query episodes only if this is declared part of the protocol; otherwise re-run support adaptation for each query evaluation.

### Success signature

The decisive pattern is:

\[
\text{correct support}
>
\text{no update}
\approx
\text{wrong/shuffled support}.
\]

Additionally:

- adaptation gain should increase with useful support information;
- the gain should not be explained solely by update norm;
- a wrong task should move behavior toward the wrong target or otherwise produce task-specific degradation;
- full second order should be at least as effective as FOMAML for learning the WRITE projections, with the pathway difference visible in diagnostics.

### Suggested go threshold

Before RLBench, require a robust adaptation-specific gain, not a marginal offline improvement. A reasonable internal gate is either:

- at least a 15-point absolute closed-loop success gain over no update; or
- at least a 30% relative reduction in query imitation loss,

with confidence intervals excluding zero and wrong/shuffled gains substantially smaller than the correct gain. These thresholds are project-management gates, not claims that must appear in a paper.

### Stop/go

- If Gate 1 passed but Gate 3 fails, investigate meta-optimization and WRITE alignment.
- If the one-batch gate passes but held-out latents fail, increase task-distribution diversity or reduce fast-model capacity before increasing architectural complexity.
- Do not move to RLBench until support-specific adaptation is clear.

---

## Phase 8 — WRITE-objective ladder and ablations

### Purpose

Determine what a robot policy should write from demonstrations without launching an uncontrolled search.

Implement objectives in this order.

### A. Primary: learned key-value reconstruction

\[
L_{\text{write}}=L_{\text{KVB}}
+\lambda_W\|W-W_0\|^2.
\]

This is the first main method. The key/value/query projections are learned entirely through the query imitation objective.

### B. Baseline: support action imitation

Use the robotics-aware action loss on support actions as WRITE. This provides a direct comparison with legacy MAML while using the same fast module and data. It answers whether the improvement comes from the module or the reconstruction rule.

### C. Future-latent or future-effect reconstruction

Once KVB works, add one aligned target:

\[
L_{\text{future}}
=
\left\|g(m_t)-\operatorname{sg}(z_{t+\Delta})\right\|^2.
\]

Candidate targets, in priority order:

- future state delta in MetaWorld;
- future end-effector displacement;
- future object displacement;
- gripper/contact transition;
- terminal object-to-goal relation;
- a future supernode/register encoding.

Use a stop-gradient target encoder or EMA target encoder if needed to avoid representation collapse. Add one target at a time.

### D. Sparse masked trajectory reconstruction

Do not reconstruct every raw timestep. Use phase-normalized sparse anchors and task-relevant quantities:

- 16–64 end-effector or object waypoints;
- terminal relation;
- gripper/contact phase;
- object displacement.

Prefer cross-demo leave-one-out reconstruction:

- WRITE from one or more context demonstrations;
- reconstruct masked anchors of a different demonstration of the same task latent;
- provide only its initial state and permitted observations;
- prevent target leakage through support tokens.

This tests the original whole-trajectory idea in a task-aligned form.

### E. Autoregressive next-token WRITE

For the low-dimensional benchmark, optionally tokenize a causal state-action sequence and use next-state/next-action prediction as WRITE. This aligns WRITE and READ more directly and may be a clean alternative if KVB is unstable.

### Objective-selection diagnostics

For every WRITE objective, log:

- support WRITE loss;
- query imitation gain;
- cosine similarity between the WRITE gradient and an oracle query gradient in fast-weight space, used only as a diagnostic during training/evaluation;
- correct versus wrong-support update directions;
- update norm and effective rank;
- information retained about the task latent, measured by a frozen probe if useful;
- sensitivity to support order and amount.

The oracle query-gradient alignment is not available at deployment and must never enter the method. It is a diagnostic for understanding whether WRITE produces a useful direction.

### Decision rule

Do not combine many losses initially. Use:

1. KVB;
2. KVB plus one future-effect term;
3. sparse trajectory reconstruction;
4. support BC.

Choose based primarily on support-specific closed-loop query improvement, not reconstruction loss.

---

## Phase 9 — Robotics-aware actions, sampling, and evaluation

### Purpose

Remove robotics-specific confounders before transferring the mechanism to RLBench.

### MetaWorld action objective

Use separate terms for:

- Cartesian delta, with normalized Huber or L1;
- gripper, with BCE if binary or Huber if genuinely continuous.

Do not add rotation to the state benchmark.

### RLBench action representation

Use separate heads and losses for:

1. translation;
2. rotation;
3. gripper.

Recommended initial representation:

- translation as current-EEF-relative or incremental delta;
- rotation as continuous 6D representation, converted to a rotation matrix and then quaternion at the environment interface;
- gripper as binary logits.

A normalized quaternion with a sign-invariant geodesic loss is an acceptable alternative, but raw quaternion component MSE should not be the default.

A suitable objective is:

\[
L_{\text{act}}
=
\lambda_p L_{\text{Huber}}(\Delta p)
+\lambda_R L_{SO(3)}(R,\hat R)
+\lambda_g L_{\text{BCE}}(g,\hat g).
\]

Normalize each term from global training-set statistics. Initialize weights so that gradient norms are of comparable scale, then log them separately.

### Sampling

Replace purely uniform local-window sampling with phase-aware sampling where possible:

- approach/free motion;
- pre-contact;
- contact/grasp;
- manipulation/transport;
- release/terminal.

At minimum, oversample gripper transitions, contact-adjacent frames, and terminal effects. The support writer should not be trained primarily on easy free-space motion.

Use multiple support demonstrations and multiple independent query episodes. Do not estimate a meta-gradient from many near-duplicate windows of one episode if alternatives exist.

### Normalization and leakage

- Fit all normalizers only on training data.
- Do not compute new-task normalization statistics from its full expert dataset unless explicitly counted as support information.
- Version and store normalization statistics with checkpoints.
- Check quaternion convention, camera frame, world frame, EEF frame, and action horizon offsets.
- Verify that support action indices align with support observations.

### Closed-loop evaluation

Offline losses are diagnostics only. Main metrics:

- success rate;
- partial completion or stage completion;
- task-specific final relation error;
- intervention/failure type;
- rollout length;
- robustness to changed starts and object poses.

Use receding-horizon execution and report how many predicted actions are executed before replanning.

---

## Phase 10 — Integrate RLBench and the space-time supernode encoder

### Purpose

Transfer the validated adaptation mechanism to visual 3D imitation while retaining the strongest architectural work already in the repository.

### Role of the space-time supernode encoder

Use it as a **local spatiotemporal event encoder**, not as the sole memory for an arbitrarily long demonstration.

Recommended pipeline:

\[
\text{8--16 frame point-cloud segment}
\rightarrow
\text{space-time supernodes}
\rightarrow
\text{small register/event token set}
\rightarrow
\text{sequential TTT WRITE}
\rightarrow
W_t.
\]

The query policy then reads \(W_t\) from the current observation.

### Preserve and initially reuse

- existing dense H5 reader;
- point-cloud cache and sampling infrastructure;
- mask-balanced sampling;
- space-time coordinates and demo IDs;
- current supernode pooling and refinement as a baseline;
- current observation encoder and action decoder where compatible.

### Remove from the adaptation-only route

- direct support-token cross-attention;
- support-summary FiLM from unadapted support;
- direct support trajectory/action tokens to the query decoder;
- task IDs;
- any support cache reused by query except the fast state.

### Recommended encoder refinements

Implement these one at a time and preserve the original encoder as an ablation.

#### 1. Separate proprioception from per-point features

The current design repeats robot state for every sampled point before pooling. Create separate robot-state and EEF tokens, then fuse them with supernodes after spatial pooling. This prevents globally repeated state channels from dominating local geometric features.

#### 2. Learn or normalize space-time bandwidths

The current supernode assignment combines spatial and temporal distances with fixed temperatures. Make positive bandwidths learnable or initialize them from empirical nearest-neighbor statistics. Log their values and assignment entropy.

#### 3. Add occupancy and valid-centre semantics

Track effective pooled mass per supernode. Mark or downweight duplicated/empty centres. Add occupancy as a token feature or attention weight. Log the effective number of occupied/distinct supernodes.

#### 4. Produce register/event tokens

Do not send all supernodes directly through every WRITE step. Pool or attend them into a small set of register tokens that capture:

- relevant object geometry;
- EEF-object relation;
- motion/event phase;
- object displacement;
- contact/gripper transition.

The register count should be small enough for sequential processing, initially perhaps 8–32.

#### 5. Retain token-type and time information

Keep distinct embeddings for:

- visual supernode;
- state/EEF;
- demonstrated action;
- phase/time;
- demo identity;
- optional segmentation class.

### RLBench task progression

Do not immediately train on broad held-out task families. Use this order:

1. one task family with held-out goals/object poses;
2. several compositionally related task families;
3. held-out variations within families;
4. only then, held-out semantic task families.

The first RLBench result should mirror the MetaWorld hidden-latent test: support and query share a task variation but differ in initial state and scene realization.

### Acceptance criteria

- Correct support improves closed-loop query success with direct support paths disabled.
- Wrong support is task-specific and does not act as generic regularization.
- The space-time encoder outperforms or complements a simpler framewise encoder under the same TTT mechanism.
- State-only and point-cloud failures are distinguishable through matched experiments.

---

## Phase 11 — Long-demonstration and long-horizon extension

### Purpose

Establish a setting where fast weights provide a real advantage over explicit context, while remaining an imitation-learning project.

### When to begin

Only after adaptation-only MetaWorld works and at least one RLBench task variation shows support-specific improvement.

### Long demonstration protocol

Stream a long demonstration in segments:

1. encode a local segment with the supernode encoder;
2. WRITE its register tokens into \(W_t\);
3. discard the raw segment from active context;
4. continue through the demonstration;
5. execute the query rollout using the final adapted state.

Compare context lengths from short to prohibitively long. Keep fast-state size constant.

### Truncated second-order training

For long sequences:

- unroll full second-order gradients within manageable segments;
- carry the numerical fast state across segment boundaries;
- detach the carried state at deliberate truncated-BPTT boundaries;
- document the truncation length and its interpretation;
- progressively increase training sequence length;
- verify that the short-sequence exact implementation and truncated implementation agree where they overlap.

### Query-time continual updates

Initially freeze the fast state during query execution to isolate support-driven adaptation. Later test continuing to WRITE from:

- current observations;
- past executed actions;
- detected events;
- corrections or interventions.

This is an extension, not part of the minimal ICIL claim.

### Required baselines

- short explicit context;
- full explicit attention context, where computationally feasible;
- feed-forward pooled support summary;
- GRU/LSTM or another recurrent state with matched state size;
- gated/retrieval memory;
- TTT fast weights;
- oracle keyframe/event context.

Compare:

- success versus demonstration length;
- memory use;
- inference latency;
- state size;
- sensitivity to irrelevant or corrupted history;
- ordering sensitivity;
- performance on Markovian control tasks where memory should not help.

### Task requirements

Use tasks where history is genuinely necessary:

- target or order shown only earlier;
- visually aliased stages;
- an object leaves view;
- repeated substeps whose progress is not observable from the current frame;
- demonstration-defined subtask ordering;
- perturbations requiring remembering whether an earlier stage succeeded.

Do not claim a memory benefit on tasks that are Markovian from the current point cloud.

---

## Phase 12 — Final experimental matrix, paper decision, and fallback rules

### Minimal experimental matrix

The final project should include, at minimum:

| Method | Direct support context | Test-time gradients | WRITE objective | Full second order | Purpose |
|---|---:|---:|---|---:|---|
| Query-only BC | No | No | None | No | Lower baseline |
| Direct ICIL | Yes | No | None | No | Existing feed-forward baseline |
| Legacy parameter MAML | As currently implemented | Yes | Support BC | Full/FOMAML | Reproduce old path |
| Legacy memory MAML | As currently implemented | Yes | Support BC | Full/FOMAML | Reproduce old path |
| Recurrent memory | No direct support at query | No gradient WRITE | Learned recurrence | No | Matched memory baseline |
| Fast-weight TTT, FOMAML | No | Yes | KVB | No | First-order ablation |
| Fast-weight TTT, full | No | Yes | KVB | Yes | Main method |
| Fast-weight TTT + future effect | No | Yes | KVB + future | Yes | Aligned WRITE extension |
| Fast-weight TTT + sparse trajectory | No | Yes | Trajectory reconstruction | Yes | Alternative WRITE |
| Fast-weight TTT + support BC | No | Yes | Action imitation | Yes | Objective baseline |
| Hybrid TTT + direct ICIL | Yes | Yes | Best WRITE | Yes | Final practical model, only after proof |

### Support-control matrix

Every adaptation model should be evaluated with:

- correct support;
- no support/no update;
- wrong-task support;
- shuffled actions;
- shuffled temporal order;
- observations only;
- actions only where meaningful;
- duplicated support;
- increasing support count;
- support corruption/noise;
- random update matched in norm.

### Statistical reporting

- Use multiple training seeds.
- Use enough closed-loop episodes to report confidence intervals.
- Report per-task and aggregate results.
- Predefine the principal comparison and avoid selecting only favorable tasks.
- Log both offline and closed-loop metrics, but make success the main result.
- Report compute and context/memory scaling for long-demonstration experiments.

### Failure tree

Use the following interpretation rather than indiscriminate hyperparameter search.

#### Ordinary adaptation fails

Problem is probably benchmark construction, data, objective, parameter subset, normalization, or base policy. Do not proceed to meta-learning.

#### Ordinary adaptation succeeds, one-meta-batch overfit fails

Problem is implementation, autodiff, masking, update semantics, or optimizer stability.

#### One-meta-batch overfit succeeds, held-out latent adaptation fails

Problem is task-distribution coverage, fast-model capacity, overfitting, or WRITE/READ alignment.

#### State TTT works, RLBench fails

Problem is perception, point-cloud tokenization, action representation, support sampling, closed-loop robustness, or visual data scale. Keep the state result and isolate the visual failure.

#### KVB fails but support BC works

The learned reconstruction pathway is not aligned or its second-order gradient is defective. Inspect projection gradients, query access to memory, and target collapse.

#### KVB works but trajectory reconstruction fails

Treat dense/sparse trajectory reconstruction as an unnecessary negative ablation; do not force it into the method.

#### Gradient TTT remains unstable after all gates pass

Use a WIZARD-like fallback:

- train small task-specific LoRA/FiLM experts on training tasks;
- train the space-time support encoder to predict an adapter initialization;
- optionally perform one or two gradient refinement steps at deployment.

This becomes amortized weight-space meta-learning with optional TTT refinement, still consistent with adapting policy weights from demonstrations.

#### Long context gives no benefit

The selected task may not require memory. Verify partial observability and history dependence before changing the architecture.

### Paper-positioning decision

The preferred final positioning is:

> **Learning What to Write for In-Context Robot Imitation:** a small fast-weight module is meta-trained through a query imitation objective, writes task evidence from demonstrations using a reconstruction-style inner loss, and uses an object-centric space-time encoder for 3D support trajectories.

Possible contributions:

1. controlled evidence of genuine support-specific gradient adaptation without a feed-forward bypass;
2. analysis of WRITE objectives for robot imitation;
3. a space-time supernode-to-fast-weight interface for 3D demonstrations;
4. small-data adaptation to held-out task variations;
5. optional constant-size compression of long demonstrations.

Avoid positioning the paper as merely “MAML works on RLBench” or as a generic long-context-memory paper. The scientific object is **gradient-written adaptation from demonstrations**.

---

# 6. Repository work map

Codex should inspect the current conventions and select names accordingly. The following map describes responsibilities, not mandatory filenames.

## Models

Add or isolate:

- fast-weight linear/MLP memory;
- meta-learned fast-state initializer;
- key/value/query projections;
- residual READ integration;
- adaptation-only policy wrapper/mode;
- optional future-effect head;
- optional supernode-to-register pooling.

Avoid placing all logic in `direct_regression_policy.py` if this makes the legacy policy unreadable. Reuse common encoders and action heads.

## Training

Add a TTT-specific step that supports:

- support sequential scan;
- nested differentiation;
- support outer-loss masking;
- query outer loss;
- exact full MAML;
- explicit FOMAML ablation;
- fast-gradient and slow-gradient clipping separately;
- single-device debug mode;
- later `pmap`/multi-device execution;
- optional truncated BPTT;
- complete metrics.

## Data

Represent a meta-batch explicitly as:

- task dimension;
- support-demo dimension;
- support-time or support-segment dimension;
- query-demo dimension;
- query-time dimension;
- masks for valid steps, support WRITE, support outer loss, and query outer loss;
- task latent only in metadata.

Avoid overloading legacy tensors if their shape semantics obscure support/query separation.

## Configs

Create configs that encode the scientific experiment clearly, for example conceptually:

- state MetaWorld KVB full-MAML;
- state MetaWorld action-BC WRITE;
- state MetaWorld FOMAML;
- RLBench space-time-supernode KVB;
- long-context truncated-TTT;
- direct-conditioning baseline;
- hybrid model.

Every config should explicitly state:

- direct support paths enabled/disabled;
- fast-model type and size;
- WRITE objective;
- READ objective;
- number and granularity of WRITE updates;
- full versus first order;
- fast reset/carry policy;
- action representation;
- support/query episode counts;
- normalizer ID;
- evaluation support conditions.

## Evaluation

Provide separate evaluators for:

- offline support/query loss diagnostics;
- MetaWorld closed-loop adaptation;
- RLBench closed-loop adaptation;
- support-control sweeps;
- context-length sweeps;
- checkpoint comparison under identical support/query episodes.

Evaluation must adapt a checkpoint at runtime. It must not accidentally reuse training-time support embeddings or cached fast weights across tasks.

## Tests

At minimum, add tests for:

- fast-state reset;
- fast-state carry from support to query;
- no direct support bypass;
- support outer-loss masking;
- query action not included in WRITE/READ inputs;
- full second-order gradient to each WRITE projection;
- FOMAML gradient-path removal;
- deterministic single-step update;
- checkpoint save/restore of slow parameters and \(W_0\), but not transient task-specific \(W_t\);
- action conversion and rotation geometry;
- sampler episode separation;
- normalizer split integrity;
- one-device versus distributed numerical agreement within tolerance.

---

# 7. Required logging and analysis

The W&B or equivalent schema should include the following.

## Fast-state metrics

- \(\|W_0\|\);
- \(\|W_t-W_0\|\) after every update;
- update norm per fast tensor;
- gradient norm per fast tensor;
- effective step size;
- singular values or effective rank periodically;
- gate value per insertion layer;
- fast-state drift over support demonstrations and query rollout.

## Objective metrics

- WRITE loss per segment and update;
- query loss before adaptation;
- query loss after each adaptation step;
- separate translation, rotation, and gripper losses;
- future-effect or trajectory reconstruction terms when enabled;
- optional gradient cosine between WRITE and oracle query gradients;
- support overfit versus query generalization curves.

## Support-specificity metrics

- correct/no/wrong/shuffled query loss;
- correct/no/wrong/shuffled closed-loop success;
- distance between fast states written by different tasks;
- within-task fast-state consistency across demonstrations;
- task-latent probe accuracy, clearly labeled as analysis only;
- sensitivity to support order, count, and corruption.

## Data metrics

- task-latent distribution;
- support/query initial-state statistics;
- phase distribution;
- action-component statistics;
- point-cloud valid counts;
- supernode occupancy and assignment entropy;
- number of distinct/occupied supernodes;
- sequence and mask lengths.

## System metrics

- compilation time;
- step time;
- peak memory;
- full-second-order overhead;
- context-length scaling;
- inference latency with and without adaptation.

---

# 8. Recommended priority order

This project should avoid a broad parallel search. The recommended order is:

1. stabilize repository and preserve baselines;
2. implement support-path switches and adaptation-only mode;
3. build the controlled hidden-latent MetaWorld dataset;
4. pass ordinary adaptation Gate 1;
5. implement tiny full-second-order fast-weight TTT;
6. pass one-meta-batch Gate 2;
7. pass held-out latent Gate 3 with KVB;
8. compare FOMAML and support-BC WRITE;
9. add one future-effect objective;
10. add sparse trajectory reconstruction as an ablation;
11. correct action objectives and integrate RLBench;
12. connect space-time supernodes to sequential TTT;
13. only then test long demonstrations and truncated BPTT;
14. introduce the hybrid direct-context + TTT model only after the adaptation-only evidence is secure.

Do not spend substantial compute on:

- full-policy MAML;
- broad ML10/ML45 or arbitrary held-out RLBench families before ML1-like adaptation works;
- diffusion WRITE objectives;
- dense RGB/point-cloud reconstruction;
- large hyperparameter sweeps before the diagnostic gates;
- long-context tasks that do not require history;
- direct support-conditioning models as evidence of TTT.

---

# 9. Expected Codex deliverables

Codex should finish with:

1. a short repository audit identifying the exact old support paths and MAML semantics;
2. the new adaptation-only fast-weight path;
3. a controlled MetaWorld data/evaluation path;
4. full-second-order and FOMAML modes;
5. KVB WRITE and action-BC WRITE;
6. diagnostic tests and instrumentation;
7. configs for each gate and main ablation;
8. a closed-loop support-control evaluator;
9. RLBench/supernode integration after the state gates pass;
10. an `IMPLEMENTATION_SUMMARY.md` documenting:
    - files changed;
    - architecture decisions;
    - exact gradient semantics;
    - config names;
    - commands;
    - tests run;
    - known issues;
    - deviations from this plan and reasons;
    - recommended first experiments.

Codex should retain agency over low-level implementation details, especially JAX/Flax module organization and memory-efficient higher-order differentiation. It should not change the scientific semantics silently.

---

# 10. References for Codex

## Most directly relevant recent work

1. **RoboTTT: Context Scaling for Robot Policies**  
   [arXiv:2607.15275](https://arxiv.org/abs/2607.15275)  
   Read for: small neural fast weights, learned key/value reconstruction, register tokens, residual gating, full gradients through WRITE, sequential support/video adaptation, and truncated BPTT.

2. **DAMI: Dynamics-Aware Meta-Imitation for Generalization to Unseen Robotic Manipulation**  
   [arXiv:2607.15880](https://arxiv.org/abs/2607.15880)  
   Read for: complete visual-motor demonstration encoding, low-level task modulation, full MAML in robotics, and the distinction between one-step ICIL and extensive few-shot fine-tuning.

3. **WIZARD: Robotic Policy Adaptation via Weight-Space Meta-Learning**  
   [arXiv:2606.07217](https://arxiv.org/abs/2606.07217)  
   Read for: restricted LoRA adaptation space, predicting weight updates from task evidence, scale-aware parameter losses, and the amortized-weight fallback.

4. **Learning to (Learn at Test Time): RNNs with Expressive Hidden States**  
   [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)  
   Read for: TTT layers, fast neural states, self-supervised inner objectives, and sequence-model interpretation.

5. **End-to-End Test-Time Training for Long Context**  
   [arXiv:2512.23675](https://arxiv.org/abs/2512.23675)  
   Read for: meta-learning an initialization through long inner optimization and using an aligned next-token objective.

6. **GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent**  
   [arXiv:2603.13875](https://arxiv.org/abs/2603.13875)  
   Read for: explicit reconstruction-based WRITE into a small memory and separation of context writing from downstream reading.

7. **In-Place Test-Time Training**  
   [arXiv:2604.06169](https://arxiv.org/abs/2604.06169)  
   Read for: why arbitrary reconstruction can be misaligned and why the fast objective should be compatible with downstream prediction.

## Meta-learning foundations and stability

8. **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks**  
   [PMLR](https://proceedings.mlr.press/v70/finn17a.html) · [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)  
   Read for: exact bilevel objective and full second-order semantics.

9. **On First-Order Meta-Learning Algorithms**  
   [arXiv:1803.02999](https://arxiv.org/abs/1803.02999)  
   Read for: FOMAML/Reptile approximations and what is discarded.

10. **CAVIA: Fast Context Adaptation via Meta-Learning**  
    [arXiv:1810.10465](https://arxiv.org/abs/1810.10465)  
    Read for: adapting a small context state instead of a large policy and reducing inner-loop sensitivity.

11. **Meta-SGD: Learning to Learn Quickly for Few-Shot Learning**  
    [arXiv:1707.09835](https://arxiv.org/abs/1707.09835)  
    Read for: learned update rates/directions rather than one scalar inner rate.

12. **How to Train Your MAML (MAML++)**  
    [arXiv:1810.09502](https://arxiv.org/abs/1810.09502)  
    Read for: stabilization, multi-step loss, normalization, and architecture sensitivity.

13. **Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML**  
    [arXiv:1909.09157](https://arxiv.org/abs/1909.09157)  
    Read for: the risk that apparent MAML performance comes mainly from slow feature learning rather than meaningful inner adaptation.

14. **One-Shot Visual Imitation Learning via Meta-Learning**  
    [arXiv:1709.04905](https://arxiv.org/abs/1709.04905)  
    Read for: early meta-imitation task construction and support/query separation.

15. **One-Shot Imitation Learning**  
    [arXiv:1703.07326](https://arxiv.org/abs/1703.07326)  
    Read for: demonstration-conditioned robotic imitation and task-distribution assumptions.

## In-context imitation learning

16. **In-Context Robot Transformer (ICRT)**  
    [arXiv:2408.15980](https://arxiv.org/abs/2408.15980)  
    Read for: causal trajectory modeling and feed-forward in-context robot imitation without test-time optimization.

17. **Instant Policy: In-Context Imitation Learning via Graph Diffusion**  
    [arXiv:2411.12633](https://arxiv.org/abs/2411.12633)  
    Read for: graph-structured ICIL, pseudo-demonstration scale, and the distinction between direct context inference and TTT.

18. **Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics**  
    [arXiv:2403.19578](https://arxiv.org/abs/2403.19578)  
    Read for: action/keypoint representation and temporal abstraction.

19. **Action Tokenizer Matters in In-Context Imitation Learning**  
    [arXiv:2503.01206](https://arxiv.org/abs/2503.01206)  
    Read for: temporal smoothness and the effect of action tokenization on ICIL.

## Long-history robot policies

20. **Learning Long-Context Diffusion Policies via Past-Token Prediction**  
    [arXiv:2505.09561](https://arxiv.org/abs/2505.09561)  
    Read for: auxiliary temporal objectives, cached long-context embeddings, and history-consistency verification.

21. **Gated Memory Policy**  
    [arXiv:2604.18933](https://arxiv.org/abs/2604.18933)  
    Read for: deciding when and what to remember, history corruption, and why longer context can hurt Markovian tasks.

22. **BPP: Behavior-Preserving Keyframe Selection for Long-History Robot Policies**  
    [arXiv:2602.15010](https://arxiv.org/abs/2602.15010)  
    Read for: selective history, keyframe extraction, and spurious-correlation problems in long contexts.

23. **HALO / long-horizon memory retrieval for robot policies**  
    [arXiv:2606.25136](https://arxiv.org/abs/2606.25136)  
    Read for: sparse retrieval and auxiliary memory-dependent supervision.

## Action representation and policy heads

24. **On the Continuity of Rotation Representations in Neural Networks**  
    [arXiv:1812.07035](https://arxiv.org/abs/1812.07035)  
    Read for: continuous 6D rotation representations and quaternion discontinuity issues.

25. **3D Diffuser Actor**  
    [arXiv:2402.10885](https://arxiv.org/abs/2402.10885)  
    Read for: RLBench-scale 3D policy design and 6D rotation outputs.

26. **PerAct**  
    [arXiv:2209.05451](https://arxiv.org/abs/2209.05451)  
    Read for: RLBench action discretization and component-specific objectives.

27. **ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**  
    [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)  
    Read for: action chunking and joint-space imitation objectives.

## Benchmarks and repository

28. **MetaWorld / MetaWorld+**  
    [Official repository](https://github.com/Farama-Foundation/Metaworld)  
    Read for: current ML1/ML10/ML45 semantics, partially observable meta-learning environments, and custom benchmark construction.

29. **RLBench: The Robot Learning Benchmark & Learning Environment**  
    [arXiv:1909.12271](https://arxiv.org/abs/1909.12271)  
    Read for: benchmark tasks, observations, demonstrations, and evaluation conventions.

30. **Current project repository**  
    [`Ricvalp/icil-jax-rlbench`](https://github.com/Ricvalp/icil-jax-rlbench)  
    Inspect the current implementation rather than assuming this plan’s description is exhaustive.

---

# 11. Final instruction to Codex

Implement this as a sequence of falsifiable mechanism tests, not as one large architecture rewrite. Preserve the existing feed-forward ICIL and legacy MAML paths. The first objective is not to maximize RLBench success; it is to prove, in the smallest valid setting, that a support demonstration produces a task-specific gradient-written state that improves a different query rollout.

Once that is established, transfer the identical fast-weight mechanism to the space-time-supernode representation. Only after the visual result works should long demonstrations, truncated BPTT, autoregressive variants, or hybrid direct-context models be added.
