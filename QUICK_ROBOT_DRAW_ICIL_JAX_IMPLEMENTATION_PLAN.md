# Quick, Robot, Draw! diversity experiments inside ICIL/JAX

**Active correction (8 September 2026):** Read
[the neighborhood addendum](QUICK_ROBOT_DRAW_ICIL_JAX_NEIGHBORHOOD_CORRECTION.md)
with this plan. Its A-NN/A-local task definitions, permitted offline classifier
curation, splits, diversity axes, controls, and acceptance criteria supersede
the conflicting category-only instructions below. The working branch is
`quick-robot-draw`, as explicitly selected by the user after merging main.

## Codex implementation brief for `Ricvalp/icil:fast-weight-ttt`

**Prepared:** 7 September 2026.  
**Destination repository:** https://github.com/Ricvalp/icil/tree/fast-weight-ttt  
**Destination source revision reviewed:** `87214e5160152cd85fe8a366b2b43bc391e60f12`.  
**Reference/data repository:** https://github.com/Ricvalp/quick-robot-draw/tree/refactor-dataset  
**Reference source revision previously reviewed:** `e76b1e6794df7205785845d3475eb015fdb6f544`.  
**Scope:** A: unseen-category sketch generation; B: instance-specific motor-program imitation; C: optional operator learning; shared evaluation, diversity controls, diagnostics and analysis.  
**Framework:** JAX for every trainable policy, WRITE update, meta-gradient, optimizer and policy-generation path. The existing frozen PyTorch sketch evaluator may run separately, offline.

**This document supersedes the previous QuickDraw implementation plans.** It is self-contained. Do not combine its JAX integration instructions with the previous instruction to implement the experiments inside the PyTorch QuickDraw repository.

**Review status:** The destination branch metadata, `AGENTS.md`, implementation summary, project configuration, and the model/training/analysis source ranges listed in Section 16 were read. The prior source-reviewed QuickDraw plan and its evaluation findings were also consulted. This is a source-level integration review, not a claim that training, checkpoints, data conversion, FID, or the test suite were executed here. Actual assets and local uncommitted changes remain for Codex to inspect.

---

## 0. Mandate, autonomy and priorities

Implement QuickDraw as a new benchmark within the existing ICIL project. Reuse its proven functional fast-weight machinery instead of building another MAML implementation. The user's motivation is to test whether limited training-task diversity encourages family recognition rather than transferable, control-relevant adaptation. A diversity threshold is a hypothesis, not a required result. Smooth gains, architecture-dependent transitions, and no detected threshold are valid findings.

Codex has the stronger local understanding of the repository. This document fixes experimental meaning, information boundaries, evaluation, and acceptance criteria. Codex should choose the smallest coherent implementation: file placement, interfaces, configuration fields, model widths, storage backend, and scheduling should follow the actual checkout. Do not create a parallel framework or force exact signatures merely to match this brief.

### Non-negotiable decisions

1. Keep policy learning and test-time adaptation entirely in JAX. Preserve the current package namespace, repository conventions, and environment unless a necessary dependency is justified.
2. Reuse the existing KVB writer, fast initialization, learned update rates, full-second-order path, delta READ, state-reset semantics, and diagnostics where appropriate.
3. Keep QuickDraw's trained sketch feature extractor, rendering convention, and FID-style evaluation. Do not silently replace them with ImageNet Inception, random features, or a newly trained evaluator per experiment.
4. Separate generation quality, response to support, and instance-specific imitation. No single pooled FID establishes all three.
5. Do not restore deliberately removed legacy robotics training paths. The new explicit-context sketch baseline is a separately identified scientific comparator, not a change to the adaptation-only robotics policy.
6. Do not initialize strict low-diversity runs from a policy trained on all QuickDraw categories. Evaluation-only pretrained features are a different, explicitly disclosed resource.
7. Implement A first, then B1 and B2, then optional C. Correctness fixtures for B may be used earlier. Large sweeps and C are disabled until explicitly launched.
8. QuickDraw is the user's requested alternative diagnostic to the current MetaWorld study. Do not block it on improving ML45 or start RLBench work. Preserve existing robotics regression tests without repeating the entire robotics research program.

### Expected initial Codex decision note

Before substantial edits, record briefly: components to reuse; minimum task-specific extensions; chosen stochastic output family for A; offline evaluator boundary; data and feature-checkpoint availability; and any deviations from this plan. Resolve low-level choices locally rather than repeatedly asking the user to choose between equivalent implementations.

---

## 1. Workspace and repository responsibilities

Open Codex in the existing **ICIL checkout on `fast-weight-ttt`** and place this file at its root. A suitable sibling layout is:

```text
<existing-research-workspace>/
  icil/                  # primary JAX implementation and experiment artifacts
  phi-mujoco/            # retain existing sibling if already used by ICIL
  quick-robot-draw/      # refactor-dataset; reference source and offline evaluator
```

The destination directory may already have a different name; do not move it simply to match this example. Its Python package remains `icil_jax_rlbench/`. The project currently resolves `phi-mujoco` as an editable sibling. Preserve that arrangement. [I1, I5]

Clone or use a separate worktree of the QuickDraw reference branch as a sibling, not as a nested Git repository inside ICIL. Record both revisions. Neither donor installation nor its training scripts should be imported by the JAX policy at runtime. Raw drawings, neutral converted records, pretrained evaluator checkpoints, and generated artifacts belong in configured data/output locations, not version control.

Inspect local changes before switching branches. If either branch advanced beyond the reviewed revision, read the relevant differences and record the actual base. Never force-reset to the reviewed SHA or overwrite unrelated work. Follow applicable `AGENTS.md` instructions in both repositories.

### Boundary between repositories

**ICIL owns:** experimental manifests; task construction; JAX policy models; adaptation; optimization; generation; B's pen environment; controls; checkpoints; analysis; and orchestration.

**QuickDraw reference provides:** preprocessing and token semantics; examples of the working diffusion formulation; canonical rendering; the trained ResNet evaluator; FID utilities; and legacy reproduction references.

**Offline metric process owns:** loading the unchanged evaluator checkpoint, raster/feature extraction, and score computation from exported outputs. This does not require PyTorch in the JAX training process, a PyTorch-to-JAX policy checkpoint converter, or a cross-framework gradient bridge.

---

## 2. What already exists in `fast-weight-ttt`

The destination is substantially further along than the repository assumed in the earlier plans. The following findings should guide reuse, not trigger wholesale refactoring.

| Existing component | Verified behavior | Implication for QuickDraw |
|---|---|---|
| `models/fast_weight_ttt.py` | Functional linear/MLP fast model; learned `fast_init`; positive per-tensor rates; K/V projections; full/first-order adaptation; clipping; explicit fast state. | Extend or factor only the task-independent pieces needed for sketches. Do not implement another optimizer/state model. |
| `adapt_encoded_support` | KVB adaptation accepts pre-encoded `[segments, registers, hidden_dim]` evidence and masks. | A useful bridge for a sketch support encoder; it is not inherently point-cloud-specific. |
| `fast_read_residual` | Both absolute-gated and delta READ exist. Delta READ subtracts the `W0` output. | Prefer the existing delta mode for an exact no-adaptation reference; retain its gradient semantics. |
| State `predict_action` and action loss | Currently expect translation components plus one gripper output, with state-specific heads/metrics. | Do not shoehorn sketch boundaries, pen lifts, STOP, and stochastic outputs into this schema. Add a sketch-specific output contract. |
| `train/ttt_step.py` | Per-task adaptation followed by query-only loss, task `vmap`, nested autodiff, Optax state, JIT/distributed options. | Reuse orchestration, but the query loss and metric dictionary are state-specific and need task-appropriate extension. |
| Existing runner/checkpoint conventions | Provenance, normalization, RNG/resume, and exclusion of transient task states are documented. | Extend the established format with sketch manifests and stochastic-stream provenance. |
| `analysis/metaworld_update_information.py` | Raw support, first gradients, final fast deltas, functional changes and probes. | Reuse mathematical ideas/helpers, not imports that drag `phi_mujoco` task contracts into QuickDraw. |

Sources: [I1–I8]. The source review did not establish runtime performance of these components. Preserve the existing tests and add tests for the new task contracts.

### Important implementation consequences

**The current READ head is not a sketch distribution model.** Experiment A cannot be evaluated responsibly by adding a deterministic two-coordinate Huber head and comparing its averaged sketches with the original generative policy. A requires a proper stochastic generator; B1/B2 can also use deterministic control heads.

**The existing training RNG was not needed for a deterministic state loss.** In the inspected step, a key is advanced but the loss does not consume stochastic inputs. A new diffusion or probabilistic sketch objective must explicitly receive correctly split keys. Merely inheriting a `rng` field does not make noise, dropout or sampling reproducible. [I4]

**Existing tensor effective rank is not dataset update diversity.** The current fast-state rank helper measures spectra of matrices inside one fast model. The diversity study additionally needs covariance across tasks' updates or functional responses. Label these separately. [I3]

**No broad cleanup is requested.** A small task-specific training adapter or a carefully factored shared objective is acceptable. Do not generalize every robotics module before the first sketch experiment.

---

## 3. Preserve and isolate the existing sketch evaluation

### 3.1 Existing assets and their meaning

The reference branch contains a one-channel ResNet18 classifier trained with 345 outputs. `ResNet18FeatureExtractor` loads that checkpoint and replaces its final classifier by an identity, exposing pooled features. The configured checkpoint is `metrics/checkpoints/resnet18_step40000.pt`. Confirm its actual presence, hash, input preprocessing, feature dimension and training provenance locally. The constructor and training code alone do not prove which drawings a particular checkpoint saw. [D1]

Its existing Fréchet calculation uses feature means and covariances, not standard Inception features. Preserve this metric and call it **Sketch-FD / sketch-feature FID**, with the exact checkpoint and preprocessing named. Do not apply the retrieval pipeline's L2 feature normalization to FID unless introducing a separately named metric. [D1, D4]

The metric configuration uses 64×64 grayscale images, antialiasing factor 2, line width 2.0, background 0, stroke 1, and `normalize_inputs=False`. The renderer maps absolute coordinates from a fixed normalized canvas and uses the destination point's pen flag for incoming segments. The generic dataclass has a different default line width; load the recorded evaluation configuration rather than silently taking defaults. [D2]

Both legacy diffusion evaluators already compute generated-versus-real FD and real-versus-real reference FD. Preserve a small reproducibility fixture for that functionality; do not advertise real-real calibration as an entirely new feature. [D3]

### 3.2 Cross-framework evaluation contract

JAX generation exports actual trajectories and metadata, or canonical rasters plus the original trajectories. The offline evaluator reads these artifacts, computes features in batches with the existing frozen ResNet, and returns per-sample features and aggregate scores. No evaluator tensors or feature gradients enter meta-training.

Use a neutral numerical format with explicit lengths, masks, schema version, coordinate mode, pen semantics, intended task/category, support IDs, query seed, condition, checkpoint identifier, and rendering configuration. Codex should choose the format compatible with existing data/artifact conventions.

Keep the metric environment separate if necessary. Avoid concurrent JAX/PyTorch GPU allocation surprises; sequential GPU use or CPU feature extraction is acceptable. Do not alter the JAX/CUDA lockfile to accommodate a legacy metric import. A JAX port of the evaluator is optional future work only after numerical parity, not a prerequisite.

**Parity gate:** pass identical saved sketches through the donor evaluator and the integration boundary. Compare rasters exactly where possible, features numerically, and FD within declared tolerances. Round-tripping data must not change pen lifts, STOP trimming, coordinate convention or feature normalization.

A missing original checkpoint is an asset limitation to report, not permission to substitute random features. Data/model smoke tests can proceed, but scores from a substitute evaluator must be clearly separate from the legacy-compatible metric.

### 3.3 Quality is not conditioning

For a fixed feature extractor `phi`, retain the standard Gaussian feature-distance calculation:

$$
\operatorname{FD}(G,R)=\|\mu_G-\mu_R\|_2^2+
\operatorname{tr}\left(\Sigma_G+\Sigma_R-
2(\Sigma_R^{1/2}\Sigma_G\Sigma_R^{1/2})^{1/2}\right).
$$

Use stable float64 numerical accumulation outside the training graph. Validate symmetry, identical-set behavior, finite covariances and numerical residuals; do not silently discard arbitrarily large complex components or conceal negative numerical results. Any numerical change to the donor routine needs a parity check and versioned score convention. [R4–R6]

Pooled FD ignores which prompt generated which output: permuting outputs between cat and dog episodes leaves the image collection, hence pooled FD, unchanged. Therefore report three distinct levels:

- **Marginal quality:** pooled Sketch-FD, real-real reference, invalid/blank/no-STOP statistics.
- **Conditional fidelity:** category-conditioned feature distances and intended-category recognition where supported by an independent frozen classifier.
- **Instance/control fidelity:** paired geometry, order, frame accuracy and executed trajectory error for B/C.

For category-conditional distances with limited samples, implement a macro-averaged unbiased kernel MMD estimator in sketch feature space, termed **Sketch-KID/MMD**, with one fixed kernel configuration. Unbiased estimates can be negative; do not clip them to zero. Per-category FD remains useful when sample counts support covariance estimation, but must report sample counts and instability. [R5]

Use fixed, independently sampled real reference manifests balanced over intended categories. The legacy evaluators take the first prompt from additional episodes as real references, which can be retrieval-selected. Preserve that as a named legacy protocol, not the primary new reference population. [D3]

### 3.4 Reference and metric governance

Freeze the renderer, extractor, feature normalization, reference IDs, category weights, kernel settings and sample budgets across models, diversity levels and support controls. Generate exactly the requested count; do not round up to batches and silently report the nominal number. Fix iterator exhaustion rather than changing the evaluation population.

Real-real comparisons should use disjoint IDs with matched category weights and report repeated-reference variability. Their score is a calibration, not a correction that removes finite-sample FD bias. Small-sample ranking and preprocessing effects need explicit qualification. [R6, R7]

A classifier trained on all categories can be an explicitly disclosed **evaluation-only** tool. It must not supply policy features, prompt retrieval, training targets or checkpoint selection on the final test set. Do not equate this evaluation privilege with policy training on those categories, but do disclose it. Unknown training-ID overlap needs an independent-reference sensitivity analysis before publication.

For B/C, do not recenter, rescale or rotate generated drawings before scoring requested-frame accuracy. Since the renderer clips outside its canvas, record out-of-bounds movement before rendering. Report shape-invariant visualization metrics only alongside, not instead of, unaligned errors.

---

## 4. Data contract and prevention of hidden task information

### 4.1 Canonical records, not imported training loaders

Reuse the reference repository's preprocessing semantics and existing processed assets where possible. Expose immutable records in ICIL containing base drawing ID, category, coordinate sequence, pen/stroke boundaries, valid length and provenance. A one-time export from LMDB/WebDataset to a neutral cache is acceptable; so is a lightweight native reader. Do not import the donor's PyTorch collators or globally named `dataset` package into the policy process.

Keep absolute and delta coordinates distinct. Define delta origin and stroke-boundary behavior, and test their round trip. Every geometric transform must update coordinates and deltas consistently. Derive transitions only within a drawing; never fabricate a motion transition from the end of one support to the start of another.

The donor has seven-channel episode tokens, while its encoder-decoder removes the RESET channel to produce six-channel tokens with a different STOP index. That is a compatibility fact, not a required JAX tensor layout. Keep the semantic schema explicit instead of copying magic indices. The incoming-segment pen convention is more reliable than ambiguous old comments such as “start.” [D5]

Metadata IDs are for sampling, logging and analysis, not policy features. No category/family label, base sketch identifier, complete query embedding or hidden operator parameters may enter the main model.

### 4.2 Sampling policy

For primary A, draw K distinct supports and independent query drawings from the same category, with no overlap of IDs or duplicate clusters within an episode. Sample category first, then drawings. Do not choose supports by similarity to the completed query.

The donor's `EpisodeBuilderSimilar` deliberately looks up the full query's embedding and retrieves neighbors excluding its exact ID. That is an **oracle-retrieval conditional task**, not independent demonstration conditioning. Keep it optional and separately labeled for legacy reproduction. It must not define the primary diversity experiment. [D4]

The reference loader loads FAISS even for the random builder and can fall back to all families when a split map is absent. Avoid inheriting either behavior. The new primary sampler needs no learned feature index and must fail loudly on missing or inconsistent split specifications. The source also contains an embedding-constructor keyword typo; fix it only if using that optional donor tool, not as a prerequisite to JAX training. [D4, D5]

### 4.3 Splits and eligibility

Create immutable category and base-ID manifests before selecting models. A holds out entire categories; B holds out entire base programs/duplicate clusters; C has its own source-program/operator splits. Record train, development and untouched test sets explicitly. Do not infer split correctness from directory names.

Define drawing eligibility once using raw length, stroke validity and normalization rules. Report category-wise retention and length distributions. Rejection sampling for short combined episodes can bias the accepted family distribution; either precompute eligible reservoirs or monitor/compensate acceptance. Do not change inclusion criteria with F or give TTT longer drawings than the explicit-context baseline.

For A, use the same finite drawing reservoir definitions for all competing architectures. Family selection should use training-pool descriptive statistics only, not performance on test categories. Detect exact duplicates and obvious near-duplicate clusters with a declared data-curation rule. Do not silently use all-category learned embeddings as training-time features under a “from scratch, low diversity” claim.

### 4.4 Public information and masks

Preserve task/support-demo/support-time/query-demo/query-time axes at the experiment boundary. Codex may reshape inside pure functions, but per-task updates cannot be averaged across tasks before adapting.

Support lengths and phase can be known because complete demonstrations are supplied. Query future length, future stroke count, completed target embedding, private time warp, and target-normalized `t/T_true` are not public inputs. Loss masks can depend on labels; model-visible inference masks cannot expose future target length. Prefer elapsed step and a fixed public maximum budget, or an explicitly requested public schedule in B.

For probabilistic generation, teacher-forced query prefixes are legitimate training inputs. They must be causally shifted and replaced by generated history during free rollout. Empty-query generation, prefix completion, and teacher-forced diagnostics are distinct protocols and need distinct names.

---

## 5. Shared JAX model and adaptation requirements

### 5.1 Reuse the current writer

Let `theta` denote slow parameters, including learned `W0`, projections, encoder and output head. Let `W` denote transient fast-model parameters. For encoded support event `h_i`:

$$
K_i=k_\theta(h_i),\qquad V_i=v_\theta(h_i),
$$

$$
L_{\mathrm{write},j}(W)=
\frac{\sum_{i\in j}m_i\|f_W(K_i)-V_i\|_2^2}
{d_v\max(1,\sum_{i\in j}m_i)}.
$$

Apply the repository's learned-rate update to W only, with full meta-gradients through keys, values, encoders and initialization. Reuse its clipping/rate semantics and K/V normalization rather than changing them during integration. [I2, I3]

The support encoder should see actual ordered sketch evidence: coordinates/movement, pen state, relevant transitions and public position within a support drawing. Segment-local sequence encoding is appropriate. Avoid starting with a global bag of drawing statistics that makes stroke order invisible. No raster ResNet features are needed in the policy.

Use the existing encoded-support bridge if it is the cleanest path. Otherwise separate generic fast adaptation from the state-specific evidence encoder with a minimal tested change. Boundary markers may provide context; whether they themselves generate a WRITE loss must be explicit and consistent.

### 5.2 Delta READ

Preserve the current delta-READ principle:

$$
\Delta h(q;W)=P_\theta f_W(q)-P_\theta f_{W_0}(q),
\qquad h'=h+\beta\Delta h.
$$

At `W=W0`, this is exactly zero. It avoids requiring a tiny learned gate to first enable the adaptation gradient. Reuse the branch implementation where possible. Do not detach the adapted path or accidentally change gradients through the reference path. [I2]

For the TTT-only model, support information reaches query predictions only through W. The read query is produced from permitted query-side information, not a support summary or cached demonstration-attention output. Freeze post-support W during all independent query generations in the primary protocol.

### 5.3 Outer objectives and stochastic generation

A's main output head must represent a conditional distribution over sketches. The recommended continuity choice is a compact JAX diffusion/chunk decoder reproducing the donor's task-level noise-prediction formulation, not its framework or trained weights. Codex may instead choose a JAX autoregressive mixture-density/event model if that is substantially simpler to integrate and validate. Record this choice before comparative runs; implement the **same output family for explicit-context and TTT variants**.

For diffusion:

$$
z_\nu=\sqrt{\bar\alpha_\nu}\,y+\sqrt{1-\bar\alpha_\nu}\,\epsilon,
\qquad
L_{\mathrm{outer}}=\mathbb E_{\nu,\epsilon}
[\|\epsilon_\theta(z_\nu,\nu,\text{query history};W_S)-\epsilon\|^2]_{\mathrm{valid}}.
$$

This is denoising MSE, not exact likelihood. Fix noise/timesteps for paired diagnostic comparisons. The diffusion denoiser's noisy candidate is a legitimate generative input, not permission to feed the clean query future to support WRITE. A conservative integration derives fast READ queries from causal query history and supplies the noisy candidate separately to the decoder.

For an autoregressive alternative, use a valid probabilistic coordinate head and explicit pen/STOP probabilities, with stable NLL. A deterministic point predictor is not an adequate replacement. Apply causal masking; no bidirectional access to the target suffix. [R8]

For B1/B2, deterministic point/control regression plus pen/STOP prediction is acceptable when targets are determined by the supplied program and public frame. Compare matched heads within B. Do not directly compare B's regression loss with A's diffusion MSE or NLL.

### 5.4 Do not conflate sketch and robot action semantics

The current `FastWeightTTTConfig` assumes `action_dim = translation_dim + 1`. Sketch tasks need planar movement, pen transitions and termination, possibly more distribution parameters. Do not satisfy the old assertion by treating STOP as a fake coordinate or losing its supervision. [I2]

Codex should add the smallest sketch-specific head/loss contract. If structural channels are diffused to preserve donor behavior, apply exactly the same scheme to all matched models. A hybrid continuous-coordinate/discrete-event head is also acceptable if implemented across comparators and recorded as a different model version.

Padding after STOP is not many equally weighted terminal demonstrations. Mask coordinates and events deliberately; retain explicit STOP supervision once per sequence, report no-STOP rates, and truncate at the declared generation limit. Generation must not use the true query length to stop.

### 5.5 Full second order and first-order ablation

Optimize only post-adaptation query loss in the main meta-objective. Support reconstruction supplies inner gradients but is not silently added as an outer auxiliary loss. Reuse nested JAX differentiation rather than custom approximations. [I4, R1, R2]

For a WRITE-only parameter psi, the one-step outer gradient includes:

$$
\nabla_\psi L_Q=-
\left(\frac{\partial^2L_W}{\partial\psi\,\partial W}\right)^\top
\eta^\top\nabla_{W'}L_Q.
$$

Stopping the fast gradient removes this route. In the strict KVB architecture, FOMAML should lose expected key/value/support-encoder meta-gradients. It can retain initialization, READ, rate or other direct paths; do not claim all gradients vanish or that FOMAML cannot work in every formulation. Test the actual graph.

Avoid exact-zero initialization of every multiplicative READ/fast path that would also zero the first useful gradient. Check initialization numerically, including delta READ.

### 5.6 WRITE objective ladder

Start with **existing KVB + full second order**. Add a matched **supervised support WRITE** using the same sketch output objective as READ: support denoising loss for diffusion or support NLL for AR; for B, paired control/trajectory loss. Match update counts and fast-state capacity. This separates auxiliary-loss alignment from dataset diversity.

Only after this comparison, add one optional ordered future-effect or masked-trajectory objective. A suitable target is a future displacement or later stroke segment recoverable from demonstrated evidence. Avoid a same-token reconstruction bypass or a target already supplied as an unmasked input to that head. Entire-support reconstruction can be a memory-compression objective, but cannot automatically be described as predictive inference.

Do not introduce hand-coded semantic motion phases, object relations, or multiple unrelated reconstruction losses into QuickDraw. They are not required to test the diversity hypothesis.

### 5.7 JAX execution and numerical contract

Reuse explicit PyTrees, per-task `vmap`, support-sequence `lax.scan`, JIT, and existing Optax/checkpoint conventions. In particular:

- All carried fast leaves have fixed shape/dtype within a compiled function. Bucket support/query lengths to limit recompilation. [R9]
- Each task starts from the same meta-initialization but has its own adapted state. Never adapt a shared batch state to averaged support gradients.
- All-padding WRITE segments must be exact no-ops, including when drift penalties or optimizer-like state are enabled. Masked reconstruction alone does not ensure this when a separate regularizer remains active.
- Padding values must be finite. Masking a NaN after it is computed is not sufficient.
- Do not create transitions across drawing boundaries or infer a task latent from padding geometry.
- Split RNG streams for data selection, transforms, support order, training diffusion noise, evaluation noise and model initialization. Save their state for resume and pair evaluation keys across controls.
- Encodings defining learned WRITE remain inside the differentiable graph. Caching detached trainable support embeddings would remove the outer signal.
- Test full precision first. Retain suitable precision for fast updates and higher derivatives before experimenting with mixed precision.
- Use rematerialization before truncating meta-gradients. `jax.checkpoint` preserves gradients by recomputation; detaching fast state at a segment boundary changes the learned objective. [R10]
- With a single outer query loss after all support segments, TBPTT detachment prevents early segments receiving that loss's meta-gradient. Do not introduce it silently into the diversity sweep.
- Adapt once per support set; do not re-WRITE on every diffusion denoising step. Generation reads a frozen W.
- Distributed training must preserve task isolation and a declared gradient aggregation/clipping convention. Verify single-device equivalence, including cases that activate clipping.

JAX test-time WRITE still computes a gradient even though slow parameters are not being trained. Do not remove that computation because the model is in evaluation. The PyTorch `no_grad` issue belongs only to the optional metric/legacy process, not the JAX implementation.

---

## 6. Model comparison: isolate the adaptation mechanism

Implement the following named conditions without restoring legacy robot-policy code:

| Model | Test-time support access | Purpose |
|---|---|---|
| `ttt_kvb_full` | Gradient updates to W only | Primary method. |
| `explicit_context` | A separately implemented support encoder/attention path; no gradient update | Feed-forward ICIL comparator on the same sketch data/output family. |
| `no_support` | None; independently trained | Genuine support-free generative baseline. |
| `ttt_supervised_write` | Same W, updated by support prediction objective | WRITE alignment control. |
| `ttt_kvb_first_order` | Same forward updates; designated first-order meta-gradient | Higher-order control. |

The exact names can follow repository conventions. The direct-context comparator needs its own declared model/checkpoint type and cannot be enabled accidentally in the main adapter. This explicitly scoped comparator does not authorize restoring old support-FiLM or direct-regression robot trainers forbidden by `AGENTS.md`. [I1]

Match shared query backbone, output family, normalization, supports, generated token budget, query-loss exposure and sampling settings as closely as possible. Report additional writer versus context-encoder parameters and compute. Where exact parameter matching is impossible, show a capacity sensitivity rather than claiming a perfectly isolated comparison.

Use random initialization or training only on each run's permitted categories for the strict study. Query-only pretraining within that same data pool is allowed as a separate, consistently applied schedule; count its examples and compute. No all-category policy checkpoint, no privileged family one-hot, no oracle latent initialization.

Within every TTT checkpoint, evaluate no update, correct support, wrong support and norm-matched random fast perturbations. A deliberately weak no-update state is not a fair substitute for the separately trained `no_support` baseline.

---

## 7. Experiment A: unseen-category sketch generation

### Task

A task is a category distribution c. K demonstrations are independent drawings from c; one or more independent query drawings supply training targets. At evaluation, generate a new drawing without its category label or completed target. Hold out categories entirely from policy training.

Encoding category identity is legitimate in A. This task measures novel-category conditioning, not instance-specific control learning. Evidence for the stronger family-signature-versus-motor-program distinction comes from B.

No unique target drawing is determined by “several cats.” Use distributional and condition-sensitive scores, not paired Euclidean error to one arbitrary cat.

### Diversity controls

Choose a fixed category train/development/test partition from eligible data. For the training pool, construct nested family subsets such as F in `{4, 8, 16, 32, 64, 128, 256}`, capped by actual availability. Keep final held-out categories identical across F. Repeat subset construction with independent subset seeds, stratified by coarse length/stroke complexity where feasible.

Run three separate regimes:

1. **Fixed unique drawing budget U:** redistribute a fixed training reservoir across F while matching training exposure and counting valid events.
2. **Fixed drawings per category:** U grows with F. Report this as diversity-plus-data scaling.
3. **Fixed F, varying drawings per category:** test whether sample count explains the trend.

Save actual raw IDs. Unique records, repeated draws, geometric augmentations, support tokens, query-loss tokens, WRITE steps and optimizer steps are different counts. Do not use “epoch” as the primary budget when dataset size varies.

Keep episode-shared hidden transformations disabled in the first A sweep. They introduce an additional continuous task axis beyond category count. Representation-preserving augmentation can be added in a separate named experiment with identical distribution across F.

### Evaluation and controls

Use fixed test episode manifests and paired stochastic seeds. Include empty-query free generation as the primary A protocol; prefix completion is a separate test with a declared prefix budget.

Report pooled Sketch-FD, real-real calibration, macro category-conditional Sketch-KID/MMD, intended-category accuracy when supported, validity/length/stroke statistics, and galleries selected before viewing results. Score against the originally intended category even when wrong-category support is supplied; scoring against the replacement category would erase the conditioning failure being tested.

Wrong supports should be approximately length/complexity matched. Add no-update and independently trained no-support controls. Support reordering, stroke reversal and action corruption are diagnostics, not universally task-destroying negatives: reversing a cat may still clearly identify the category.

### Copying and diversity

A support-copy generator can score very well because supports come from the correct distribution. Include it as an explicit baseline. Also include training-reservoir retrieval where feasible, and report nearest-support/training distances in fixed feature space and geometry, duplicate rate, within-episode diversity, and sample galleries. Calibrate near-copy thresholds with exact duplicates and independent real pairs.

Good FD plus category accuracy does not prove novel generation or an abstract learning algorithm. State conclusions accordingly.

---

## 8. Experiment B: instance-specific motor-program imitation

### B1: reproduce a particular demonstrated program in a new observable frame

A task is a base drawing u, including ordered strokes and pen-up transitions, not its category. Different drawings from the same category are different tasks.

Create multiple executions `T_i(u)` using invertible, public 2D frames. Begin with bounded rotations, translations and positive isotropic scales. Every model receives each support frame and the requested query frame `T_q`. The target is `T_q(u)`. The hidden information is the program; an independently randomized unobserved query transform is not a solvable imitation problem.

Start from canonical base records, apply frame transforms afterward, and do not renormalize each transformed execution back to a common box. Verify inverse transforms and frame accuracy before training. Frames and execution-noise profiles must be independent of category and identical across diversity levels.

Use either a declared public resampling schedule or explicit STOP prediction. Do not leak private length/time warps through masks. Add timing variation only with clear semantics: a public requested schedule, or evaluation invariant to timing but still sensitive to geometry/order.

### B diversity axis

Vary number N of unique base programs, not just number of categories. Use nested reservoirs over several orders of magnitude as data permits. Hold out base IDs and duplicate clusters. Report both unseen programs in familiar categories and programs in held-out categories.

A primary N sweep can keep coarse category coverage fixed while increasing within-category programs. Do not silently increase category count and program count together and attribute the result solely to motor-task diversity. A limited crossed `(category count, programs per category)` study is an optional follow-up.

Transformed versions of one program do not count as new base programs. Keep transformation distribution and processed-event budget fixed while varying N.

### B1 baselines and metrics

Before model training, implement:

- an oracle using the hidden canonical program and requested frame;
- canonicalize one supplied execution with `T_i^{-1}`, then transform and replay using `T_q`;
- a category prototype;
- an untransformed replay control.

The transformed replay baseline should be nearly exact in noiseless B1. Failure indicates a construction/metric error. Successful TTT reproduction alone does not beat this simple solution or prove primitive abstraction.

Score unaligned point/trajectory error in the requested frame, endpoint error, pen-state and stroke-order accuracy, termination, and optional geometric-set/DTW measures. DTW/Chamfer cannot replace order/frame errors because they can hide control mistakes. Sketch-FD is supplementary.

**Decisive negative:** replace supports with a different base drawing of the same category, matched for length/stroke complexity. Retain the original target. A family-label strategy should fail this distinction.

### B2: actual closed-loop pen control

After B1 integrity passes, add a lightweight simulator-free pen environment. Public state includes actual XY/pen state and declared execution frame or phase information. The policy issues bounded motion commands and pen control; a specified integrator produces the next state. The evaluated path is the executed path, not merely a predicted sequence.

Start with deterministic dynamics and then add controlled disturbances, delays or execution noise as separate profiles. Query history must contain executed states/actions, not expert suffixes. Keep support-adapted W fixed during the query rollout initially.

Define recovery semantics: rejoin a timed reference, follow ordered geometry, or complete the shape regardless of timing. Provide a feedback path-tracking oracle consistent with that definition. New start locations may require pen-up travel to the first stroke; specify this explicitly rather than letting the oracle teleport.

Compare open-loop transformed replay, closed-loop transformed-replay tracking, category-prototype control, no support, explicit context and TTT. A feedback replay controller is a serious baseline; disturbances do not automatically make neural meta-learning necessary.

Score tracking/shape error, order, frame accuracy, recovery after perturbations, pen-up travel, completion and invalid actions. Do not render away or align away the errors defining control success.

---

## 9. Experiment C: optional operator learning

Disabled by default until A/B infrastructure and evaluation are reliable.

Sample a hidden operator `T_tau`. Support consists of paired source and transformed trajectories `(u_i, T_tau(u_i))`; query receives a new source program `u_q` and must produce `T_tau(u_q)`. Unlike B1, operator parameters are not supplied. Category must not identify the operator.

Start with identifiable 2D affine transformations or a restricted similarity family. Ensure support points have sufficient rank and geometric coverage. An independent least-squares estimator followed by transformed replay is the primary reference; record numerical conditioning and residuals.

Split source programs, operator instances and, where claimed, operator families explicitly. Hold out both new operators within a family and new combinations separately. Vary a finite training-operator dictionary while keeping the source-program distribution fixed; a continuous-operator training regime is a separate experiment. Merely saying “many episodes” does not define operator diversity.

Use paired coordinate/frame errors, recovered-operator error, transfer to novel source programs, and wrong-operator support controls. FID is secondary. Failure to match the linear estimator on a noiseless identifiable task should trigger diagnostics before considering nonlinear operators or larger models.

C is a controlled learning-to-learn diagnostic, not automatically a realistic robotics task. Keep its claims distinct.

---

## 10. Integration-specific gates

These gates reuse existing ICIL correctness tests. They do not ask Codex to redo the completed MetaWorld implementation.

### Gate Q0: source, asset and dependency audit

Confirm both revisions, actual data schema, immutable IDs, evaluator checkpoint availability, existing relevant tests and chosen A output model. Record what is reused versus new. Confirm QuickDraw training does not require simulator startup, FAISS indices or PyTorch policy imports.

### Gate Q1: representation and metric parity

Run coordinate/delta/pen/STOP fixtures, frame inversion, token-to-raster parity and identical feature/FD examples. Include blanks, one-point strokes, lifted travel, maximum length, missing STOP, out-of-bounds outputs, and legitimate padded examples.

A test that permutes samples between categories must leave pooled FD unchanged while degrading the designated condition-sensitive score on a known fixture. B's wrong-frame and wrong-program fixtures must be rejected by B's metrics even when raster appearance is similar.

### Gate Q2: causal data and split integrity

Assert no forbidden category/base-ID overlap, no shared support/query drawing, independent A supports, no complete-query retrieval, and no query-future leakage through tokens, lengths, masks, normalized phase or evaluation seeds.

Replace hidden query labels/suffixes while keeping public inputs fixed: support-adapted W and free-generation outputs with a fixed key must not change. Loss may change; generated output must not. For diffusion training, test this at the generation API rather than demanding invariance to the deliberately constructed noisy training candidate.

### Gate Q3: JAX adaptation and regression tests

Run the destination's existing relevant tests, then extend them with the actual sketch encoder/head:

- full versus first-order expected gradient paths;
- finite differences of outer loss through W updates for representative parameters, including support encoder, K/V, W0, rates and READ;
- fixed diagnostic noise/keys for every finite-difference evaluation;
- single-task loop versus `vmap`, eager versus JIT, and supported distributed equivalence;
- exact task resets, reuse of post-support W, no cross-condition contamination;
- all-padding no-op including nonzero drift regularization;
- delta READ exactly zero at W0 but nonzero response/derivative under a perturbation;
- save/resume equivalence of parameters, optimizer, sampling streams and training noise.

Check numerical derivatives in a tiny, smooth, high-precision fixture away from clipping thresholds. Do not require every parameter's gradient to be nonzero at arbitrary symmetric inputs. Instead test the intended pathways on nondegenerate fixtures. [R9, R11]

### Gate Q4: adaptation feasibility and tiny-data fit

Establish task feasibility with B replay/oracles and a small support-conditioned generative fixture for A. Ordinary support prediction-loss adaptation **within the same fast subspace** tests whether W can affect the needed output. Full slow-model fine-tuning can be an optimistic capacity control but is not the same upper bound.

Do not treat fitting KVB with random/untrained projections as a necessary precondition: the projections are specifically learned by the outer objective. Conversely, a falling reconstruction loss alone does not demonstrate useful adaptation.

Overfit a fixed meta-batch with the full new computation graph. For diffusion, use a fixed small set of diagnostic noises/timesteps and also check fresh diagnostic noise to distinguish a broken loss from noise memorization. This is an implementation check, not a held-out adaptation result.

### Gate Q5: matched A pilot

Run a small-F and a substantially larger-F condition for `ttt_kvb_full` and `explicit_context`, with the same output family and data budget. Add a genuine no-support baseline. Use full generation and the shared metrics, not only training losses.

Passing this gate means the experiments and controls execute correctly. Positive support gain is a research outcome, not a software acceptance criterion. A negative pilot triggers the declared diagnosis, not selective resampling until the result becomes positive.

### Gate Q6: B instance/control evaluation

Verify B1 transformed replay and same-category wrong-instance sensitivity, then B2 feedback tracking. Compare neural models only after these construction checks pass. C uses analogous estimator/oracle checks.

---

## 11. Analysis of the diversity hypothesis

### 11.1 Separate three claims

1. Increasing diversity improves held-out performance.
2. There is an identifiable empirical change point rather than a smooth trend.
3. TTT reaches a predeclared competence/support-specificity criterion at lower diversity than matched explicit-context ICIL.

Claim 1 does not imply 2; neither implies 3. Existing regression work motivates the study but does not prove a sketch, robotics or TTT threshold. [R3]

For a lower-is-better condition-sensitive score L, report:

$$
G(F)=L_{\mathrm{no\ update}}(F)-L_{\mathrm{correct}}(F),
\qquad
S(F)=L_{\mathrm{wrong}}(F)-L_{\mathrm{correct}}(F).
$$

Also compare against the independently trained no-support baseline. For B/C substitute task-appropriate errors and the N/operator-diversity axis. Do not define support specificity solely from pooled FD.

### 11.2 Budget and capacity controls

Count and plot performance against actual unique drawings/programs, category count, valid support events, scored query events, optimizer steps, WRITE steps and measured compute. Match what is practical, disclose the remainder, and include limited convergence-controlled runs to distinguish optimization undertraining from diversity.

A global F sweep cannot alter model size, context limits, fast-state size, output sampler or pretraining exposure per point. Tune shared settings on development data with equal opportunity for both model families. Architecture-specific tuning is allowed but must use the same budget and remain fixed during confirmation.

After the first curve is credible, repeat selected diversity levels with at least two additional capacity settings. The expected result is a dependence on task/model/fast-state capacity, not a universal critical number.

### 11.3 Statistical units and change-point analysis

Use independent model seeds and independent nested-category-subset seeds. Pair evaluation tasks and stochastic keys across conditions. Bootstrap at task/category or base-program level and account for model/subset variation; generated samples sharing one support set or one checkpoint are not independent experimental replicates.

Choose practical effect/competence thresholds on development experiments before final tests. Report confidence intervals, entire curves and retained negative results. A minimum F meeting criteria at tested points is an **operational threshold**, not evidence of a physical phase transition.

For a change-point claim, compare a smooth model with a segmented alternative and report uncertainty. Do not pick a threshold on the untouched test set. If uncertainty is broad, the fitted location lies outside tested diversity, or the smooth model suffices, report no identifiable threshold.

### 11.4 Update-space and functional probes

Extend the existing diagnostic concepts to sketches: raw support statistics, first WRITE gradient, actual rate-scaled/clipped update, final fast delta, and changes in READ/output behavior on fixed public probe inputs. For stochastic heads, fix noise/time or evaluate distribution parameters.

Collect across tasks. For vectors `d_tau = vec(W_tau-W0)`, compute centered covariance C and participation ratio:

$$
D_{\mathrm{eff}}=\frac{(\operatorname{tr}C)^2}{\operatorname{tr}(C^2)}.
$$

State the all-zero convention and sample/rank limits. With many parameters, use a fixed projection or Gram formulation; fit PCA/scaling on analysis-training data only. Do not compare absolute weight directions across independently initialized models as though neural parameterizations were aligned. Functional response geometry is a useful companion. [I8]

Probe A category information and shape features; probe B program geometry, stroke order and transformation-correct behavior; probe C operators. Use separate probe splits. Category decodability is not a failure in A. Strong category decoding plus poor same-category instance transfer in B is more relevant to the user's concern.

Report oracle query-gradient alignment and actual first-step query improvement. With an actual update deltaW, `-<grad_W L_Q, deltaW>` is a local improvement predictor. Label its sign explicitly; a raw WRITE-gradient cosine is not equivalent after learned rates and clipping.

Target gradients use query labels only in offline diagnostics. Never feed them to test-time adaptation. Family clustering, low effective rank or weak time-shuffle sensitivity alone do not prove a deficient algorithm: a simple task may genuinely need few dimensions or be order-invariant.

---

## 12. Staged execution and bounded scope

### Stage 1: shared foundations

Complete the source/asset note, neutral data interface, immutable manifests, renderer/evaluator parity, and sketch-specific information/mask tests. Preserve raw generation artifacts before model integration so evaluation can be tested independently.

### Stage 2: smallest JAX model extension

Implement the chosen A output family and its no-support/explicit-context variants; integrate the existing fast writer/READ path; verify full meta-gradients and task resets. Do not introduce a new meta-learning library or transformer framework without necessity.

If no donor policy checkpoint is available, reproduce its rendering/feature pathway from saved or real sketches and use JAX-native smoke training. Reproducing old training results is not a prerequisite to testing the new matched models; legacy feature parity still is.

### Stage 3: A pilot

A reasonable initial contrast is F=8 versus F=128 when category availability permits, with K=4, one model seed, one subset seed, one common finite drawing budget and bounded generation evaluation. These values are starting choices, not evidence-derived optimum settings. Pick exact training steps from a smoke profile before launch and log valid-event exposure.

Start with two TTT/direct-context pairs and the corresponding no-support references; do not launch the Cartesian product of objectives, capacities, F, K and seeds. Small evaluation samples diagnose plumbing; large-sample FD and replicated conditional metrics are required for conclusions.

### Stage 4: B1 then B2

Reuse the same support writer and evaluation exports. Implement frame replay/oracles first, then learn instance-specific reproduction, then add closed-loop disturbances. Reuse A's infrastructure without forcing its stochastic head into a deterministic control problem.

### Stage 5: controlled scaling

Expand F/N levels, independent subsets and training seeds. Add the supervised-WRITE comparator and first-order ablation, then selected capacity sensitivity. Keep validation-driven development distinct from the final held-out evaluation.

### Stage 6: optional C and extensions

Only then add C, transformer/AR replications, continuous task augmentations, selective writing or longer support. No massive drawing run, foundation-model features or robotic dataset conversion is required to begin.

---

## 13. Failure interpretation and safeguards

| Observation | Next diagnostic; do not jump directly to “insufficient diversity” |
|---|---|
| Legacy FID and integrated FID differ on identical drawings | Rendering, checkpoint, input scale, channel order, L2 normalization or covariance numerics. |
| Low pooled FD but poor conditional score | Ignoring support, category swapping, or an imbalanced mixture. |
| Strong A but poor B with same-category wrong supports | Category inference without instance-specific program use; a meaningful dissociation. |
| KVB loss falls but query loss does not | WRITE/READ alignment, fast-subspace placement, scales, masking, or inadequate representation. |
| Ordinary adaptation works only when slow heads change | The current fast module is not an established sufficient subspace. |
| KVB fails but supervised WRITE succeeds | Auxiliary WRITE alignment, not yet evidence for a diversity threshold. |
| Both TTT and explicit context improve similarly with F | Diversity helps; no demonstrated TTT-specific threshold advantage. |
| TTT no-update is very poor while correct support matches a strong no-support model | Possible self-created dependence, not useful adaptation beyond a support-free baseline. |
| More families also increase valid token exposure | Data/compute confound; rerun controlled or report as combined scaling. |
| B1 learned model works but loses to transformed replay | Demonstrated memory/reproduction; not evidence of an improved control algorithm. |
| Good teacher-forced loss and poor rollout sketches | Exposure shift, generation settings, STOP/pen decoding or stochastic-head calibration. |
| Wrong action labels have little effect | Other support channels may still identify the task; inspect information remaining in next-state/geometry evidence. |
| Padding length changes post-support W | Masking, drift on empty segments, false transitions or position-encoding dependence. |

All experimental variants must retain the originally intended query task when applying counterfactual supports. Randomizing support does not redefine what “correct” means for scoring.

---

## 14. Required outputs, checkpoints and handoff

Codex should produce a small integrated set of capabilities, not a prescribed collection of dozens of new files:

1. Native QuickDraw data/manifests and A/B/C task construction, with C optional.
2. A JAX sketch output model and shared fast-weight integration, plus isolated comparators.
3. Deterministic generation/rollout export and a frozen external Sketch-FD evaluation interface.
4. Category-conditional and instance/control metrics, with parity/invariance tests.
5. Existing-style train/eval entry points, configs, provenance, exact resume and diagnostics.
6. Bounded pilot and analysis tooling; separate commands for preparing manifests, training, generating, scoring and aggregating.
7. A concise implementation summary explaining reused components, extensions, scientific contracts and remaining limitations.

Retain the branch's checkpoint rule: slow parameters include W0 and learned rates; transient task-adapted W is not a global checkpoint parameter. Saving task deltas as explicitly named analysis artifacts is acceptable, separate from model checkpoints. Persist optimizer, all necessary RNG streams, sampler position, manifests, eligibility/normalization metadata, architecture/WRITE schema, scientific config and code revisions.

Generation records must carry intended task, support IDs, conditions, frame/operator public inputs, raw untrimmed outputs where useful, effective lengths, STOP state, seed and model ID. Score artifacts additionally record extractor hash, reference IDs, renderer and feature version, actual counts and uncertainty method. Expensive generation should not need repeating merely to correct aggregation.

Respect the destination's locked environment and optional CUDA setup. Do not upgrade packages solely to match current documentation. Keep new optional drawing/metric dependencies isolated; do not alter `phi_mujoco` for this task. A full robotics test run requiring unavailable local assets must be reported separately from passing self-contained sketch/core tests.

### Final acceptance summary

Implementation is accepted when A and B pipelines are runnable in ICIL/JAX; all support/label boundaries are tested; the existing fast writer is reused or minimally factored; full-gradient tests pass; metric parity is established; primary sampling avoids full-query retrieval; independent splits and budgets are recorded; and the matched pilots produce reproducible artifacts. C is accepted as optional implementation only if its identifiable operator/oracle tests also pass.

Scientific success is assessed separately from software acceptance. Never suppress a negative result or modify the final test split to manufacture the proposed transition.

---

## 15. Reading references and purpose

Only a focused set is needed for this implementation. These are background and method references, not claims that the requested threshold has been established.

**[R1] Finn, Abbeel and Levine (2017), Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.**  
https://arxiv.org/abs/1703.03400  
Bilevel optimization and full differentiation through adaptation.

**[R2] Sun et al. (2024), Learning to (Learn at Test Time): RNNs with Expressive Hidden States.**  
https://arxiv.org/abs/2407.04620  
Fast weights as recurrent state and learned self-supervised WRITE. ICIL already implements the relevant mechanism; do not reimplement the paper wholesale.

**[R3] Raventos et al. (2023), Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression.**  
https://arxiv.org/abs/2306.15063  
Controlled diversity, capacity and empirical transitions. Its regression setting does not prove this project's hypothesis.

**[R4] Heusel et al. (2017), GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.**  
https://arxiv.org/abs/1706.08500  
Original FID formulation; distinguish Inception-FID from the existing sketch-ResNet feature distance.

**[R5] Binkowski et al. (2018), Demystifying MMD GANs.**  
https://arxiv.org/abs/1801.01401  
Kernel-distance estimation/KID; apply it transparently to sketch features.

**[R6] Chong and Forsyth (2020), Effectively Unbiased FID and Inception Score and where to find them.**  
https://arxiv.org/abs/1911.07023  
Finite-sample, model-dependent FID bias; equal sample counts and real-real subtraction are not complete corrections.

**[R7] Parmar, Zhang and Zhu (2022), On Aliased Resizing and Surprising Subtleties in GAN Evaluation.**  
https://arxiv.org/abs/2104.11222  
Freeze rendering and preprocessing for reproducible feature metrics.

**[R8] Ha and Eck (2017), A Neural Representation of Sketch Drawings.**  
https://arxiv.org/abs/1704.03477  
Probabilistic stroke modeling and coordinate/event outputs; useful if selecting an AR replication.

**[R9] JAX documentation: transformations and scan.**  
https://docs.jax.dev/en/latest/automatic-differentiation.html  
https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html  
Composable differentiation and fixed-shape sequential state. Use the installed version's supported behavior.

**[R10] JAX documentation: gradient checkpointing.**  
https://docs.jax.dev/en/latest/gradient-checkpointing.html  
Rematerialization versus changing the objective by stopping gradients.

**[R11] JAX documentation: numerical gradient checks.**  
https://docs.jax.dev/en/latest/_autosummary/jax.test_util.check_grads.html  
Directional finite-difference checks for the new sketch meta-objective.

**[R12] Yin et al., Meta-Learning without Memorization.**  
https://arxiv.org/abs/1912.03820  
Support dependence versus task-recognition shortcuts; motivation for separating A from instance-specific B.

---

## 16. Repository source index and review boundaries

### Destination ICIL source

All links below pin `87214e5160152cd85fe8a366b2b43bc391e60f12`; Codex must inspect later local changes.

**[I1] Agent instructions.**  
https://github.com/Ricvalp/icil/blob/87214e5160152cd85fe8a366b2b43bc391e60f12/AGENTS.md  
Read in full: deliberate legacy removal, package boundaries, adaptation-only invariants and verification requirements.

**[I2] Fast model, initialization, encoding and READ.**  
https://github.com/Ricvalp/icil/blob/87214e5160152cd85fe8a366b2b43bc391e60f12/icil_jax_rlbench/models/fast_weight_ttt.py  
Reviewed lines 1–260: state config, per-tensor rates, K/V normalization, generic fast model and absolute/delta READ.

**[I3] Adaptation and encoded-support bridge.**  
https://github.com/Ricvalp/icil/blob/87214e5160152cd85fe8a366b2b43bc391e60f12/icil_jax_rlbench/models/fast_weight_ttt.py  
Reviewed from line 260 through the final diagnostics: state loss, segmentation, scan-based WRITE, first-order switch, encoded registers, query metrics and per-state rank.

**[I4] Meta-training step.**  
https://github.com/Ricvalp/icil/blob/87214e5160152cd85fe8a366b2b43bc391e60f12/icil_jax_rlbench/train/ttt_step.py  
Reviewed lines 1–280: state-specific outer loss, task vmap, nested gradients, training state, optimizer/distribution and alignment.

**[I5] Project configuration.**  
https://github.com/Ricvalp/icil/blob/87214e5160152cd85fe8a366b2b43bc391e60f12/pyproject.toml  
Read in full: JAX/Flax/Optax dependencies, uv groups, editable sibling and package name. Preserve lockfile/environment rather than repeating version assumptions in this plan.

**[I6] README.**  
https://github.com/Ricvalp/icil/blob/87214e5160152cd85fe8a366b2b43bc391e60f12/README.md  
Reviewed lines 1–180: current project organization, setup, core invariant and state workflow.

**[I7] Implementation summary.**  
https://github.com/Ricvalp/icil/blob/87214e5160152cd85fe8a366b2b43bc391e60f12/IMPLEMENTATION_SUMMARY.md  
Reviewed through current next-experiment section. Its numerical results are repository-reported, not independently reproduced here; they are not acceptance thresholds for QuickDraw.

**[I8] Update-information analysis.**  
https://github.com/Ricvalp/icil/blob/87214e5160152cd85fe8a366b2b43bc391e60f12/icil_jax_rlbench/analysis/metaworld_update_information.py  
Reviewed lines 1–190: domain-specific imports, representation inventory and probe utilities. Do not wholesale import the simulator-specific module into sketches.

### QuickDraw reference source

The earlier source audit is pinned to `e76b1e6794df7205785845d3475eb015fdb6f544`. These are reference assets, not a second training implementation required by the new plan.

**[D1] Feature extractor and training.**  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/metrics/resnet18.py  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/metrics/train_resnet18.py

**[D2] Rendering and exact metric configuration.**  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/dataset/rasterize.py  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/configs/metrics/cache.py

**[D3] Legacy generated/real/reference FID evaluation.**  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/diffusion/evaluate_encoder_decoder_in_context_imitation_learning.py  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/diffusion/evaluate_decoder_only_in_context_imitation_learning.py

**[D4] Full-query similarity retrieval and embedding generation.**  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/dataset/episode_builder.py  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/metrics/compute_embeddings.py

**[D5] Loading, collator schemas and generation.**  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/dataset/loader.py  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/dataset/diffusion.py  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/diffusion/sampling.py

**[D6] Existing stochastic output formulation.**  
https://github.com/Ricvalp/quick-robot-draw/blob/e76b1e6794df7205785845d3475eb015fdb6f544/diffusion/policies/dit_encdec_policy.py  
Noise-prediction DDPM loss and chunk generation. Use for semantic continuity, not as a request to convert old policy weights.

---

## Final instruction to Codex

Integrate the QuickDraw diversity study into the current ICIL JAX branch. Preserve its existing fast-weight mechanism and scientific controls, add only the sketch-specific data/output/evaluation components needed for A and B, and keep C optional. Retain the original frozen sketch-FID assets through an offline interface. Use this document to decide what must be measured and prevented, while choosing the lowest-complexity implementation consistent with the repository. Produce reproducible pilots and an honest record of results, not a predetermined diversity threshold.
