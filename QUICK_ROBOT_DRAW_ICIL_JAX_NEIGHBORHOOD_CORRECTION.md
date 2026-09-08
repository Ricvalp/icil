# Correction: similarity-defined tasks for the QuickDraw ICIL/JAX study

**Applies to:** `QUICK_ROBOT_DRAW_ICIL_JAX_IMPLEMENTATION_PLAN.md`  
**Destination:** `Ricvalp/icil`, branch `fast-weight-ttt`  
**Reference implementation:** `Ricvalp/quick-robot-draw`, branch `refactor-dataset`  
**Date:** 8 September 2026  
**Document type:** targeted addendum, not a replacement implementation plan.

## 1. Decision and precedence

The user intentionally selected in-context demonstrations using nearest neighbors under cosine similarity of frozen sketch-classifier embeddings. That selection defines finer-grained tasks than the human category labels. The previous plan incorrectly restricted it to legacy reproduction and made random same-category sampling mandatory.

**Promote similarity-defined tasks to the main Experiment A.** Preserve the original query-centered nearest-neighbor protocol as a first-class experiment. Add a neighborhood-centered variant for controlled task-diversity sweeps. Retain random same-category sampling as a coarse-task ablation, not the primary definition of A.

This addendum overrides conflicting instructions about task construction, classifier use, A's diversity axis, evaluation, and acceptance criteria. All other instructions remain in force. In particular:

- Policy learning, WRITE, full-second-order meta-gradients, optimization, and generation stay in JAX.
- Reuse the existing fast-weight mechanism, delta READ, explicit resets, checkpoint conventions, and tests.
- Support reaches the main query policy only through adapted fast weights. The explicit-context comparator remains a separately identified baseline with the same output family.
- Keep the frozen sketch ResNet, renderer, and Sketch-FD evaluation through the external offline interface.
- B1/B2 remain instance-specific motor-program experiments; C remains optional. Do not redesign them because of this correction.
- Do not restart working implementation or overwrite unrelated changes. Make the smallest coherent changes to manifests, sampling, evaluation, and launch configuration.

Codex should read the local implementation and applicable agent instructions, then instantiate these contracts using its existing abstractions. No new framework or prescribed module layout is requested.

### Explicit overrides in the existing plan

| Existing location | Correction |
|---|---|
| Section 0, pretrained-feature restriction; Section 3.4 | The frozen classifier may be used for **offline task construction as well as evaluation**. It may not provide policy inputs, supervised WRITE targets, or gradients. |
| Section 4.2, sampling policy | Replace the requirement for independent same-category support with the A-NN / A-local protocols below. Completed-query-based offline curation is allowed in A-NN. |
| Section 4.3, A splits and eligibility | Add drawing/duplicate-cluster and neighborhood-region splits. Whole-category holdout becomes one distinct evaluation axis, not the sole definition of generalization. |
| Section 7, Experiment A | Replace category-only tasks and category-count-only sweeps with local similarity tasks and separate category/neighborhood diversity axes. |
| Gate Q2, Section 10 | Remove the blanket assertion forbidding completed-query retrieval. Test the protocol-specific information boundary and split integrity instead. |
| Sections 11–12, analysis and A pilot | Count neighborhoods, overlap, drawing exposure, and fixed coarse-category coverage; use the revised bounded pilot below. |
| Sections 13–14, interpretation and acceptance | Require fine-grained conditional evaluation and same-category wrong-neighborhood controls. Replace “primary sampling avoids full-query retrieval” with “declared offline curation; no forbidden target information reaches inference.” |

Do not leave old prohibitions active in configuration validators or tests while merely adding a new sampler flag.

## 2. Why the task definition changes

With independent same-category sampling, once category `c` is known, the particular support drawings do not identify a narrower query distribution:

\[
p(y\mid S,c)=p(y\mid c).
\]

A policy that learns only the category can therefore be behaving correctly in that experiment. The user instead wants support to identify a local shape/style distribution within a category: for example, one family of side-view cats rather than every kind of cat.

The intended latent task is approximately:

\[
\tau=(\text{coarse category},\text{local shape/style neighborhood}).
\]

Treat this as the experimental construction, not proof that classifier features capture all meaningful trajectory distinctions. Raster similarity may identify appearance while ignoring stroke order. Experiment A measures fine-grained demonstration-conditioned generation; B remains the test of a particular motor program and control behavior.

Do not assume that every drawing or anchor defines an independent task. Overlapping neighborhoods may provide almost identical conditional distributions. The proposed critical-diversity effect remains a hypothesis.

## 3. A-NN: retain the user's query-centered protocol

Implement this first for continuity with `EpisodeBuilderSimilar` in the reference branch.

1. Choose an eligible target/query drawing `y` from the declared policy-data split.
2. Obtain its frozen ResNet embedding and normalize it for cosine retrieval.
3. Retrieve neighbors from the eligible same-category candidate pool.
4. Exclude `y`, its duplicate cluster, and any other ineligible IDs.
5. Select `K` distinct support drawings from the eligible neighbors.
6. Send only support trajectories to the JAX writer. Use `y` as a supervised query target, not as an inference input.

Provide two reproducibly named selection modes:

- **Exact top-K:** closest eligible neighbors, reproducing the original intention.
- **Sample-from-top-M:** retrieve a larger eligible pool and sample K distinct supports from it. This reduces support-set repetition without changing the similarity-based task concept.

Exact reproduction should also retain a separately documented legacy exact-ID-only exclusion mode if needed. The scientific protocol excludes declared duplicates as well. Any difference from historical results must be attributable to a named protocol change.

### Offline curation is not policy-side target leakage

A-NN defines the supervised joint distribution:

\[
y\sim p_{\mathrm{data}},\qquad S\sim p_{\mathrm{NN}}(S\mid y).
\]

It is legitimate to use complete drawings to construct this joint distribution offline. At deployment, the model receives the curated demonstrations and generates without receiving `y` or its embedding. This evaluates similarity-curated demonstration conditioning, not arbitrary independently supplied demonstrations.

The following remain forbidden at policy inference: complete query trajectory, query embedding, anchor ID, neighbor ranks/scores, category label, task label, target-derived total length or normalized phase, and hidden query-future features. Available query prefixes remain allowed only in a separately declared completion protocol.

Selecting supports with a completed target does not make exact reconstruction identifiable. Keep a stochastic output model and a conditional generative objective; do not require one generated drawing to match one arbitrary target point by point.

If N is implemented simply as the number of permissible query drawings in A-NN, report that it also limits the number of distinct training targets. Do not interpret this alone as an isolated task-diversity intervention. Use A-local for the cleaner N sweep.

## 4. A-local: neighborhood-centered tasks for diversity sweeps

This is the preferred construction for varying fine-grained task diversity while preserving coarse categories.

1. Select an anchor `a_tau` from a declared construction pool.
2. Form a same-category eligible neighborhood `N_tau` using frozen cosine similarity.
3. Exclude the anchor itself and its duplicate cluster from model examples.
4. Sample K distinct support drawings and one or more distinct query drawings from that neighborhood. Support and query IDs and duplicate clusters must not overlap within an episode.
5. Keep the anchor, embedding, neighbor scores, and neighborhood ID as metadata only.
6. Repeat support/query sampling to obtain different executions of the same local distribution.

A starting neighborhood size of 32–128 drawings with K=4 is reasonable for a pilot, subject to available eligible data. These are proposed settings, not evidence-derived optima. Fix the construction rule across the primary sweep; changing neighborhood radius or size changes task difficulty and is a separate experiment.

Use eligibility and deterministic tie-breaking consistently. For undersized or incoherent neighborhoods, follow a declared rejection/replenishment rule and log rejection rates. Do not silently expand to arbitrary same-category supports.

A-local defines a local distribution, not a unique hidden trajectory. Distinct anchors with nearly identical member sets should not be counted as strongly different tasks. Ambiguous support may also identify several nearby neighborhoods; evaluate distributional fidelity rather than requiring perfect anchor recognition.

Keep both A-NN and A-local. A-NN restores the user's intended experiment; A-local makes the diversity intervention easier to interpret. **A-category**, random same-category sampling, is the coarse-task comparator.

## 5. Frozen feature use and information boundaries

Use the existing ResNet checkpoint and rasterization convention. Record its hash, feature definition, preprocessing, available training provenance, and embedding/index versions. Keep them fixed across all policies, diversity levels, and support conditions.

The classifier may have been trained across all categories. That is permitted as a **disclosed, fixed task-curation resource**, including for held-out-category task construction. Consequently, the claim is:

> Policy task-diversity requirements under a fixed representation-based curation procedure.

It is not an end-to-end claim of discovering task structure without prior supervision. Distinguish policy-unseen categories from categories unseen by every component of the pipeline. Do not initialize the policy from all-category weights or provide classifier embeddings to its support/query encoder.

Maintain the external process boundary: embeddings, cosine indexes, and manifests can be prepared offline; the JAX learner consumes numerical trajectory records and permitted inputs. The policy process need not import PyTorch or FAISS. Do not retrain or convert the ResNet merely to implement this correction.

Keep feature conventions separate: cosine retrieval uses L2-normalized embeddings; legacy Sketch-FD uses its existing feature convention. Do not silently substitute normalized retrieval vectors for FID features.

## 6. Splits, manifests, and diversity controls

### Split the policy data before building split-local neighborhoods

Group exact/declared near duplicates first, then assign groups to training, development, and untouched test pools. Construct retrieval membership from the appropriate eligible pool. No held-out policy example may enter a training support or training query through a global index. A fixed pretrained curation model is allowed; crossing policy-data boundaries is not.

Report three distinct generalization regimes:

1. **Unseen drawings in familiar shape regions:** distinct IDs but similar local distributions. This tests exemplar generalization, not necessarily new tasks.
2. **Held-out shape regions within familiar categories:** reserve groups/regions using a deterministic, declared embedding-space rule, then build neighborhoods within those pools. Report train/test proximity and overlap diagnostics. Holding out anchor IDs alone is insufficient.
3. **Held-out categories:** reserve complete categories from policy training and construct their neighborhoods separately. This is the stronger semantic-transfer track.

Region boundaries are approximate. Do not claim perfect semantic separation from a clustering label. Choose splits without inspecting model test performance, and keep final evaluation tasks fixed across the diversity sweep.

### Change the principal diversity axis

Let `F` be the number of coarse training categories and `N` the number of permitted local training neighborhoods. For the primary study, **hold F and the exact category set fixed while varying N**. Choose category-balanced neighborhood subsets and repeat with independent subset seeds. Nested subsets are useful within each seed.

Preserve the original budget controls, now indexed by N:

- Fixed eligible drawing reservoir and matched optimization/valid-token exposure, varying neighborhood count. Log the actual unique drawings used; fixing the available reservoir does not guarantee equal realized exposure.
- Fixed per-neighborhood data allowance, with total data allowed to grow. Label this diversity-plus-data scaling.
- Fixed N with more examples/exposure, to test whether data volume explains the effect.

Neighborhood overlap may prevent exact matching of all quantities. Report the mismatch rather than inventing an “independent task” count. Include unique targets, unique supports, support-target pair counts, reuse frequencies, candidate-pool size, and total feature-curation resources.

Record neighborhood Jaccard overlap, anchor distances, within-neighborhood feature/geometry dispersion, lengths and stroke complexity, and empirical category frequencies. Sample a neighborhood first, then its drawings; do not let large neighborhoods dominate by sampling all trajectories uniformly.

Each immutable task/episode manifest must identify the protocol, split, intended neighborhood/category, hidden construction anchor where applicable, eligible member IDs, support/query IDs, duplicate groups, retrieval rule, tie policy, random seeds, and feature/index hashes. Only permitted trajectory inputs enter the model.

## 7. Evaluation corrections

### Keep Sketch-FD; add neighborhood-sensitive evidence

Retain pooled Sketch-FD and real-real calibration for overall generated quality. Category accuracy and category-conditional scores remain useful, but are insufficient when two tasks belong to the same category.

For A-local, prepare a fixed real reference reservoir for each evaluated neighborhood using drawings not shown as support in scored episodes. Reserve enough eligible examples before sampling episodes, or document a reproducible disjoint-reference scheme. Reference IDs and weighting must remain fixed across models and support controls.

Generate multiple outputs under each support set and evaluate neighborhood-conditional Sketch-KID/MMD, macro-averaged over tasks. Per-neighborhood FD is optional when sample counts support covariance estimation; do not rely on tiny-sample covariance scores. Retain the original plan's kernel, small-sample, and hierarchical uncertainty rules. Drawings sharing a support set or checkpoint are not independent experimental replicates.

For A-NN, retain held-out conditional target loss with paired diffusion noise/timesteps or the corresponding probabilistic objective. A deterministic neighborhood-reference protocol around the offline target can be added, with supports and duplicates excluded, but label it as a local-distribution proxy. Do not present top-K supports as independent samples from the true target-given-support distribution. Report A-NN and A-local separately.

For both, preserve actual sample counts, invalid/empty outputs, STOP rates, and the raw generation artifacts. A paired diffusion denoising loss is not exact likelihood.

### Required same-category negative

Replace support with a different, sufficiently separated neighborhood **of the same category**, matched approximately for length and complexity. Keep the original intended task, query targets/reference set, public query inputs, and random sampling keys fixed.

Report correct-support versus same-category-wrong-neighborhood performance alongside no update, independently trained no support, different-category support, and the existing corruption controls. Do not score wrong-support outputs against their replacement neighborhood; that would change the question being tested.

Use a declared separation rule so the “wrong” neighborhood is not effectively the same local distribution. Reordering support demos or reversing strokes is not guaranteed to destroy raster shape information; retain these as diagnostics, not mandatory failures.

### Copying and metric circularity

Retain support-copy and nearest-training-example baselines, raw-trajectory/geometry nearest-neighbor distances, duplicate rates, and diversity measures. Low local feature distance can result from copying.

Because the same representation constructs tasks and supplies Sketch-FD features, add at least non-classifier geometric and stroke-level checks plus deterministic galleries. Use a separately trained/frozen evaluator as a later sensitivity analysis when practical, not a prerequisite for the first pilot. Do not tune the task selector to maximize its own evaluation score.

Add an evaluation test that swaps generated sets between two same-category neighborhoods: pooled FD must remain unchanged, while a validated neighborhood-sensitive metric should worsen. First validate this using real drawings from clearly separated local distributions. If the conditional evaluator cannot detect the swap, it cannot establish fine-grained support use.

## 8. Revised gates and bounded launch order

Do not rerun completed JAX mechanism work without a regression reason. Change the existing integration gates as follows:

1. **Retrieval reproduction:** match frozen cosine neighbors and exclusions on a small donor fixture; then test the explicitly named duplicate-filtered and stochastic-neighbor variants. Record intended differences.
2. **Information/split integrity:** verify that offline retrieval has only the declared privileges, candidate membership is split-safe, and model input structures contain no query/anchor embeddings or IDs. With support and all public inputs fixed, changing metadata-only target/anchor identity must not change inference. Changing a target and consequently rebuilding its curated supports is a different legitimate episode, not this test.
3. **Fine-grained evaluation:** establish same-category real-set separation, pooled-score swap invariance, conditional-score sensitivity, reference/support disjointness, and copy baselines.
4. **Matched A-NN pilot:** use the existing selected JAX stochastic head, TTT, and explicit-context comparator. Reuse fixed episodes and noise keys. Include A-category on the same coarse categories as a targeted ablation. Do not require replacing the output family.
5. **A-local diversity pilot:** hold a small coarse-category set fixed and contrast two feasible N levels, for example F=8 with N=32 versus N=256 and K=4, only if eligibility and overlap checks support them. Keep the same test neighborhoods, available drawing budget, and bounded exposure. Start with one model/subset seed for plumbing, then replicate before claiming a transition.
6. Continue B1/B2 and optional C under the original plan. Expand N, seeds, capacities, and protocol replications only after the conditional evaluator and basic pilots work.

Add controls for the actual finite update direction, support-specific gains, and functional-response diversity using the existing analysis infrastructure. Family decodability alone is not a failure: the relevant question is whether support distinguishes local tasks within a family. Report update diversity across neighborhoods, separately from category diversity and single-fast-matrix rank.

A better result at larger N is not automatically a critical threshold. Preserve smooth-trend, no-threshold, and non-TTT-specific outcomes as valid conclusions. Compare TTT versus explicit context under the same task construction and disclosed resource budgets.

## 9. Codex deliverable and acceptance

Produce a short correction summary naming changed sampler/manifest/evaluator contracts, preserved implementation, and any necessary local deviations. Update stale assertions and launch defaults so they agree with this addendum. Do not replace the entire implementation plan or refactor unrelated model code.

This correction is implemented when:

- A-NN is a supported primary protocol, not confined to legacy evaluation.
- A-local supports reproducible fixed-category neighborhood-count sweeps.
- A-category remains available as an ablation.
- Offline classifier-based curation is allowed and disclosed, while policy inputs remain unprivileged.
- Splits, duplicates, task overlap, and data/exposure counts are auditable.
- Existing Sketch-FD is retained and neighborhood-specific support use is independently tested.
- JAX training, full-second-order WRITE, and the original B/C scope remain intact.

**Source anchors:** the parent plan's Sections 3–4, 7, 10–14 and donor references [D1–D5]. The reference nearest-neighbor implementation is `dataset/episode_builder.py::EpisodeBuilderSimilar`; embedding generation is in `metrics/compute_embeddings.py`; frozen features/Sketch-FD are in `metrics/resnet18.py`. These are continuity references from the prior source audit, not a new audit of subsequently changed branches. Inspect the current local versions before editing.
