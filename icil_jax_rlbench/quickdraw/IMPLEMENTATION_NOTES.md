# QuickDraw implementation decisions

## Source and scope

- Work on `quick-robot-draw`, created from merged `master` at `3af009b`.
  The user's branch instruction supersedes the older branch names in the plan.
- Reference source: sibling `quick-robot-draw`, `refactor-dataset`,
  `e76b1e6794df7205785845d3475eb015fdb6f544`.
- Preserve the existing MetaWorld paths and sibling `phi-mujoco` dependency.
- Implement A, then B1 and B2. C and large sweeps remain disabled.

## Neighborhood correction (2026-09-08)

The user's [addendum](../../QUICK_ROBOT_DRAW_ICIL_JAX_NEIGHBORHOOD_CORRECTION.md)
supersedes the category-only task definition. A-NN uses offline target-centered
cosine retrieval; A-local samples supports and targets from a declared local
neighborhood. A-category remains the coarse-task ablation. Primary diversity
sweeps keep the exact coarse category set fixed and vary neighborhood count.

Preserve the JAX autoregressive mixture head, KVB WRITE, full meta-gradients,
fast-state resets, comparators, B1/B2, and the external rendering/ResNet boundary.
Change the numerical manifests/samplers, resource provenance, conditional
evaluation, exposure accounting, and bounded launch configuration. Embeddings
are permitted for offline task construction and evaluation. They, anchor IDs,
neighbor scores, and hidden target information remain outside policy inputs.

Reuse the completed vector and classifier/policy caches. The prepared policy
pool has 1,725,000 drawings; classifier train/validation/test have
690,000/69,000/69,000. The full real-data classifier in
`outputs/quickdraw_evaluator/resnet18_v1` has completed two epochs and can resume
using these same rasters. Its trainer now logs directly to W&B and retains
detailed per-category results on disk while printing compact epoch summaries.
The two-update smoke checkpoint remains diagnostic only.
Old immutable provenance labels reflect the resource's creation;
new embedding/task manifests disclose its additional offline curation role.

Implemented contracts and local choices:

- Immutable manifests record split-local exact cosine rankings, duplicate
  exclusions, held-out regions/categories, reserved anchors/references, overlap,
  fixed category sets, and nested neighborhood subsets. FAISS exports mirror
  the NumPy curation rankings without entering the JAX process.
- Evaluation keeps intended targets/references fixed under separated
  same-category wrong-neighborhood supports, adds conditional raw-feature MMD
  and real-set swap checks, and retains pooled FD plus geometry/copy baselines.
- Training records realized support/query/pair reuse. A-local supports a fixed
  available reservoir and a separately named per-neighborhood allowance regime.
  Pilots match optimizer steps and report actual valid-token mismatches; they
  do not claim an exact token-matched or independent-task intervention.
- The existing AR mixture head supplies conditional target loss. No diffusion
  migration, JAX mechanism redesign, or B1/B2 scope change is required.

Validation after the direct classifier logging update on 2026-09-08: 158 tests
passed with one optional original-checkpoint parity skip. Obsolete uploader
tests were removed with that script. Actual offline W&B history and exact
classifier resume passed, as did a one-step JAX TTT smoke run. Compilation,
forbidden-import and whitespace checks passed.
Real JAX neighborhood training and exact resume, donor feature/retrieval parity,
split-local FAISS, metadata invariance, and conditional/copy controls passed on
explicit fixtures. Full classifier training, real-drawing conditional gates,
and scientific pilots remain pending; fixture checks establish software
behavior only. See [the continuation commands](NEIGHBORHOODS.md).

## Reuse and extensions

Reuse functional fast initialization, normalized key/value projections, positive
per-tensor update rates, clipped differentiable updates, delta READ, Optax train
state and atomic checkpoint storage. Extend shared fast helpers only where
needed for sketch outputs and exact no-ops on empty WRITE segments.

Use a compact autoregressive mixture-density coordinate head with explicit
incoming-pen and STOP probabilities for A. This is the plan's permitted AR
alternative to diffusion. All A comparators use the same output family and
causal query backbone. B1 and B2 use matched deterministic heads, public frames,
and explicit pen/STOP supervision. Category and drawing IDs remain metadata.

Native numerical sketch records and immutable manifests separate category
diversity, base-program diversity and drawing budgets. Public generation limits
do not reveal future target length. B2 feeds executed states into its policy.

B2 complete demonstrations additionally expose public program-knot markers,
distinguishing authored waypoints from initial pen-up travel and integrator
subdivision points. KVB/context evidence includes these markers; query READ
ignores them. Transformed replay reconstructs the supplied execution's waypoint
positions using these markers, then constructs the requested frame's timed
reference. Supervised WRITE retains its causal prediction-loss input contract.

The explicit-context comparator is a separately named sketch model. It does
not enable direct support access in the TTT model or any robotics policy.

## Evaluator boundary and assets

Update: the user explicitly requested rebuilding the data and training a new
ResNet classifier from scratch. This supersedes the original-weights requirement
for the new experiment. Its checkpoint is labeled as a newly trained evaluator,
with known drawing splits and label mapping, and will be frozen across policy
comparisons. Its scores are not numerically comparable to historical scores
from `resnet18_step40000.pt`. Original-checkpoint parity remains a separate
optional legacy check. See [the fresh workflow](FROM_SCRATCH.md).

Keep the donor's grayscale ResNet18 and canonical renderer in an isolated
offline Python process. Preserve raw 512D features, the recorded 64x64/AA2/
line-width-2 rendering convention and the incoming-segment pen convention.
Record extractor/source hashes. Frozen features define offline neighborhoods
and evaluation scores; evaluator weights and features never enter the JAX
policy encoders, WRITE targets, or gradients.

At the initial audit, real QuickDraw data and the original evaluator checkpoint
were absent. The user has since prepared real data and chosen fresh evaluator
training. Explicit synthetic fixtures test software only. Real-data comparative
pilots require the fully trained frozen evaluator, with no random or ImageNet
feature substitution.

## Interpretation

Software correctness and held-out adaptation are separate outcomes. Preserve
negative pilots. Report pooled Sketch-FD, conditional sketch-feature MMD and
instance/control metrics separately. Across-task participation ratio and
within-matrix effective rank are different quantities. Actual update alignment
uses `-dot(query_gradient, update)` so positive predicts local improvement.

## Local implementation record (2026-09-07)

Final verification: the complete repository suite passed on CPU (122 passed,
one explicit original-checkpoint parity skip). All 19 sketch-model tests also
passed on the local NVIDIA GPU. Compilation, forbidden-import checks, and Git
whitespace checks passed. The GPU batching test records float32 tolerance and
separately verifies per-task fast states, clipped update norms, and query losses.

The fixture cache is `datasets/quickdraw_fixture`, identifier
`a75964084e5518116f48291ae7372ffe87d3befab26588279558e1d080674ddd`.
It contains 16 synthetic categories with 64 independent records each. None of
the following runs establishes real QuickDraw performance.

- GPU two-step full-KVB training completed. Exact checkpoint resume is tested
  against uninterrupted training, including Optax/JAX state, all four sampler
  streams, observed IDs, and subsequent sampled batches.
- `outputs/quickdraw_smoke/pilot_final/pilot.json` records F=2 and F=8,
  U=64, two supports, one query, 32 output steps, seed 0, and 50 optimizer steps
  for each of full KVB, explicit context, and independent no-support. These
  tiny models have 1,520, 1,356, and 660 parameters respectively. Capacity is
  fixed across F, but parameter counts are not matched across mechanisms.
  Evaluation uses two independent support sets per development category and
  paired generation keys. Empty/very short outputs remain in the artifacts;
  this budget did not establish competent generation or support-specific gain.
- The same pilot directory contains frozen reference halves, galleries,
  `aggregate.json`, actual-checkpoint update diagnostics, and A copying-input
  populations. Feature scores remain pending the original extractor asset.
- `outputs/quickdraw_smoke/gates/report.json` records full-graph fixed-batch
  query loss 2.13306 -> -0.893735 over 50 steps. The independently trained
  no-support reference reaches -0.825852. These are continuous-density NLLs,
  so negative values are valid.
- The Q4 supervised adaptation check optimized exactly 212 fast parameters
  with slow weights frozen. Its fixed 25-step budget gave mean query gain
  0.316093 across two optimized synthetic tasks; per-task gains were -0.127237
  and 0.759423. One task worsened. This is not held-out adaptation evidence.
- `outputs/quickdraw_smoke/b_checks/report.json` records actual full-TTT B1/B2
  training and held-out synthetic program exports. Transformed replay matches
  the oracle within 5e-7; B2 executes bounded motion and feedback reduces
  perturbation error. Neural two-step runs are execution checks only.

The implementation is single-device float32 JAX. Finite differences use a
smooth high-precision outer fixture while documenting the shared writer's
float32 normalization. Source hashes, actual reservoir identifiers, parameter
groups, sampled versus consumed exposure, and measured runtime accompany runs.
B2 stroke reversal is explicitly unavailable. Broad capacity/seed/convergence
sweeps, C, real-data pilots, and original-checkpoint feature parity remain
outside this completed synthetic validation. No positive research gate or
diversity threshold is claimed.
