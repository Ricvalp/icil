# Proposed context-copy evaluation (awaiting agreement)

This proposal is not enabled in training. FID measures distributional similarity:
a policy that selects and copies a context drawing could achieve a good score.
Novelty and fidelity therefore need separate measurements.

## Recommended measurements

Measure the minimum symmetric Chamfer distance between a generated drawing and
each of its actual K demonstrations. Decode incoming pen states and resample
uniformly along the drawn segments, never across pen-up moves. Center each
drawing and apply isotropic scale normalization, so translating or resizing a
copy still counts as copying. Retain orientation: rotation and reflection can
change the meaning of a sketch. Chamfer ignores drawing order, which is useful
for detecting visual copies drawn with a different action sequence.

Report a separate pen-aware dynamic time warping (DTW) distance for action
similarity. This compares execution order while accommodating different
traversal speeds. It is a diagnostic, not an additional automatic success
criterion. Its multistroke cost and pen-event weighting require validation.

Calibrate a near-copy threshold on real held-out query sketches compared with
their own NN contexts, using the same K and retrieval mode as the policy.
Start with a declared 1% false-positive target; measure the achieved rate on
separate calibration-validation pairs. Freeze the calibration population,
normalization and threshold before comparing policies. Test sensitivity using
exact copies, small perturbations, resampling, translation and scaling. Inspect
borderline matches and report uncertainty in the real baseline.

Report near-copy fraction, real-query baseline, and invalid/empty fraction
separately. Empty or nonfinite drawings must never be counted as successful
novel sketches. A high geometric distance alone also cannot establish quality
or adherence to the requested neighborhood. Chamfer can miss structural
differences in simple line drawings, so the rate should remain an audit signal.

## Literature and limitations

[Ge et al., Creative Sketch Generation (ICLR 2021)](https://arxiv.org/html/2011.10039v2)
audit novelty by inspecting nearest training drawings under Chamfer distance.
This is a direct sketch-generation precedent, but does not establish a universal
binary copying threshold.

[Carlini et al., Extracting Training Data from Diffusion Models (USENIX 2023)](https://arxiv.org/html/2301.13188v1#S5.SS1)
use local-neighborhood normalization to distinguish copying from ordinary
similarity. Their natural-image thresholds do not transfer to QuickDraw vectors.
The NN-conditioned setting here makes matched real-query calibration essential.

[Meehan et al., A Three Sample Hypothesis Test for Evaluating Generative Models
(AISTATS 2020)](https://proceedings.mlr.press/v108/meehan20a.html)
compare generated-to-training proximity with independent real-to-training
proximity. Their statistical guarantees do not automatically apply to our
overlapping, query-selected contexts. [Bhattacharjee et al. (ICML
2023)](https://proceedings.mlr.press/v202/bhattacharjee23a.html) further examine
limitations of aggregate copy tests and formalize a local notion of copying.

[Schmidt and Weber (2018)](https://journals.sagepub.com/doi/10.3233/IDT-180326)
study DTW for single-stroke input matching; the proposed pen-aware multistroke
extension is our adaptation. [Pizzi et al., SSCD (CVPR
2022)](https://arxiv.org/abs/2202.10261) offer a learned image-copy descriptor as a
possible raster alternative, but sparse-sketch transfer would require its own
calibration. The existing classification ResNet is not a trained copy detector.

Existing repository routines report exact trajectory matches, raw point
Chamfer, or nearest coarse geometry descriptors. They do not implement the
stroke-resampled, calibrated near-copy rate described above.
