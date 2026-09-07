# AGENTS.md

Instructions for coding agents working on the `fast-weight-ttt` branch.

## Hard Constraints

- Do not import from `icil`, `icil_jax_query_memory`, `diagnostics`, or the
  upstream `metaworld` package. Use only public `phi_mujoco` contracts for the
  explicitly requested MetaWorld policy experiment.
- The legacy direct-regression pretraining, parameter-MAML, and memory-MAML paths
  were intentionally removed. Do not restore them unless the user explicitly
  asks.
- Keep implementation under `icil_jax_rlbench/` unless external tooling is
  explicitly requested.
- Do not add old PyTorch checkpoint conversion unless explicitly requested.
- Avoid destructive Git operations and never revert unrelated user changes.

## Project Purpose

This branch tests whether one or a few demonstrations can produce useful
gradient-based test-time adaptation for robot imitation. It starts with a
controlled hidden-goal state benchmark and moves to RLBench only after the state
diagnostic gates pass.

The main method uses a small fast-weight module, full second-order gradients, a
key-value reconstruction WRITE objective, and an outer action-imitation READ
objective. FOMAML and support action-BC WRITE are ablations.

## Scientific Invariants

- Support influences query predictions only through an explicit fast-state
  update. Do not add direct support cross-attention, support FiLM, task labels,
  language, or support-token initialization of READ memory to the main path.
- Query prediction accepts only slow parameters, fast state, and query
  observation. Query actions must never enter the query encoder or READ path.
- Support and query are different episodes with different starts but the same
  hidden task latent.
- Reset fast state to meta-learned `W0` at every task boundary. Adapt on support,
  then freeze and reuse that state for declared query episodes.
- Support WRITE losses train adaptation but are not added to the outer objective.
- Full second-order differentiation is the main method. `first_order=True` is a
  named FOMAML ablation and must visibly remove the expected meta-gradient paths.
- Never save transient task-adapted fast state in checkpoints or optimize it with
  the slow optimizer.

## Gate Order

1. Verify benchmark integrity and train a competent query-only policy.
2. Pass Gate 1: ordinary support adaptation improves independent held-out query
   episodes.
3. Pass Gate 2: fixed-meta-batch loss, finite-difference, JIT, reset, and update
   direction checks.
4. Pass Gate 3: held-out goals show support-specific adaptation under matched
   controls and multiple seeds.
5. Only then implement and run end-to-end RLBench TTT.

Do not interpret a passing Gate 2 as evidence of held-out adaptation.

## Directory Structure

- `data/metaworld_hidden_goal.py`: task-aware phi cache, normalization, and
  support/query samplers shared by ML1 Reach and ML1 Push.
- `train/metaworld_query_runner.py`: support-free MetaWorld behavior cloning.
- `eval/metaworld_hidden_goal_gate1.py`: ordinary-adaptation upper bound.
- `eval/metaworld_hidden_goal_ttt.py`: held-out fast-weight support controls.
- `eval/metaworld_policy.py`: JAX implementation of the phi policy interface.
- `data/hidden_goal.py`: controlled environment, task splits, normalization,
  samplers, and integrity checks.
- `models/fast_weight_ttt.py`: state-policy KVB WRITE and gated READ.
- `train/ttt_step.py`: full/FOMAML meta-objectives and train steps.
- `train/ttt_runner.py`: training, validation, provenance, checkpoints, resume.
- `eval/ttt_state_gate1.py`: ordinary-adaptation upper bound.
- `eval/ttt_state_gate2.py`: implementation-correctness overfit gate.
- `eval/ttt_state.py`: held-out closed-loop support controls.
- `models/ttt_supernode.py`: self-contained RLBench visual register encoder.
- `models/robotics_actions.py`: RLBench action encoding and component losses.
- `data/h5_cache.py`: retained dense-cache reader for the later visual phase.

## Controlled Benchmark Contract

The controlled scientific experiments use the `phi_mujoco`
`metaworld_ml1_reach` and `metaworld_ml1_push` integrations: 40/10/50 disjoint
tasks, hidden goal slots, 39D state, and 4D continuous action. Never pass task
IDs or provenance goals to the model. Fit observation and action normalization
only from training-task episodes. Support, offline query, and fresh closed-loop
query starts must not overlap.

The synthetic diagnostic has a 2D goal excluded from `[x, y, gripper, phase]`
and a normalized planar-delta-plus-binary-gripper action. Its train,
validation, and test goals are also disjoint.

Meta-batches retain explicit task, support-demo, support-time, query-demo, and
query-time axes. Do not collapse those semantic axes in the sampler.

## RLBench Contract

The later visual path uses dense H5 episodes with `xyz`, `valid`, `state`, and
`action`, plus optional `rgb` and `mask_id`. Infer dimensions with
`RLBenchCacheStore.infer_dims()`.

Use space-time supernodes, separate state/action/time token types, and small event
registers. Query encoding must not accept demonstrated query actions. Use delta
translation, continuous 6D rotation with geodesic loss, and gripper BCE.

## Coding Guidelines

- Prefer explicit functional JAX code and static shapes inside JIT/pmap.
- Keep shape comments near nontrivial reshapes, scans, and vmaps.
- Add abstractions only when they remove real duplication or clarify semantics.
- Keep source ASCII and comments concise.
- Use `apply_patch` for manual edits.
- Do not commit generated outputs, checkpoints, caches, W&B data, or bytecode.

## Verification

Always run:

```bash
uv run --frozen --group metaworld python -m compileall -q icil_jax_rlbench tests
uv run --frozen --group metaworld pytest -q
rg -n "^(from|import) (icil|icil_jax_query_memory|diagnostics|metaworld)(\\.|\\s|$)" \
  icil_jax_rlbench tests
```

For model or training changes, also run Gate 2 or a one-step TTT smoke job. For
visual changes, run `tests/test_ttt_supernode.py` and
`tests/test_robotics_actions.py`.
