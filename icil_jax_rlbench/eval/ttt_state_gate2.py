from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from absl import app
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np

from icil_jax_rlbench.data.hidden_goal import (
    HiddenGoalMetaSampler,
    HiddenGoalTaskBank,
    fit_state_normalizer,
)
from icil_jax_rlbench.models.fast_weight_ttt import (
    TTTAdaptConfig,
    segment_support,
    initial_fast_state,
    init_fast_weight_ttt_params,
    inner_learning_rates,
    query_imitation_loss,
    tree_difference_norm,
    tree_l2_norm,
    write_loss,
)
from icil_jax_rlbench.train.ttt_runner import (
    _optimizer,
    adaptation_config_from,
    fast_weight_config_from,
    hidden_goal_config_from,
    step_config_from,
    validate_adaptation_only_config,
)
from icil_jax_rlbench.train.ttt_step import (
    create_ttt_train_state,
    create_ttt_train_step,
    ttt_meta_objective,
)


_CONFIG = config_flags.DEFINE_config_file(
    'config', None, 'One-meta-batch full-second-order diagnostic.', lock_config=False
)


def _one_task(section):
    return jax.tree_util.tree_map(lambda value: value[0], section)


def _gradient_norms(gradient):
    return {
        name: float(tree_l2_norm(value))
        for name, value in gradient.items()
    }


def run_gate2(cfg) -> Path:
    validate_adaptation_only_config(cfg)
    benchmark_cfg = hidden_goal_config_from(cfg)
    task_bank = HiddenGoalTaskBank(benchmark_cfg)
    normalizer = fit_state_normalizer(
        benchmark_cfg,
        task_bank,
        episodes_per_task=int(cfg.benchmark.normalizer_episodes_per_task),
    )
    sampler = HiddenGoalMetaSampler(
        benchmark_cfg,
        task_bank,
        normalizer,
        split='train',
        seed=int(cfg.diagnostic.seed),
    )
    fixed_batch = jax.tree_util.tree_map(
        jnp.asarray, sampler.build_batch(int(cfg.train.batch_size))
    )
    model_cfg = fast_weight_config_from(cfg, benchmark_cfg)
    full_cfg = adaptation_config_from(cfg)
    first_cfg = TTTAdaptConfig(**{**full_cfg.__dict__, 'first_order': True})
    step_cfg = step_config_from(cfg)
    params = init_fast_weight_ttt_params(
        jax.random.key(int(cfg.diagnostic.seed)), model_cfg
    )

    def objective(value, adapt_cfg):
        return ttt_meta_objective(
            value, fixed_batch, model_cfg, adapt_cfg, step_cfg
        )[0]

    initial_loss = float(objective(params, full_cfg))
    full_gradient = jax.grad(lambda value: objective(value, full_cfg))(params)
    first_gradient = jax.grad(lambda value: objective(value, first_cfg))(params)
    eager_loss = objective(params, full_cfg)
    compiled_loss = jax.jit(lambda value: objective(value, full_cfg))(params)

    epsilon = float(cfg.diagnostic.finite_difference_epsilon)
    index = (0, 0)
    analytic = float(full_gradient['value_projection']['kernel'][index])

    def shifted_loss(delta):
        shifted = dict(params)
        projection = dict(params['value_projection'])
        projection['kernel'] = projection['kernel'].at[index].add(delta)
        shifted['value_projection'] = projection
        return float(objective(shifted, full_cfg))

    finite_difference = (shifted_loss(epsilon) - shifted_loss(-epsilon)) / (
        2.0 * epsilon
    )

    support = _one_task(fixed_batch['support'])
    query = _one_task(fixed_batch['query'])
    first_segment = jax.tree_util.tree_map(
        lambda value: value[0],
        segment_support(support, int(full_cfg.write_segment_size)),
    )

    def update_sign_diagnostic(current_params):
        fast_initial = initial_fast_state(current_params)
        write_gradient = jax.grad(write_loss, argnums=1)(
            current_params,
            fast_initial,
            first_segment,
            model_cfg,
            full_cfg,
            fast_initial,
        )
        rates = inner_learning_rates(current_params, model_cfg)
        forward_fast = jax.tree_util.tree_map(
            lambda value, rate, gradient: value - rate * gradient,
            fast_initial,
            rates,
            write_gradient,
        )
        reverse_fast = jax.tree_util.tree_map(
            lambda value, rate, gradient: value + rate * gradient,
            fast_initial,
            rates,
            write_gradient,
        )
        return {
            'query_before': float(
                query_imitation_loss(
                    current_params, fast_initial, query, model_cfg, full_cfg
                )[0]
            ),
            'query_after_forward_write': float(
                query_imitation_loss(
                    current_params, forward_fast, query, model_cfg, full_cfg
                )[0]
            ),
            'query_after_reversed_write': float(
                query_imitation_loss(
                    current_params, reverse_fast, query, model_cfg, full_cfg
                )[0]
            ),
        }

    initial_update_sign = update_sign_diagnostic(params)

    optimizer = _optimizer(cfg, params)
    state = create_ttt_train_state(
        params, optimizer, jax.random.key(int(cfg.diagnostic.seed) + 1)
    )
    train_step = create_ttt_train_step(
        optimizer, model_cfg, full_cfg, step_cfg
    )
    loss_curve = [initial_loss]
    for step in range(1, int(cfg.diagnostic.steps) + 1):
        state, metrics = train_step(state, fixed_batch)
        if step == 1 or step % max(1, int(cfg.diagnostic.steps) // 20) == 0:
            loss_curve.append(float(metrics['loss']))
    final_loss = float(
        ttt_meta_objective(
            state.params, fixed_batch, model_cfg, full_cfg, step_cfg
        )[0]
    )
    relative_reduction = (initial_loss - final_loss) / max(abs(initial_loss), 1e-8)
    trained_update_sign = update_sign_diagnostic(state.params)

    reset_a = initial_fast_state(state.params)
    reset_b = initial_fast_state(state.params)
    report = {
        'gate': 'one_fixed_meta_batch',
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'relative_loss_reduction': relative_reduction,
        'loss_curve': loss_curve,
        'full_meta_gradient_norms': _gradient_norms(full_gradient),
        'fomaml_meta_gradient_norms': _gradient_norms(first_gradient),
        'finite_difference': {
            'parameter': 'value_projection/kernel[0,0]',
            'analytic': analytic,
            'numeric': finite_difference,
            'absolute_error': abs(analytic - finite_difference),
        },
        'update_sign': {
            'at_initialization': initial_update_sign,
            'after_fixed_batch_overfit': trained_update_sign,
        },
        'jit_absolute_difference': float(jnp.abs(eager_loss - compiled_loss)),
        'reset_difference_norm': float(tree_difference_norm(reset_a, reset_b)),
        'checks': {
            'fixed_batch_overfit': relative_reduction
            >= float(cfg.diagnostic.expected_relative_loss_reduction),
            'write_projection_gradients_nonzero': (
                float(tree_l2_norm(full_gradient['key_projection'])) > 0.0
                and float(tree_l2_norm(full_gradient['value_projection'])) > 0.0
            ),
            'fomaml_removes_key_value_path': (
                float(tree_l2_norm(first_gradient['key_projection'])) < 1e-12
                and float(tree_l2_norm(first_gradient['value_projection'])) < 1e-12
            ),
            'finite_difference_close': bool(
                np.isclose(analytic, finite_difference, rtol=0.25, atol=2e-5)
            ),
            'jit_consistent': float(jnp.abs(eager_loss - compiled_loss)) < 1e-5,
            'reset_exact': float(tree_difference_norm(reset_a, reset_b)) == 0.0,
            'trained_forward_update_beats_reverse': (
                trained_update_sign['query_after_forward_write']
                < trained_update_sign['query_after_reversed_write']
            ),
            'trained_forward_update_improves_query': (
                trained_update_sign['query_after_forward_write']
                < trained_update_sign['query_before']
            ),
        },
    }
    output_dir = (
        Path(cfg.diagnostic.output_dir).expanduser().resolve()
        / datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / 'report.json').open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return output_dir


def main(argv):
    del argv
    run_gate2(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
