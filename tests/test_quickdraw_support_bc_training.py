"""Method identity and bounded compatibility with already trained policies."""

from dataclasses import replace

import pytest

from icil_jax_rlbench.quickdraw import policy_backend as backend
from icil_jax_rlbench.quickdraw import supervised_train as training
from icil_jax_rlbench.quickdraw.support_bc_models import SupportBCModelConfig


def test_support_bc_dispatch_and_checkpoint_identity_are_distinct():
    cfg = training.resolve_config({'method': 'support_bc'})
    model = backend.model_config(cfg['model'], cfg['method'])
    assert type(model) is SupportBCModelConfig
    assert model.adapt_config().write_objective == 'action_bc'
    assert model.inner_steps == 3 and not model.first_order
    assert backend.method_name(model) == 'support_bc'
    assert backend.method_name(replace(model, first_order=True)) == 'support_bc_first_order'
    assert backend._backend(model).__name__.endswith('.support_bc_models')
    sources = backend.numerical_sources(model)
    assert {'support_bc_models.py', 'kvb_models.py', 'supervised_models.py',
            'fast_weight_ttt.py', 'policy_backend.py'} == set(sources)
    assert len(set(training.CHECKPOINT_TYPES.values())) == 3
    assert training.CHECKPOINT_TYPES['support_bc'] == training.SUPPORT_BC_CHECKPOINT_TYPE
    assert cfg['output_dir'].endswith('/support_bc_transformer_v1')
    kvb = training.resolve_config({**cfg, 'method': 'kvb'})
    assert training._scientific_config(cfg) != training._scientific_config(kvb)


@pytest.mark.parametrize('method', ['icil', 'kvb'])
@pytest.mark.parametrize('pre_holdout', [False, True])
def test_existing_policy_source_upgrade_preserves_every_other_execution_constraint(method, pre_holdout):
    model = backend.model_config({}, method)
    current = training._execution_signature(model)
    sources = {**current['source_hashes'],
               'policy_backend.py': training.PRE_SUPPORT_BC_BACKEND_SHA256,
               'supervised_train.py': (training.PRE_CLASS_HOLDOUT_TRAINER_SHA256 if pre_holdout
                                       else training.PRE_SUPPORT_BC_TRAINER_SHA256)}
    if pre_holdout:
        sources.pop('class_split.py')
    previous = {**current, 'source_hashes': sources}
    assert training._compatible_execution(previous, current)
    assert previous['source_hashes'] == sources  # Do not mutate checkpoint provenance.
    for name in sources:
        changed = {**previous, 'source_hashes': {**sources, name: 'unreviewed'}}
        assert not training._compatible_execution(changed, current)
    for name, value in [('backend', 'different'), ('versions', {}), ('device_kinds', ['different'])]:
        assert not training._compatible_execution({**previous, name: value}, current)
    bc_execution = training._execution_signature(backend.model_config({}, 'support_bc'))
    assert not training._compatible_execution(previous, bc_execution)


def test_support_bc_resume_requires_unchanged_numerics():
    execution = training._execution_signature(backend.model_config({}, 'support_bc'))
    assert training._compatible_execution(execution, execution)
    for name in execution['source_hashes']:
        changed = {**execution, 'source_hashes': {**execution['source_hashes'], name: 'unreviewed'}}
        assert not training._compatible_execution(changed, execution)
