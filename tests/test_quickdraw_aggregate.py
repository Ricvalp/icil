import copy

import pytest

from icil_jax_rlbench.quickdraw.aggregate import paired_control_summary
from icil_jax_rlbench.quickdraw.metrics import timed_tracking_metrics


def _record(program, sample, error, complete=True):
    return dict(task_index=program, sample_index=sample, query_index=0,
                intended_category='fixture', intended_base_id=str(program),
                query_seed=[0,sample],frame=[0,0,0,1],
                timed_tracking=dict(raw_timestep_xy_rmse=error,
                    complete_output=complete,no_stop_failure=not complete,
                    invalid_pair_count=0))


def test_paired_aggregation_uses_program_units_and_retains_missing_outputs():
    left = [_record(0,i,3.) for i in range(8)]+[_record(1,0,None,False)]
    right = [_record(0,i,1.) for i in range(8)]+[_record(1,0,1.)]
    result = paired_control_summary(left,right,experiment='b2')
    assert result['programs'] == 2
    assert result['error_difference']['independent_tasks'] == 1
    assert result['error_difference']['mean'] == 2.
    assert result['failure_fraction_difference']['mean'] == .5
    assert result['paired_evaluable_samples'] == 8
    altered = copy.deepcopy(right)
    altered[0]['query_seed'] = [9,9]
    with pytest.raises(ValueError,match='identical'):
        paired_control_summary(left,altered,experiment='b2')


def test_b1_aggregation_rejects_wrong_order_with_identical_endpoint_and_no_stop():
    reference = [[0.,0.,0.],[.2,.4,1.],[.8,.4,1.],[1.,0.,1.]]
    reversed_middle = [reference[0],reference[2],reference[1],reference[3]]
    left = _record(0,0,0.)
    right = copy.deepcopy(left)
    left['ordered_tracking'] = timed_tracking_metrics(reversed_middle,reference,stopped=False)
    right['ordered_tracking'] = timed_tracking_metrics(reference,reference,stopped=True)
    result = paired_control_summary([left],[right],experiment='b1')
    assert result['error_difference']['mean'] > .1
    assert result['failure_fraction_difference']['mean'] == 1.
