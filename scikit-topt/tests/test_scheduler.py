import numpy as np

from sktopt.tools.scheduler import (
    schedule_step,
    schedule_step_accelerating,
    schedule_step_decelerating,
)


def test_step_decelerating_endpoints_and_monotone():
    total = 10
    init = 1.0
    target = 5.0
    # sample the center of each step to check discrete values
    sample_iters = [1, 3, 5, 7, 9]
    values = [
        schedule_step_decelerating(
            it=i,
            total=total,
            initial_value=init,
            target_value=target,
            num_steps=5,
            curvature=2.0,
        )
        for i in sample_iters
    ]

    assert np.isclose(values[0], init)
    assert np.isclose(values[-1], target)
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_decelerating_is_faster_early_and_slower_late_than_baselines():
    total = 10
    init = 1.0
    target = 5.0
    kwargs = dict(total=total, initial_value=init, target_value=target, num_steps=5, curvature=2.0)

    # Early stage should jump faster than linear and accelerating
    dec_early = schedule_step_decelerating(it=3, **kwargs)
    lin_early = schedule_step(it=3, **kwargs)
    acc_early = schedule_step_accelerating(it=3, **kwargs)
    assert acc_early < lin_early < dec_early

    # Late stage should already be closer to target than the others
    dec_late = schedule_step_decelerating(it=7, **kwargs)
    lin_late = schedule_step(it=7, **kwargs)
    acc_late = schedule_step_accelerating(it=7, **kwargs)
    assert acc_late < lin_late < dec_late


def test_curvature_one_matches_linear_step():
    total = 10
    init = 0.5
    target = 1.5
    kwargs = dict(total=total, initial_value=init, target_value=target, num_steps=4, curvature=1.0)
    sample_iters = [1, 4, 7, 9]

    linear = np.array([schedule_step(it=i, **kwargs) for i in sample_iters])
    decel = np.array([schedule_step_decelerating(it=i, **kwargs) for i in sample_iters])
    accel = np.array([schedule_step_accelerating(it=i, **kwargs) for i in sample_iters])

    np.testing.assert_allclose(linear, decel)
    np.testing.assert_allclose(linear, accel)
