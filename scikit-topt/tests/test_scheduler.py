from pathlib import Path

import numpy as np

from sktopt.tools.scheduler import (
    schedule_step,
    schedule_step_accelerating,
    schedule_step_decelerating,
    Scheduler,
    SchedulerConfig,
    SchedulerStepAcceleratingToOne,
    SchedulerStepDeceleratingToOne,
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


def test_step_to_one_autosets_init_and_target():
    cfg = SchedulerConfig.step_to_one(name="x", num_steps=4, iters_max=8)
    sched = Scheduler.from_config(cfg)

    assert np.isclose(cfg.target_value, 1.0)
    assert np.isclose(cfg.init_value, 0.25)
    assert np.isclose(sched.value(1), 0.25)
    assert np.isclose(sched.value(8), 1.0)


def test_step_to_one_rejects_non_one_target():
    with np.testing.assert_raises(ValueError):
        SchedulerConfig.from_defaults(
            name="x",
            num_steps=3,
            target_value=0.5,
            scheduler_type="StepToOne",
        )


def test_step_accel_to_one_sets_target_and_init():
    cfg = SchedulerConfig.step_accelerating_to_one(
        name="x", num_steps=5, iters_max=10, curvature=2.0
    )
    sched = Scheduler.from_config(cfg)

    assert np.isclose(cfg.target_value, 1.0)
    assert np.isclose(cfg.init_value, 0.2)
    assert np.isclose(sched.value(1), 0.2)
    assert sched.value(10) <= 1.0


def test_step_decel_to_one_sets_target_and_init():
    cfg = SchedulerConfig.step_decelerating_to_one(
        name="x", num_steps=5, iters_max=10, curvature=2.0
    )
    sched = Scheduler.from_config(cfg)

    assert np.isclose(cfg.target_value, 1.0)
    assert np.isclose(cfg.init_value, 0.2)
    assert np.isclose(sched.value(1), 0.2)
    assert sched.value(10) <= 1.0


# def test_plot_schedules_to_local_png():
#     import matplotlib

#     matplotlib.use("Agg")  # ensure headless-friendly backend
#     import matplotlib.pyplot as plt

#     total = 20
#     num_steps = 5
#     xs = np.arange(1, total + 1)

#     scheds = {
#         "Step": [
#             schedule_step(
#                 it=i,
#                 total=total,
#                 initial_value=0.0,
#                 target_value=1.0,
#                 num_steps=num_steps,
#             )
#             for i in xs
#         ],
#         "Accelerating": [
#             schedule_step_accelerating(
#                 it=i,
#                 total=total,
#                 initial_value=0.0,
#                 target_value=1.0,
#                 num_steps=num_steps,
#                 curvature=2.0,
#             )
#             for i in xs
#         ],
#         "Decelerating": [
#             schedule_step_decelerating(
#                 it=i,
#                 total=total,
#                 initial_value=0.0,
#                 target_value=1.0,
#                 num_steps=num_steps,
#                 curvature=2.0,
#             )
#             for i in xs
#         ],
#     }

#     # Include the ToOne schedulers via the factory to mirror real usage.
#     sto_cfg = SchedulerConfig.step_to_one(name="sto", num_steps=num_steps, iters_max=total)
#     sto_sched = Scheduler.from_config(sto_cfg)
#     scheds["StepToOne"] = [sto_sched.value(i) for i in xs]

#     sto_acc_cfg = SchedulerConfig.step_accelerating_to_one(
#         name="sto_acc", num_steps=num_steps, iters_max=total, curvature=2.0
#     )
#     sto_acc_sched = Scheduler.from_config(sto_acc_cfg)
#     scheds["StepAccToOne"] = [sto_acc_sched.value(i) for i in xs]

#     sto_dec_cfg = SchedulerConfig.step_decelerating_to_one(
#         name="sto_dec", num_steps=num_steps, iters_max=total, curvature=2.0
#     )
#     sto_dec_sched = Scheduler.from_config(sto_dec_cfg)
#     scheds["StepDecToOne"] = [sto_dec_sched.value(i) for i in xs]

#     n_plots = len(scheds)
#     cols = 3
#     rows = int(np.ceil(n_plots / cols))
#     fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
#     axes = np.atleast_1d(axes).ravel()
#     for ax, (name, ys) in zip(axes, scheds.items()):
#         ax.plot(xs, ys, marker="o", linestyle="-")
#         ax.set_title(name)
#         ax.set_xlabel("Iteration")
#         ax.set_ylabel("Value")
#         ax.grid(True)

#     # Hide any unused axes slots.
#     for ax in axes[len(scheds) :]:
#         ax.axis("off")

#     outfile = Path(__file__).parent / "scheduler_plots.png"
#     fig.tight_layout()
#     fig.savefig(outfile)
#     plt.close(fig)

#     assert outfile.exists()
