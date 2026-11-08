import os
import tempfile

import numpy as np

import sktopt


def _shared_config(dst_path):
    schedule = sktopt.tools.SchedulerConfig
    return {
        "dst_path": dst_path,
        # Compare the objective after one completed density update.  The
        # first recorded compliance is evaluated before the first update.
        "max_iters": 2,
        "record_times": 1,
        "p": schedule.constant(target_value=3.0),
        "vol_frac": schedule.constant(target_value=0.8),
        "beta": schedule.constant(target_value=1.0),
        "move_limit": schedule.constant(target_value=0.2),
        "filter_radius": schedule.constant(target_value=0.1),
    }


def test_mma_one_step_is_comparable_to_oc():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_oc = sktopt.mesh.toy_problem.toy_test()
        oc_cfg = sktopt.core.OC_Config(**_shared_config(
            os.path.join(tmpdir, "oc")
        ))
        oc = sktopt.core.OC_Optimizer(oc_cfg, task_oc)
        oc.parameterize()
        oc.optimize()

        task_mma = sktopt.mesh.toy_problem.toy_test()
        mma_cfg = sktopt.core.MMA_Config(
            **_shared_config(os.path.join(tmpdir, "mma")),
            mma_maxeval=40,
        )
        mma = sktopt.core.MMA_Optimizer(mma_cfg, task_mma)
        mma.parameterize()
        mma.optimize()

        oc_result = oc.recorder.as_object_latest()
        mma_result = mma.recorder.as_object_latest()
        assert np.isfinite(oc_result.compliance)
        assert np.isfinite(mma_result.compliance)
        assert oc_result.compliance > 0.0
        assert mma_result.compliance > 0.0
        assert abs(mma_result.vol_error) < 1e-3
        assert mma_result.rho_change_max > 1e-6
        assert max(oc_result.compliance, mma_result.compliance) / min(
            oc_result.compliance, mma_result.compliance
        ) < 1.2
