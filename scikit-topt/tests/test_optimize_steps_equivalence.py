import os
import tempfile

import numpy as np

import sktopt
from sktopt.core import misc


def _latest_rho(dst_path: str) -> np.ndarray:
    """Load the most recent rho snapshot from the optimizer output."""
    _, data_path = misc.find_latest_iter_file(f"{dst_path}/data")
    data = np.load(data_path)
    return data["rho_design_elements"]


def _configure_oc(dst_path: str, max_iters: int):
    cfg = sktopt.core.optimizers.OC_Config()
    cfg.max_iters = max_iters
    cfg.record_times = max_iters
    cfg.dst_path = dst_path
    return cfg


def test_optimize_full_vs_steps_are_equal():
    tsk = sktopt.mesh.toy_problem.toy_test()
    max_iters = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        # Full run
        cfg_full = _configure_oc(os.path.join(tmpdir, "full"), max_iters)
        opt_full = sktopt.core.OC_Optimizer(cfg_full, tsk)
        opt_full.parameterize()
        opt_full.optimize()
        rho_full = _latest_rho(cfg_full.dst_path)
        hist_full = opt_full.recorder.as_object()

        # Stepwise run (1 iteration at a time)
        cfg_step = _configure_oc(os.path.join(tmpdir, "steps"), max_iters)
        opt_step = sktopt.core.OC_Optimizer(
            cfg_step, sktopt.mesh.toy_problem.toy_test()
        )
        opt_step.parameterize()
        for _ in range(max_iters):
            opt_step.optimize_steps(1)
        rho_step = _latest_rho(cfg_step.dst_path)
        hist_step = opt_step.recorder.as_object()

    np.testing.assert_allclose(rho_full, rho_step, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.array(hist_full.compliance), np.array(hist_step.compliance)
    )
