"""Run one MMA optimization with the selected FEM backend."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback


def run_case(args: argparse.Namespace) -> dict:
    # The backend is read at sktopt import time. compare.py sets this in the
    # child process environment before this module imports sktopt.
    import numpy as np
    import sktopt
    from sktopt.core import MMA_Config, MMA_Optimizer
    from sktopt.mesh.toy_problem import toy_base

    backend = sktopt.skfem_loader.backend
    update_timing = {"seconds": 0.0, "calls": 0}
    original_rho_update = MMA_Optimizer.rho_update

    def timed_rho_update(self, *call_args, **call_kwargs):
        start = time.perf_counter()
        try:
            return original_rho_update(self, *call_args, **call_kwargs)
        finally:
            update_timing["seconds"] += time.perf_counter() - start
            update_timing["calls"] += 1

    MMA_Optimizer.rho_update = timed_rho_update

    setup_start = time.perf_counter()
    task = toy_base(args.mesh_size)
    with tempfile.TemporaryDirectory(prefix=f"mma-{backend}-") as dst_path:
        cfg = MMA_Config(
            dst_path=dst_path,
            max_iters=args.iterations,
            record_times=args.iterations,
            p=sktopt.tools.SchedulerConfig.constant(target_value=3.0),
            vol_frac=sktopt.tools.SchedulerConfig.constant(
                target_value=args.volume_fraction
            ),
            beta=sktopt.tools.SchedulerConfig.constant(target_value=1.0),
            move_limit=sktopt.tools.SchedulerConfig.constant(
                target_value=args.move_limit
            ),
            filter_radius=sktopt.tools.SchedulerConfig.constant(
                target_value=args.filter_radius
            ),
            export_img=False,
        )
        optimizer = MMA_Optimizer(cfg, task)
        optimizer.parameterize()
        setup_seconds = time.perf_counter() - setup_start

        optimization_start = time.perf_counter()
        optimizer.optimize()
        optimization_seconds = time.perf_counter() - optimization_start

        final = optimizer.recorder.as_object_latest()
        density = optimizer._state.rho[optimizer.tsk.design_elements]
        result = {
            "status": "success",
            "backend": backend,
            "requested_iterations": args.iterations,
            "mesh_size": args.mesh_size,
            "elements": int(task.basis.mesh.nelements),
            "dofs": int(task.basis.N),
            "setup_seconds": setup_seconds,
            "optimization_seconds": optimization_seconds,
            "mma_adapter_update_seconds": update_timing["seconds"],
            "mma_adapter_update_calls": update_timing["calls"],
            "compliance": float(final.compliance),
            "volume_error": float(final.vol_error),
            "kkt_residual": float(optimizer.kkt_residual),
            "density_min": float(np.min(density)),
            "density_max": float(np.max(density)),
        }
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-size", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--volume-fraction", type=float, required=True)
    parser.add_argument("--move-limit", type=float, required=True)
    parser.add_argument("--filter-radius", type=float, required=True)
    args = parser.parse_args()

    try:
        result = run_case(args)
        return_code = 0
    except Exception as exc:  # Preserve the exception as benchmark data.
        result = {
            "status": "failed",
            "backend": os.environ.get("SKTOPT_FEM_BACKEND", "skfem"),
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        return_code = 1

    print("MMA_BACKEND_RESULT " + json.dumps(result, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
