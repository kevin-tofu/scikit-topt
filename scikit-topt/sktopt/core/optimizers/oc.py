import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import sktopt
from sktopt.core import misc
from sktopt.core import projection
from sktopt.core.optimizers import common_density
from sktopt.tools.logconf import mylogger

logger = mylogger(__name__)


def bisection_with_physical_volume(
    dC,
    rho_e,
    rho_full,
    design_elements,
    filter_obj,
    rho_min,
    rho_max,
    move_limit,
    eta,
    eps,
    vol_frac,
    beta,
    beta_eta,
    scaling_rate,
    rho_design_eles,
    rho_clip_lower,
    rho_clip_upper,
    elements_volume,
    elements_volume_sum,
    scaling_rate_min,
    scaling_rate_max,
    rho_full_candidate,
    rho_filtered_candidate,
    rho_projected_candidate,
    max_iter: int = 100,
    tolerance: float = 1e-4,
    vol_tol: float = 1e-4,
    l1: float = 1e-7,
    l2: float = 1e+7,
):
    iter_num = 0
    lmid = 0.5 * (l1 + l2)
    vol_error = 0.0
    while True:
        np.negative(dC, out=scaling_rate)
        scaling_rate /= (lmid + eps)
        np.power(scaling_rate, eta, out=scaling_rate)
        np.clip(
            scaling_rate, scaling_rate_min, scaling_rate_max,
            out=scaling_rate
        )

        np.multiply(rho_e, scaling_rate, out=rho_design_eles)
        np.maximum(rho_e - move_limit, rho_min, out=rho_clip_lower)
        np.minimum(rho_e + move_limit, rho_max, out=rho_clip_upper)
        np.clip(
            rho_design_eles, rho_clip_lower, rho_clip_upper,
            out=rho_design_eles
        )

        np.copyto(rho_full_candidate, rho_full)
        rho_full_candidate[design_elements] = rho_design_eles
        np.copyto(rho_filtered_candidate, filter_obj.forward(rho_full_candidate))
        projection.heaviside_projection_inplace(
            rho_filtered_candidate,
            beta=beta,
            eta=beta_eta,
            out=rho_projected_candidate,
        )

        vol_error = np.sum(
            rho_projected_candidate[design_elements] * elements_volume
        ) / elements_volume_sum - vol_frac

        if abs(vol_error) < vol_tol:
            break
        if iter_num >= max_iter:
            break
        if abs(l2 - l1) <= tolerance:
            break

        if vol_error > 0:
            l1 = lmid
        else:
            l2 = lmid
        iter_num += 1
        lmid = 0.5 * (l1 + l2)

    return lmid, vol_error


@dataclass
class OC_Config(common_density.DensityMethod_OC_Config):
    interpolation: Literal["SIMP"] = "SIMP"
    eta: sktopt.tools.SchedulerConfig = field(
        default_factory=lambda: sktopt.tools.SchedulerConfig.constant(
            target_value=0.5
        )
    )
    scaling_rate_min: float = 0.7
    scaling_rate_max: float = 1.3
    sensitivity_scale_floor: float = 1e-12


class OC_Optimizer(common_density.DensityMethod):
    def __init__(
        self,
        cfg: OC_Config,
        tsk: sktopt.mesh.FEMDomain,
    ):
        assert cfg.lambda_lower < cfg.lambda_upper
        super().__init__(cfg, tsk)
        self.recorder = self.add_recorder(tsk)
        self.recorder.add("-dC", plot_type="min-max-mean-std", ylog=True)
        self.recorder.add("lmid", ylog=True)
        self.running_scale = 0

    def init_schedulers(self, export: bool = True):
        super().init_schedulers(False)
        if export:
            self.schedulers.export()

    @contextmanager
    def _scaled_sensitivity_mode(self):
        previous = os.environ.get("SCITOPT_SENSITIVITY_MODE")
        os.environ["SCITOPT_SENSITIVITY_MODE"] = "current"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("SCITOPT_SENSITIVITY_MODE", None)
            else:
                os.environ["SCITOPT_SENSITIVITY_MODE"] = previous

    def optimize(self):
        with self._scaled_sensitivity_mode():
            super().optimize()

    def optimize_steps(self, num_steps: int):
        with self._scaled_sensitivity_mode():
            super().optimize_steps(num_steps)

    def rho_update(
        self,
        iter_num: int,
        rho_design_eles: np.ndarray,
        rho_projected: np.ndarray,
        dC_drho_design_eles: np.ndarray,
        u_dofs: np.ndarray,
        strain_energy_mean: np.ndarray,
        scaling_rate: np.ndarray,
        move_limit: float,
        eta: float,
        beta: float,
        rho_clip_lower: np.ndarray,
        rho_clip_upper: np.ndarray,
        percentile: float | None,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        vol_frac: float,
    ):
        del rho_projected, u_dofs, strain_energy_mean
        cfg = self.cfg
        tsk = self.tsk
        state = self._state
        if state is None:
            raise RuntimeError("Optimizer state is not initialized.")
        if self._rho_e_buffer is None:
            self._rho_e_buffer = np.empty_like(rho_design_eles)
            self._dC_raw_buffer = np.empty_like(dC_drho_design_eles)
        if not hasattr(self, "_rho_full_candidate"):
            self._rho_full_candidate = np.empty_like(state.rho)
            self._rho_filtered_candidate = np.empty_like(state.rho)
            self._rho_projected_candidate = np.empty_like(state.rho)

        with self._timed_section("copy_buffers"):
            np.copyto(self._dC_raw_buffer, dC_drho_design_eles)
            np.copyto(self._rho_e_buffer, rho_design_eles)

        eps = 1e-12
        with self._timed_section("percentile_scale"):
            if isinstance(percentile, float):
                scale = np.percentile(np.abs(dC_drho_design_eles), percentile)
            else:
                scale = np.max(np.abs(dC_drho_design_eles))
            scale = max(scale, cfg.sensitivity_scale_floor)
            self.running_scale = 0.6 * self.running_scale + \
                (1 - 0.6) * scale if iter_num > 1 else scale
            dC_drho_design_eles /= self.running_scale
            kkt_scale = self.running_scale

        with self._timed_section("bisection"):
            lmid, vol_error = bisection_with_physical_volume(
                dC_drho_design_eles,
                self._rho_e_buffer,
                state.rho,
                tsk.design_elements,
                self.filter,
                cfg.rho_min,
                cfg.rho_max,
                move_limit,
                eta,
                eps,
                vol_frac,
                beta,
                cfg.beta_eta,
                scaling_rate,
                rho_design_eles,
                rho_clip_lower,
                rho_clip_upper,
                elements_volume_design,
                elements_volume_design_sum,
                cfg.scaling_rate_min,
                cfg.scaling_rate_max,
                self._rho_full_candidate,
                self._rho_filtered_candidate,
                self._rho_projected_candidate,
                max_iter=1000,
                tolerance=1e-5,
                l1=cfg.lambda_lower,
                l2=cfg.lambda_upper,
            )

        with self._timed_section("kkt"):
            mask_int = (
                (rho_design_eles > cfg.rho_min + 1e-6) &
                (rho_design_eles < cfg.rho_max - 1e-6)
            )
            if np.any(mask_int):
                dL = self._dC_raw_buffer[mask_int] + \
                    (lmid * kkt_scale) * self._dV_drho_design[mask_int]
                self.kkt_residual = float(np.linalg.norm(dL, ord=np.inf))
            else:
                self.kkt_residual = 0.0

        logger.info(
            f"λ: {lmid:.4e}, vol_error: {vol_error:.4f}, "
            f"mean(rho): {np.mean(rho_design_eles):.4f}, "
            f"kkt_res: {self.kkt_residual:.4e}"
        )
        self.recorder.feed_data("lmid", lmid)
        self.recorder.feed_data("vol_error", vol_error)
        self.recorder.feed_data("-dC", -dC_drho_design_eles)
        self.recorder.feed_data("kkt_residual", self.kkt_residual)


if __name__ == '__main__':
    import argparse
    from sktopt.mesh import toy_problem

    parser = argparse.ArgumentParser(description='')
    parser = misc.add_common_arguments(parser)
    parser.add_argument(
        '--eta_init', '-ETI', type=float, default=0.01, help=''
    )
    parser.add_argument(
        '--eta_step', '-ETR', type=float, default=-1.0, help=''
    )
    args = parser.parse_args()

    if args.task_name == "toy1":
        tsk = toy_problem.toy1()
    elif args.task_name == "toy1_fine":
        tsk = toy_problem.toy1_fine()
    elif args.task_name == "toy2":
        tsk = toy_problem.toy2()
    else:
        tsk = toy_problem.toy_msh(args.task_name, args.mesh_path)

    print("load toy problem")
    print("generate OC_Config")
    cfg = OC_Config.from_defaults(
        **misc.args2OC_Config_dict(vars(args))
    )
    print("optimizer")
    optimizer = OC_Optimizer(cfg, tsk)
    print("parameterize")
    optimizer.parameterize()
    print("optimize")
    optimizer.optimize()
