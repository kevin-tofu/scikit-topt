from typing import Literal
from dataclasses import dataclass, field
import numpy as np
import sktopt
from sktopt.core import projection
from sktopt.core import misc
from sktopt.core.optimizers import common_density
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


@dataclass
class OC_Config(common_density.DensityMethod_OC_Config):
    """
    Optimality-Criteria (OC) configuration for density-based optimization.

    This class specializes :class:`common_density.DensityMethod_OC_Config`
    for OC-style updates. The material interpolation is fixed to SIMP, and
    the projection threshold ``eta`` is controlled via a
    :class:`sktopt.tools.SchedulerConfig`.

    Attributes
    ----------
    interpolation : {"SIMP"}
        Material interpolation model. Fixed to "SIMP" for OC.
    eta : sktopt.tools.SchedulerConfig
        Threshold (η) used in Heaviside projection / update logic.
        The default keeps η constant at 0.5 using a ``Constant`` scheduler.
        Pass a custom scheduler to run continuation (e.g., Step from 0.7 → 0.3).

    Notes
    -----
    - Other continuation parameters (e.g., ``p``, ``beta``, ``vol_frac``,
      ``filter_radius``) are inherited from the parent config if present there.
    - To override ``eta`` from outside, provide a pre-built
      :class:`SchedulerConfig` at construction time.
    """
    interpolation: Literal["SIMP"] = "SIMP"
    eta: sktopt.tools.SchedulerConfig = field(
        default_factory=lambda: sktopt.tools.SchedulerConfig.constant(
            target_value=0.5
        )
    )


def bisection_with_projection(
    dC, rho_e, rho_min, rho_max, move_limit,
    eta, eps, vol_frac,
    beta, beta_eta,
    scaling_rate, rho_design_eles,
    rho_clip_lower, rho_clip_upper,
    elements_volume, elements_volume_sum,
    max_iter: int = 100,
    tolerance: float = 1e-4,
    vol_tol: float = 1e-4,
    l1: float = 1e-7,
    l2: float = 1e+7
):
    # for _ in range(100):
    # while abs(l2 - l1) <= tolerance * (l1 + l2) / 2.0:
    # while abs(l2 - l1) > tolerance * (l1 + l2) / 2.0:
    iter_num = 0
    while abs(l2 - l1) > tolerance:
        lmid = 0.5 * (l1 + l2)
        # must be dC < 0
        np.negative(dC, out=scaling_rate)
        scaling_rate /= (lmid + eps)
        np.power(scaling_rate, eta, out=scaling_rate)

        # Clip
        np.clip(scaling_rate, 0.8, 1.2, out=scaling_rate)
        np.multiply(rho_e, scaling_rate, out=rho_design_eles)
        np.maximum(rho_e - move_limit, rho_min, out=rho_clip_lower)
        np.minimum(rho_e + move_limit, rho_max, out=rho_clip_upper)
        np.clip(
            rho_design_eles, rho_clip_lower, rho_clip_upper,
            out=rho_design_eles
        )
        projection.heaviside_projection_inplace(
            rho_design_eles, beta=beta, eta=beta_eta, out=rho_design_eles
        )

        # vol_error = np.mean(rho_design_eles) - vol_frac
        vol_error = np.sum(
            rho_design_eles * elements_volume
        ) / elements_volume_sum - vol_frac

        if abs(vol_error) < vol_tol:
            break

        if iter_num >= max_iter:
            break

        iter_num += 1
        if vol_error > 0:
            l1 = lmid
        else:
            l2 = lmid

    return lmid, vol_error


class OC_Optimizer(common_density.DensityMethod):
    """
    Topology optimization solver using the classic Optimality Criteria (OC) method.
    This class implements the standard OC algorithm for compliance minimization problems.
    It uses a multiplicative density update formula derived from Karush-Kuhn-Tucker (KKT)
    optimality conditions under volume constraints.

    The update rule typically takes the form:
        ρ_new = clamp(ρ * sqrt(-dC / λ), ρ_min, ρ_max)
    where:
        - dC is the sensitivity of the compliance objective,
        - λ is a Lagrange multiplier for the volume constraint.

    This method is widely used in structural optimization due to its simplicity,
    interpretability, and solid theoretical foundation.

    Advantages
    ----------
    - Simple and easy to implement
    - Intuitive update rule based on physical insight
    - Well-established and widely validated in literature

    Attributes
    ----------

    config : DensityMethodConfig
        Configuration object specifying the interpolation method, volume fraction,
        continuation settings, filter radius, and other numerical parameters.

    mesh, basis, etc. : inherited from common_density.DensityMethod
        FEM components required for simulation, including boundary conditions and loads.

    """

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
        vol_frac: float
    ):
        cfg = self.cfg
        # tsk = self.tsk
        if self._rho_e_buffer is None:
            self._rho_e_buffer = np.empty_like(rho_design_eles)
            self._dC_raw_buffer = np.empty_like(dC_drho_design_eles)

        # rho_e = self._rho_e_buffer
        # dC_raw = self._dC_raw_buffer

        eps = 1e-6
        if isinstance(percentile, float):
            with self._timed_section("percentile_scale"):
                scale = np.percentile(np.abs(dC_drho_design_eles), percentile)
                # scale = max(scale, np.mean(np.abs(dC_drho_design_eles)), 1e-4)
                # scale = np.median(np.abs(dC_drho_full[tsk.design_elements]))
                self.running_scale = 0.6 * self.running_scale + \
                    (1 - 0.6) * scale if iter_num > 1 else scale
                dC_drho_design_eles /= (self.running_scale + eps)
        else:
            pass

        # store dC_drho_design_eles for kkt residual before scaling
        with self._timed_section("copy_buffers"):
            np.copyto(self._dC_raw_buffer, dC_drho_design_eles)
            np.copyto(self._rho_e_buffer, rho_design_eles)
        # np.copyto(rho_e, rho_projected[tsk.design_elements])
        # rho_e = rho[tsk.design_elements]
        # rho_e = rho_design_eles.copy()

        with self._timed_section("bisection"):
            lmid, vol_error = bisection_with_projection(
                dC_drho_design_eles,
                self._rho_e_buffer, cfg.rho_min, cfg.rho_max, move_limit,
                eta, eps, vol_frac,
                beta, cfg.beta_eta,
                scaling_rate, rho_design_eles,
                rho_clip_lower, rho_clip_upper,
                elements_volume_design, elements_volume_design_sum,
                max_iter=1000, tolerance=1e-5,
                l1=cfg.lambda_lower,
                l2=cfg.lambda_upper
            )

        #
        # compute kkt residual
        #
        with self._timed_section("kkt"):
            mask_int = (
                (rho_design_eles > cfg.rho_min + 1e-6) &
                (rho_design_eles < cfg.rho_max - 1e-6)
            )
            if np.any(mask_int):
                # dL/dρ_i = dC/dρ_i + λ * dV/dρ_i
                # KKT residual
                dL = self._dC_raw_buffer[mask_int] + \
                    lmid * self._dV_drho_design[mask_int]
                self.kkt_residual = float(np.linalg.norm(dL, ord=np.inf))
            else:
                self.kkt_residual = 0.0

        l_str = f"λ: {lmid:.4e}"
        vol_str = f"vol_error: {vol_error:.4f}"
        rho_str = f"mean(rho): {np.mean(rho_design_eles):.4f}"
        kkt_str = f"kkt_res: {self.kkt_residual:.4e}"
        message = f"{l_str}, {vol_str}, {rho_str}, {kkt_str}"

        logger.info(message)
        self.recorder.feed_data("lmid", lmid)
        self.recorder.feed_data("vol_error", vol_error)
        self.recorder.feed_data("-dC", -dC_drho_design_eles)
        self.recorder.feed_data("kkt_residual", self.kkt_residual)


if __name__ == '__main__':

    import argparse
    from sktopt.mesh import toy_problem

    parser = argparse.ArgumentParser(
        description=''
    )
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
    # cfg = OC_Config.from_defaults(
    #     **vars(args)
    # )
    cfg = OC_Config.from_defaults(
        **misc.args2OC_Config_dict(vars(args))
    )
    print("optimizer")
    optimizer = OC_Optimizer(cfg, tsk)
    print("parameterize")
    optimizer.parameterize()
    # optimizer.parameterize(preprocess=False)
    # optimizer.load_parameters()
    print("optimize")
    optimizer.optimize()
    # optimizer.optimize_org()
