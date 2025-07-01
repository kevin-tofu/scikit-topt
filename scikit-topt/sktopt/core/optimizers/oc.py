from typing import Literal
from dataclasses import dataclass
import numpy as np
import sktopt
from sktopt.core import projection
from sktopt.core import misc
from sktopt.core.optimizers import common
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


@dataclass
class OC_Config(common.DensityMethodLagrangianConfig):
    interpolation: Literal["SIMP"] = "SIMP"
    eta_init: float = 0.1
    eta: float = 0.5
    eta_step: int = 3


def bisection_with_projection(
    dC, rho_e, rho_min, rho_max, move_limit,
    eta, eps, vol_frac,
    beta, beta_eta,
    scaling_rate, rho_candidate,
    tmp_lower, tmp_upper,
    elements_volume, elements_volume_sum,
    max_iter: int = 100,
    tolerance: float = 1e-4,
    l1: float = 1e-7,
    l2: float = 1e+7
):
    # for _ in range(100):
    # while abs(l2 - l1) <= tolerance * (l1 + l2) / 2.0:
    # while abs(l2 - l1) > tolerance * (l1 + l2) / 2.0:
    while abs(l2 - l1) > tolerance:
        lmid = 0.5 * (l1 + l2)
        np.negative(dC, out=scaling_rate)
        scaling_rate /= (lmid + eps)
        np.power(scaling_rate, eta, out=scaling_rate)

        # Clip
        np.clip(scaling_rate, 0.8, 1.2, out=scaling_rate)
        np.multiply(rho_e, scaling_rate, out=rho_candidate)
        np.maximum(rho_e - move_limit, rho_min, out=tmp_lower)
        np.minimum(rho_e + move_limit, rho_max, out=tmp_upper)
        np.clip(rho_candidate, tmp_lower, tmp_upper, out=rho_candidate)

        projection.heaviside_projection_inplace(
            rho_candidate, beta=beta, eta=beta_eta, out=rho_candidate
        )

        # vol_error = np.mean(rho_candidate) - vol_frac
        vol_error = np.sum(
            rho_candidate * elements_volume
        ) / elements_volume_sum - vol_frac

        if abs(vol_error) < 1e-6:
            break
        if vol_error > 0:
            l1 = lmid
        else:
            l2 = lmid

    return lmid, vol_error


class OC_Optimizer(common.DensityMethod):
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

    mesh, basis, etc. : inherited from common.DensityMethod
        FEM components required for simulation, including boundary conditions and loads.

    """

    def __init__(
        self,
        cfg: OC_Config,
        tsk: sktopt.mesh.TaskConfig,
    ):
        assert cfg.lambda_lower < cfg.lambda_upper
        super().__init__(cfg, tsk)
        self.recorder = self.add_recorder(tsk)
        self.recorder.add("-dC", plot_type="min-max-mean-std", ylog=True)
        self.recorder.add("lmid", ylog=True)
        self.running_scale = 0

    def init_schedulers(self, export: bool = True):
        super().init_schedulers(False)
        self.schedulers.add(
            "eta",
            self.cfg.eta_init,
            self.cfg.eta,
            self.cfg.eta_step,
            self.cfg.max_iters
        )
        if export:
            self.schedulers.export()

    def rho_update(
        self,
        iter_loop: int,
        rho_candidate: np.ndarray,
        rho_projected: np.ndarray,
        dC_drho_ave: np.ndarray,
        u_dofs: np.ndarray,
        strain_energy_ave: np.ndarray,
        scaling_rate: np.ndarray,
        move_limit: float,
        eta: float,
        beta: float,
        tmp_lower: np.ndarray,
        tmp_upper: np.ndarray,
        percentile: float,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        vol_frac: float
    ):
        cfg = self.cfg
        # tsk = self.tsk
        eps = 1e-6
        if percentile > 0:
            scale = np.percentile(np.abs(dC_drho_ave), percentile)
            # scale = max(scale, np.mean(np.abs(dC_drho_ave)), 1e-4)
            # scale = np.median(np.abs(dC_drho_full[tsk.design_elements]))
            self.running_scale = 0.6 * self.running_scale + \
                (1 - 0.6) * scale if iter_loop > 1 else scale
            dC_drho_ave /= (self.running_scale + eps)
        else:
            pass

        # rho_e = rho_projected[tsk.design_elements]
        # rho_e = rho[tsk.design_elements]
        rho_e = rho_candidate.copy()

        lmid, vol_error = bisection_with_projection(
            dC_drho_ave,
            rho_e, cfg.rho_min, cfg.rho_max, move_limit,
            eta, eps, vol_frac,
            beta, cfg.beta_eta,
            scaling_rate, rho_candidate,
            tmp_lower, tmp_upper,
            elements_volume_design, elements_volume_design_sum,
            max_iter=1000, tolerance=1e-5,
            l1=cfg.lambda_lower,
            l2=cfg.lambda_upper
        )
        l_str = f"λ: {lmid:.4e}"
        vol_str = f"vol_error: {vol_error:.4f}"
        rho_str = f"mean(rho): {np.mean(rho_candidate):.4f}"
        message = f"{l_str}, {vol_str}, {rho_str}"

        logger.info(message)
        self.recorder.feed_data("lmid", lmid)
        self.recorder.feed_data("vol_error", vol_error)
        self.recorder.feed_data("-dC", -dC_drho_ave)


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
    cfg = OC_Config.from_defaults(
        **vars(args)
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
