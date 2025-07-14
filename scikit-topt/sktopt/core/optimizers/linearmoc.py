from typing import Literal
from dataclasses import dataclass
import numpy as np
import sktopt
from sktopt.core import misc
from sktopt.core.optimizers import common
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


@dataclass
class LinearMOC_Config(common.DensityMethodLagrangianConfig):
    """
    Configuration for log-space gradient update with optional mean-centering.

    This configuration supports a variant of log-space Lagrangian descent used in topology
    optimization. The update is performed in the logarithmic domain to ensure strictly positive
    densities and to simulate multiplicative behavior. Optionally, the computed update direction
    can be centered (mean-subtracted) to improve numerical stability.

    This method differs from standard OC-based updates: it adds the volume constraint penalty
    directly to the compliance sensitivity and applies the gradient step in log(ρ)-space:

        log(ρ_new) = log(ρ) - η · (∂C/∂ρ + λ)

    Optionally (when enabled), the update direction is centered as:

        Δ = (∂C/∂ρ + λ) - mean(∂C/∂ρ + λ)

    Attributes
    ----------
    interpolation : Literal["SIMP"]
        Interpolation scheme used for material penalization. Currently only "SIMP" is supported.

    mu_p : float
        Penalty weight for the volume constraint. This controls how strongly the constraint
        influences the descent direction.

    lambda_v : float
        Initial Lagrange multiplier for the volume constraint.

    lambda_decay : float
        Factor by which lambda_v decays over iterations. Smaller values cause λ to adapt more rapidly.

    """

    interpolation: Literal["SIMP"] = "SIMP"
    mu_p: float = 300.0
    lambda_v: float = 0.1
    lambda_decay: float = 0.90


def moc_linear_update_logspace(
    rho,
    dC, lambda_v, scaling_rate,
    eta, move_limit,
    tmp_lower, tmp_upper,
    rho_min, rho_max
):
    eps = 1e-8
    logger.info(f"dC: {dC.min()} {dC.max()}")
    np.negative(dC, out=scaling_rate)
    scaling_rate /= (lambda_v + eps)
    np.maximum(scaling_rate, eps, out=scaling_rate)
    np.log(scaling_rate, out=scaling_rate)
    np.clip(scaling_rate, -0.50, 0.50, out=scaling_rate)
    np.clip(rho, rho_min, 1.0, out=rho)
    np.log(rho, out=tmp_lower)

    # tmp_upper = exp(tmp_lower) = rho (real space)
    np.exp(tmp_lower, out=tmp_upper)
    # tmp_upper = log(1 + move_limit / rho)
    np.divide(move_limit, tmp_upper, out=tmp_upper)
    np.add(tmp_upper, 1.0, out=tmp_upper)
    np.log(tmp_upper, out=tmp_upper)

    # tmp_lower = lower bound = log(rho) - log_move_limit
    np.subtract(tmp_lower, tmp_upper, out=tmp_lower)

    # tmp_upper = upper bound = log(rho) + log_move_limit
    np.add(tmp_lower, 2 * tmp_upper, out=tmp_upper)

    # rho = log(rho)
    np.log(rho, out=rho)

    # log(rho) += η * scaling_rate
    rho += eta * scaling_rate

    # clip in log-space
    np.clip(rho, tmp_lower, tmp_upper, out=rho)

    # back to real space
    np.exp(rho, out=rho)
    np.clip(rho, rho_min, rho_max, out=rho)


# Lagrangian Dual MOC
class LinearMOC_Optimizer(common.DensityMethod):
    """
    Topology optimization solver using linear-space Lagrangian gradient descent.

    This optimizer performs sensitivity-based topology optimization by applying
    gradient descent on the Lagrangian (compliance + volume penalty) in the
    logarithmic domain. The update is computed in log(ρ)-space to ensure positive
    densities and simulate multiplicative behavior.

    Unlike traditional Optimality Criteria (OC) methods, this approach adds the
    volume constraint penalty λ directly to the compliance sensitivity, rather than
    forming a multiplicative ratio. The update is then applied as:

        log(ρ_new) = log(ρ) - η · (∂C/∂ρ + λ)

    which is equivalent to:

        ρ_new = ρ · exp( - η · (∂C/∂ρ + λ) )

    This method maintains positivity of the density field without explicit clipping
    and offers improved numerical robustness for problems involving low volume fractions
    or sharp sensitivity gradients.

    Advantages
    ----------
    - Ensures positive densities via log-space formulation
    - Simulates OC-like multiplicative update behavior
    - Straightforward gradient descent formulation

    Limitations
    -----------
    - Not derived from strict OC/KKT conditions
    - May still require step size tuning and clipping for stability
    - Involves logarithmic and exponential operations at each iteration

    Attributes
    ----------
    config : LogGradientUpdateConfig
        Configuration object specifying parameters such as mu_p, lambda_v,
        decay schedules, and interpolation settings (currently SIMP only).

    mesh, basis, etc. : inherited from common.DensityMethod
        FEM components used to evaluate sensitivities and apply boundary conditions.
    """
    def __init__(
        self,
        cfg: LinearMOC_Config,
        tsk: sktopt.mesh.TaskConfig,
    ):
        assert cfg.lambda_lower < cfg.lambda_upper
        super().__init__(cfg, tsk)
        self.recorder = self.add_recorder(tsk)
        self.recorder.add("-dC", plot_type="min-max-mean-std", ylog=True)
        self.recorder.add("lambda_v", ylog=True)

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
        tsk = self.tsk
        eps = 1e-8
        if percentile > 0:
            scale = np.percentile(np.abs(dC_drho_ave), percentile)
            self.recorder.feed_data("-dC", -dC_drho_ave)
            self.running_scale = 0.2 * self.running_scale + \
                (1 - 0.2) * scale if iter_loop > 1 else scale
            logger.info(f"running_scale: {self.running_scale}")
            dC_drho_ave = dC_drho_ave / (self.running_scale + eps)
        else:
            pass

        vol_error = np.sum(
            rho_projected[tsk.design_elements] * elements_volume_design
        ) / elements_volume_design_sum - vol_frac

        penalty = cfg.mu_p * vol_error
        self.lambda_v = cfg.lambda_decay * self.lambda_v + \
            penalty if iter_loop > 1 else penalty
        self.lambda_v = np.clip(
            self.lambda_v, cfg.lambda_lower, cfg.lambda_upper
        )
        self.recorder.feed_data("lambda_v", self.lambda_v)
        self.recorder.feed_data("vol_error", vol_error)
        self.recorder.feed_data("-dC", -dC_drho_ave)

        moc_linear_update_logspace(
            rho_candidate,
            dC_drho_ave,
            self.lambda_v, scaling_rate,
            move_limit,
            eta,
            tmp_lower, tmp_upper,
            cfg.rho_min, 1.0,
        )


if __name__ == '__main__':
    import argparse
    from sktopt.mesh import toy_problem

    parser = argparse.ArgumentParser(
        description=''
    )
    parser = misc.add_common_arguments(parser)
    parser.add_argument(
        '--mu_p', '-MUP', type=float, default=100.0, help=''
    )
    parser.add_argument(
        '--lambda_v', '-LV', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--lambda_decay', '-LD', type=float, default=0.95, help=''
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
    print("generate LogMOC_Config")
    cfg = LogMOC_Config.from_defaults(
        **vars(args)
    )

    print("optimizer")
    optimizer = LogMOC_Optimizer(cfg, tsk)
    print("parameterize")
    optimizer.parameterize()
    # optimizer.parameterize(preprocess=False)
    # optimizer.load_parameters()
    print("optimize")
    optimizer.optimize()
    # optimizer.optimize_org()
