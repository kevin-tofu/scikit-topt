from dataclasses import dataclass
import numpy as np
import sktopt
from sktopt.core.optimizers import common
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


@dataclass
class LogLagrangian_Config(common.DensityMethodLagrangianConfig):
    """
    Configuration for Log-space Lagrangian Gradient Update method.

    This class defines the configuration parameters for performing topology optimization
    via gradient-based updates in the logarithmic domain. Unlike traditional Optimality Criteria (OC)
    methods, this approach explicitly follows the gradient of the Lagrangian
    (compliance + volume penalty) and applies the update in log-space to ensure
    positive densities and multiplicative-like behavior.

    Attributes
    ----------
    mu_p : float
        Weighting factor applied to the volume constraint in the Lagrangian.
        This term scales the influence of the volume penalty in the descent direction.

    lambda_v : float
        Initial value for the Lagrange multiplier associated with the volume constraint.
        This is added directly to the compliance gradient to form the full Lagrangian gradient.

    lambda_decay : float
        Decay factor applied to lambda_v over iterations, allowing gradual tuning
        of constraint strength.

    lambda_lower : float
        Minimum allowed value for the Lagrange multiplier. Can be negative in this formulation,
        since lambda_v is added to the gradient rather than used as a ratio.

    lambda_upper : float
        Maximum allowed value for the Lagrange multiplier. Clamping helps avoid instability
        in the update steps due to large penalties.

    Notes
    -----
    This method is sometimes referred to as 'EUMOC', but it is mathematically distinct
    from classical OC-based updates. It performs gradient descent on the Lagrangian
    in log(ρ)-space, leading to multiplicative behavior while maintaining gradient fidelity.

    """
    mu_p: float = 5.0
    lambda_v: float = 0.1
    lambda_decay: float = 0.90
    lambda_lower: float = -1e+7
    lambda_upper: float = 1e+7


# log(x) = -0.4   →   x ≈ 0.670
# log(x) = -0.3   →   x ≈ 0.741
# log(x) = -0.2   →   x ≈ 0.819
# log(x) = -0.1   →   x ≈ 0.905
# log(x) =  0.0   →   x =  1.000
# log(x) = +0.1   →   x ≈ 1.105
# log(x) = +0.2   →   x ≈ 1.221
# log(x) = +0.3   →   x ≈ 1.350
# log(x) = +0.4   →   x ≈ 1.492


def kkt_log_update(
    rho,
    dC, lambda_v, scaling_rate,
    eta, move_limit,
    tmp_lower, tmp_upper,
    rho_min: float, rho_max: float,
    percentile: float,
    interpolation: str
):
    """
    In-place version of the modified OC update (log-space),
    computing dL = dC + lambda_v inside the function.

    Parameters:
    - rho: np.ndarray, design variables (will be updated in-place)
    - dC: np.ndarray, compliance sensitivity (usually negative)
    - lambda_v: float, Lagrange multiplier for volume constraint
    - move: float, maximum allowed change per iteration
    - eta: float, learning rate
    - rho_min: float, minimum density
    - rho_max: float, maximum density
    - tmp_lower, tmp_upper, scaling_rate: work arrays (same shape as rho)
    """

    # eps = 1e-8
    # Compute dL = dC + lambda_v
    # np.copyto(scaling_rate, dC)
    # scaling_rate += lambda_v
    # norm = np.percentile(np.abs(scaling_rate), percentile) + 1e-8
    # scaling_rate /= norm

    # Normalize: subtract mean
    # print(f"interpolation: {interpolation}")
    np.copyto(scaling_rate, dC)
    if percentile > 0:
        if interpolation == "SIMP":
            norm = np.percentile(np.abs(dC), percentile) + 1e-8
            np.divide(scaling_rate, norm, out=scaling_rate)
        elif interpolation == "RAMP":
            scaling_rate -= np.mean(dC)
            percentile_value = np.percentile(np.abs(scaling_rate), percentile)
            # norm = max(percentile_value, 1e-4)
            norm = percentile_value
            # norm = max(np.abs(scaling_rate), 1e-4)
            # print(f"percentile_value: {percentile_value}, norm: {norm}")
            np.divide(scaling_rate, norm, out=scaling_rate)
        else:
            raise ValueError("should be SIMP/RAMP")
    else:
        pass

    clip_range = 1.0
    # np.copyto(scaling_rate, dC)
    # np.clip(scaling_rate, -clip_range, clip_range, out=scaling_rate)
    scaling_rate += lambda_v
    np.clip(scaling_rate, -clip_range, clip_range, out=scaling_rate)

    # Ensure rho is in [rho_min, 1.0] before log
    np.clip(rho, rho_min, 1.0, out=rho)

    # tmp_lower = log(rho)
    np.log(rho, out=tmp_lower)

    # tmp_upper = exp(log(rho)) = rho
    np.exp(tmp_lower, out=tmp_upper)

    # tmp_upper = log(1 + move / rho)
    np.divide(move_limit, tmp_upper, out=tmp_upper)
    np.add(tmp_upper, 1.0, out=tmp_upper)
    np.log(tmp_upper, out=tmp_upper)

    # tmp_lower = lower bound in log-space
    np.subtract(tmp_lower, tmp_upper, out=tmp_lower)

    # tmp_upper = upper bound in log-space
    np.add(tmp_lower, 2.0 * tmp_upper, out=tmp_upper)

    # rho = log(rho)
    np.log(rho, out=rho)

    # Update in log-space
    rho -= eta * scaling_rate

    # Apply move limits
    np.clip(rho, tmp_lower, tmp_upper, out=rho)

    # Convert back to real space
    np.exp(rho, out=rho)

    # Final clipping
    np.clip(rho, rho_min, rho_max, out=rho)


class LogLagrangian_Optimizer(common.DensityMethod):
    """
    Topology optimization solver using log-space Lagrangian gradient descent.

    This optimizer performs sensitivity-based topology optimization by applying
    gradient descent on the Lagrangian (compliance + volume penalty) in log(ρ)-space.
    Unlike traditional Optimality Criteria (OC) methods, this method adds the
    volume constraint term (λ) directly to the compliance gradient before updating.

    By performing updates in log-space, the method ensures strictly positive densities
    and exhibits multiplicative behavior in the original density space, which can enhance
    numerical stability—particularly for problems with low volume fractions or steep gradients.

    In each iteration, the update follows:

        log(ρ_new) = log(ρ) - η · (∂C/∂ρ + λ)

    which is equivalent to:

        ρ_new = ρ · exp( -η · (∂C/∂ρ + λ) )

    Here:
    - ∂C/∂ρ is the compliance sensitivity,
    - λ is the Lagrange multiplier (volume penalty weight),
    - η is a step size parameter.

    Attributes
    ----------
    config : LogGradientUpdateConfig
        Configuration object specifying optimization parameters such as mu_p,
        lambda_v, decay schedules, and filtering strategies.

    mesh, basis, etc. : inherited from common.DensityMethod
        Core finite element components required for stiffness evaluation,
        boundary conditions, and optimization loop execution.

    Notes
    -----
    Although previously referred to as 'EUMOC', this method is not derived
    from the traditional Optimality Criteria framework. Instead, it implements
    log-space gradient descent on the Lagrangian, directly adding the volume
    penalty to the sensitivity.


    """

    def __init__(
        self,
        cfg: LogLagrangian_Config,
        tsk: sktopt.mesh.TaskConfig,
    ):
        assert cfg.lambda_lower < 0
        assert cfg.lambda_upper > 0
        super().__init__(cfg, tsk)
        self.recorder = self.add_recorder(tsk)
        self.recorder.add("dC", plot_type="min-max-mean-std")
        self.recorder.add("lambda_v", ylog=False)
        self.lambda_v = cfg.lambda_v

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
        self.recorder.feed_data("dC", dC_drho_ave)

        kkt_log_update(
            rho=rho_candidate,
            dC=dC_drho_ave,
            lambda_v=self.lambda_v, scaling_rate=scaling_rate,
            move_limit=move_limit,
            eta=eta,
            tmp_lower=tmp_lower, tmp_upper=tmp_upper,
            rho_min=cfg.rho_min, rho_max=1.0,
            percentile=percentile,
            interpolation=cfg.interpolation
        )


if __name__ == '__main__':
    import argparse
    from sktopt.mesh import toy_problem
    from sktopt.core import misc

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
    print("generate LogLagrangian_Config")
    cfg = LogLagrangian_Config.from_defaults(
        **vars(args)
    )

    print("optimizer")
    optimizer = LogLagrangian_Optimizer(cfg, tsk)
    print("parameterize")
    optimizer.parameterize()
    # optimizer.parameterize(preprocess=False)
    # optimizer.load_parameters()
    print("optimize")
    optimizer.optimize()
    # optimizer.optimize_fosm()
