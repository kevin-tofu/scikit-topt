from dataclasses import dataclass
import numpy as np
import scitopt
from scitopt.core.optimizers import common
from scitopt.tools.logconf import mylogger
logger = mylogger(__name__)


@dataclass
class EUMOC_Config(common.SensitivityConfig):
    """
    Configuration for Exponential Update Modified Optimality Criteria (EUMOC) method.

    This class extends SensitivityConfig and adds specific parameters for performing
    topology optimization in the exponential (log-domain) space using a variant of
    the MOC method.

    Attributes
    ----------
    mu_p : float
        Penalization weight for the volume constraint term, used in \
            the Pseudo-Inverse
        formulation (PIV) of the MOC update rule. Higher values enforce \
            the volume
        constraint more strictly.

    lambda_v : float
        Initial value for the Lagrange multiplier controlling the volume \
            constraint.
        Used in the exponentiated update equation.

    lambda_decay : float
        Exponential decay weight applied to the Lagrange multiplier between \
            iterations.
        Allows smoothing and gradual adaptation of the volume constraint \
            influence.

    lambda_lower : float
        Lower bound for the Lagrange multiplier in the exponential domain.
        Note that this can be negative, unlike in standard OC methods, due \
            to the log-domain
        formulation.

    lambda_upper : float
        Upper bound for the Lagrange multiplier. Used to clamp updates 
        in the exponential MOC framework.

        Differences from LDMOC
    ----------------------
    - Domain:
        EUMOC operates in the **exponential/log-domain**, applying \
            multiplicative updates of the form:
        ρ_new = ρ * exp( - scaling_rate / λ ), which offers smoother control \
            in ill-conditioned problems.
        In contrast, LDMOC uses additive updates in the linear domain.

    - λ behavior:
        EUMOC allows **negative values** for λ, which are valid in the \
            exponential framework.
        This flexibility enables more robust control over the update direction \
            and magnitude.

    - Numerical stability:
        The exponential formulation makes EUMOC more stable in the presence of
        high stiffness contrasts or extreme sensitivity variations.

    - Interpretation:
        Although EUMOC can be more stable, its update rule is \
            **less intuitive** than LDMOC's
        arithmetic update, and the effect of λ and μ_p is more nonlinear.

    - Applicability:
        EUMOC is particularly useful in **advanced or unstable optimization \
            scenarios**, such as:
        - very low volume fractions
        - highly heterogeneous designs
        - stress- or constraint-driven problems

    For simpler or educational use cases, LDMOC may be easier to interpret and \
        tune.
    """
    mu_p: float = 300.0
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


def kkt_moc_log_update(
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


# Exponential Update MOC
class EUMOC_Optimizer(common.SensitivityAnalysis):
    """
    Topology optimization solver using the Exponential Update Modified \
        Optimality Criteria (EUMOC) method.

    This optimizer performs sensitivity-based topology optimization \
        in the log-domain
    using a multiplicative update rule. By leveraging exponential updates and \
        a decaying
    Lagrange multiplier, EUMOC provides enhanced numerical stability and \
        robustness,
    particularly for problems involving low volume fractions or high \
        sensitivity gradients.

    In each iteration, the density update takes the form:
        ρ_new = ρ * exp( - dC / λ )
    where dC is the sensitivity of compliance and λ is the Lagrange multiplier.

    This method is well-suited for advanced or ill-conditioned optimization \
        problems.

    Attributes
    ----------
    config : EUMOC_Config
        Configuration object specifying parameters such as mu_p, lambda_v,
        continuation schedules, and filtering options.

    mesh, basis, etc. : inherited from common.SensitivityAnalysis
        Core FEM components required for analysis, boundary conditions, and \
            optimization loops.

    Advantages
    ----------
    - High numerical stability for ill-conditioned problems
    - Better handling of extreme sensitivities
    - Suitable for low volume fraction designs

    Limitations
    ----------
    - Less intuitive than traditional OC methods
    - Requires care when tuning λ and μ_p
    """

    def __init__(
        self,
        cfg: EUMOC_Config,
        tsk: scitopt.mesh.TaskConfig,
    ):
        assert cfg.lambda_lower < 0
        assert cfg.lambda_upper > 0
        super().__init__(cfg, tsk)
        self.recorder.add("dC", plot_type="min-max-mean-std")
        self.recorder.add("lambda_v", ylog=False)
        self.lambda_v = cfg.lambda_v

    def rho_update(
        self,
        iter_loop: int,
        rho_candidate: np.ndarray,
        rho_projected: np.ndarray,
        dC_drho_ave: np.ndarray,
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

        kkt_moc_log_update(
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
    from scitopt.mesh import toy_problem
    from scitopt.core import misc

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
    print("generate EUMOC_Config")
    cfg = EUMOC_Config.from_defaults(
        **vars(args)
    )

    print("optimizer")
    optimizer = EUMOC_Optimizer(cfg, tsk)
    print("parameterize")
    optimizer.parameterize()
    # optimizer.parameterize(preprocess=False)
    # optimizer.load_parameters()
    print("optimize")
    optimizer.optimize()
    # optimizer.optimize_fosm()
