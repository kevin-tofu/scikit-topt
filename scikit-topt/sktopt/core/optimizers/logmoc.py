from typing import Literal
from dataclasses import dataclass
import numpy as np
import sktopt
from sktopt.core import misc
from sktopt.core.optimizers import common_density
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


@dataclass
class LogMOC_Config(common_density.DensityMethod_OC_Config):
    """
    Configuration for a log-space, OC-like multiplicative density update.

    This configuration supports a variant of the Optimality Criteria (OC)
    method implemented in :math:`\\log \\rho`-space. The update remains
    multiplicative in :math:`\\rho`, but it is applied by adding a scaled
    log-factor to :math:`\\log \\rho`. The dual variable associated with
    the volume constraint is updated by an exponential moving average (EMA)
    driven by the volume error.

    Conceptually, the update has the form

    .. math::

        \\rho^{\\text{new}}
        = \\rho^{\\text{old}}
          \\left(-\\frac{\\partial C / \\partial \\rho}{\\lambda_v}
          \\right)^{\\eta},

    where the ratio :math:`-\\partial C/\\partial \\rho / \\lambda_v`
    is clamped to avoid numerical instabilities and move limits are
    enforced in log-space.

    Attributes
    ----------
    interpolation : Literal["SIMP"]
        Interpolation scheme used for material penalization. Currently
        fixed to ``"SIMP"``.
    mu_p : float
        Gain used to convert the volume error into a dual update. Larger
        values make the dual variable :attr:`lambda_v` react more strongly
        to constraint violations.
    lambda_v : float
        Initial value for the volume-related dual variable. This is used
        in the OC-like scaling term :math:`-\\partial C/\\partial \\rho
        / \\lambda_v`.
    lambda_decay : float
        Decay factor in the EMA update of :attr:`lambda_v`. Values closer
        to 1.0 yield slower, smoother updates; smaller values react more
        quickly to changes in the volume error.
    """

    interpolation: Literal["SIMP"] = "SIMP"
    mu_p: float = 1e-1
    lambda_v: float = 1e+2
    lambda_decay: float = 0.70


# log(x) = -0.4   →   x ≈ 0.670
# log(x) = -0.3   →   x ≈ 0.741
# log(x) = -0.2   →   x ≈ 0.819
# log(x) = -0.1   →   x ≈ 0.905
# log(x) =  0.0   →   x =  1.000
# log(x) = +0.1   →   x ≈ 1.105
# log(x) = +0.2   →   x ≈ 1.221
# log(x) = +0.3   →   x ≈ 1.350
# log(x) = +0.4   →   x ≈ 1.492


def moc_log_update_logspace(
    rho,
    dC, lambda_v, scaling_rate,
    eta, move_limit,
    rho_clip_lower, rho_clip_upper,
    rho_min, rho_max
):
    eps = 1e-10
    logger.info(f"dC: {dC.min()} {dC.max()}")
    np.negative(dC, out=scaling_rate)
    scaling_rate /= (lambda_v + eps)
    np.maximum(scaling_rate, eps, out=scaling_rate)
    np.log(scaling_rate, out=scaling_rate)
    np.clip(scaling_rate, -0.50, 0.50, out=scaling_rate)
    np.clip(rho, rho_min, 1.0, out=rho)
    np.log(rho, out=rho_clip_lower)

    # rho_clip_upper = exp(rho_clip_lower) = rho (real space)
    np.exp(rho_clip_lower, out=rho_clip_upper)
    # rho_clip_upper = log(1 + move_limit / rho)
    np.divide(move_limit, rho_clip_upper, out=rho_clip_upper)
    np.add(rho_clip_upper, 1.0, out=rho_clip_upper)
    np.log(rho_clip_upper, out=rho_clip_upper)

    # rho_clip_lower = lower bound = log(rho) - log_move_limit
    np.subtract(rho_clip_lower, rho_clip_upper, out=rho_clip_lower)

    # rho_clip_upper = upper bound = log(rho) + log_move_limit
    np.add(rho_clip_lower, 2 * rho_clip_upper, out=rho_clip_upper)

    # rho = log(rho)
    np.log(rho, out=rho)

    # log(rho) += η * scaling_rate
    rho += eta * scaling_rate

    # clip in log-space
    np.clip(rho, rho_clip_lower, rho_clip_upper, out=rho)

    # back to real space
    np.exp(rho, out=rho)
    np.clip(rho, rho_min, rho_max, out=rho)


# Lagrangian Dual MOC
class LogMOC_Optimizer(common_density.DensityMethod):
    """
    Topology optimization solver using a log-space, OC-like multiplicative update.

    This optimizer implements a density-based compliance minimization
    algorithm in which the classic OC multiplicative update is expressed
    in log(ρ)-space. The design densities are updated according to a
    smoothed ratio of sensitivities and a dual variable associated with
    the volume constraint.

    In terms of ρ, the update has the conceptual form

    .. math::

        \\rho^{\\text{new}}
        = \\rho^{\\text{old}}
          \\left(-\\frac{\\partial C / \\partial \\rho}{\\lambda_v}
          \\right)^{\\eta},

    where :math:`\\lambda_v` is a dual variable updated by an exponential
    moving average (EMA) driven by the volume constraint violation. The
    implementation performs this update in log-space, enforces move limits
    on :math:`\\log \\rho`, and finally clips the densities to the admissible
    interval :math:`[\\rho_{\\min}, \\rho_{\\max}]`.

    Compared to the classic OC method, this log-space variant can offer
    improved numerical robustness while preserving the intuitive
    multiplicative structure.

    Attributes
    ----------
    cfg : LogMOC_Config
        Configuration object specifying the OC-like settings, including
        :attr:`mu_p`, :attr:`lambda_v`, EMA decay factors, and bounds on
        the densities and dual variables.
    tsk : sktopt.mesh.FEMDomain
        Finite element model (mesh, basis, boundary conditions and loads).
    recorder : HistoryLogger
        Logger that stores histories of selected quantities such as
        ``-dC``, ``lambda_v``, volume error and KKT residuals.
    """

    def __init__(
        self,
        cfg: LogMOC_Config,
        tsk: sktopt.mesh.FEMDomain,
    ):
        assert cfg.lambda_lower < cfg.lambda_upper
        super().__init__(cfg, tsk)
        self.recorder = self.add_recorder(tsk)
        if isinstance(
            cfg.percentile.target_value, float
        ):
            ylog_dC = True if cfg.percentile.target_value > 0 else False
        else:
            ylog_dC = True
        ylog_lambda_v = True if cfg.lambda_lower > 0.0 else False
        self.recorder.add("-dC", plot_type="min-max-mean-std", ylog=ylog_dC)
        self.recorder.add(
            "lambda_v", ylog=ylog_lambda_v
        )
        self.lambda_v = cfg.lambda_v

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
        tsk = self.tsk

        if self._dC_raw_buffer is None:
            self._dC_raw_buffer = np.empty_like(dC_drho_design_eles)

        np.copyto(self._dC_raw_buffer, dC_drho_design_eles)

        eps = 1e-8
        if isinstance(percentile, float):
            scale = np.percentile(np.abs(dC_drho_design_eles), percentile)
            self.recorder.feed_data("-dC", -dC_drho_design_eles)
            self.running_scale = 0.2 * self.running_scale + \
                (1 - 0.2) * scale if iter_num > 1 else scale
            logger.info(f"running_scale: {self.running_scale}")
            dC_drho_design_eles = dC_drho_design_eles / (self.running_scale + eps)
        else:
            pass

        # EMA
        volume = np.sum(
            rho_projected[tsk.design_elements] * elements_volume_design
        ) / elements_volume_design_sum
        volume_ratio = volume / vol_frac  # Target = 1.0

        # error in volume ratio ( without log)
        vol_error = volume_ratio - 1.0  # >0: over, <0: under

        # Averaging with EMA
        temp = cfg.lambda_decay * self.lambda_v
        temp += (1 - cfg.lambda_decay) * cfg.mu_p * vol_error
        self.lambda_v = temp
        self.lambda_v = np.clip(
            self.lambda_v, cfg.lambda_lower, cfg.lambda_upper
        )
        # lam_e = self.lambda_v * \
        #     (elements_volume_design / (elements_volume_design_sum + 1e-10))

        mask_int = (
            (rho_design_eles > cfg.rho_min + 1e-6) &
            (rho_design_eles < cfg.rho_max - 1e-6)
        )

        if np.any(mask_int):
            # dL/dρ_i ≈ dC_raw_i + λ_eff * dV/drho_i
            # The definition of λ_eff would be A or B
            #   A: λ_eff = self.lambda_v
            #   B: λ_eff = cfg.mu_p * self.lambda_v
            lambda_eff = self.lambda_v  # or cfg.mu_p * self.lambda_v
            dL = self._dC_raw_buffer[mask_int] \
                + lambda_eff * self._dV_drho_design[mask_int]
            self.kkt_residual = float(np.linalg.norm(dL, ord=np.inf))
        else:
            self.kkt_residual = 0.0

        self.recorder.feed_data("lambda_v", self.lambda_v)
        self.recorder.feed_data("vol_error", vol_error)
        self.recorder.feed_data("-dC", -dC_drho_design_eles)
        self.recorder.feed_data("kkt_residual", self.kkt_residual)

        moc_log_update_logspace(
            rho_design_eles,
            dC_drho_design_eles,
            # lam_e,
            self.lambda_v,
            scaling_rate,
            eta,
            move_limit,
            rho_clip_lower, rho_clip_upper,
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
        '--mu_p', '-MUP', type=float, default=5000.0, help=''
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
    # cfg = LogMOC_Config.from_defaults(
    #     **vars(args)
    # )
    cfg = LogMOC_Config.from_defaults(
        **misc.args2OC_Config_dict(vars(args))
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
