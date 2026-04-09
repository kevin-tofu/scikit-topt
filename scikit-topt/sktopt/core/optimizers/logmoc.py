from typing import Literal
from dataclasses import dataclass
import numpy as np
import sktopt
from sktopt.core import misc
from sktopt.core import projection
from sktopt.core.optimizers import common_density
from sktopt.tools.logconf import mylogger

logger = mylogger(__name__)


@dataclass
class LogMOC_Config(common_density.DensityMethod_OC_Config):
    interpolation: Literal["SIMP"] = "SIMP"
    mu_p: float = 5.0
    augmented_lagrangian_mu: float = 0.0
    lambda_v: float = 0.1
    lambda_decay: float = 0.90
    lambda_lower: float = -1e+7
    lambda_upper: float = 1e+7
    lagrangian_clip: float = 1.0
    lagrangian_percentile: float = 95.0
    lagrangian_scale_floor: float = 1e-8


def lagrangian_log_update(
    rho,
    dL,
    scaling_rate,
    eta,
    move_limit,
    rho_clip_lower,
    rho_clip_upper,
    rho_min,
    rho_max,
    lagrangian_clip,
):
    np.copyto(scaling_rate, dL)
    np.clip(scaling_rate, -lagrangian_clip, lagrangian_clip, out=scaling_rate)

    np.clip(rho, rho_min, rho_max, out=rho)
    np.log(rho, out=rho_clip_lower)

    np.exp(rho_clip_lower, out=rho_clip_upper)
    np.divide(move_limit, rho_clip_upper, out=rho_clip_upper)
    np.add(rho_clip_upper, 1.0, out=rho_clip_upper)
    np.log(rho_clip_upper, out=rho_clip_upper)

    np.subtract(rho_clip_lower, rho_clip_upper, out=rho_clip_lower)
    np.add(rho_clip_lower, 2.0 * rho_clip_upper, out=rho_clip_upper)

    np.log(rho, out=rho)
    rho -= eta * scaling_rate
    np.clip(rho, rho_clip_lower, rho_clip_upper, out=rho)
    np.exp(rho, out=rho)
    np.clip(rho, rho_min, rho_max, out=rho)


class LogMOC_Optimizer(common_density.DensityMethod):
    def __init__(
        self,
        cfg: LogMOC_Config,
        tsk: sktopt.mesh.FEMDomain,
    ):
        super().__init__(cfg, tsk)
        self.recorder = self.add_recorder(tsk)
        self.recorder.add("dL", plot_type="min-max-mean-std", ylog=False)
        self.recorder.add("-dC", plot_type="min-max-mean-std", ylog=True)
        self.recorder.add("lambda_v", ylog=False)
        self.recorder.add("constraint_coeff", ylog=False)
        self.lambda_v = cfg.lambda_v
        self._dL_buffer = None
        self._dV_chain_design = None
        self._dV_projected_full = None
        self._dV_filtered_full = None

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
        del u_dofs, strain_energy_mean
        cfg = self.cfg
        tsk = self.tsk

        if self._dC_raw_buffer is None:
            self._dC_raw_buffer = np.empty_like(dC_drho_design_eles)
        if self._dL_buffer is None:
            self._dL_buffer = np.empty_like(dC_drho_design_eles)
        if self._dV_chain_design is None:
            self._dV_chain_design = np.empty_like(dC_drho_design_eles)
        if self._dV_projected_full is None:
            state = self._state
            if state is None:
                raise RuntimeError("Optimizer state is not initialized.")
            self._dV_projected_full = np.zeros_like(state.rho)
            self._dV_filtered_full = np.zeros_like(state.rho)

        np.copyto(self._dC_raw_buffer, dC_drho_design_eles)
        np.copyto(self._dV_projected_full, 0.0)
        self._dV_projected_full[tsk.design_elements] = (
            elements_volume_design / elements_volume_design_sum
        )
        projection.heaviside_projection_derivative_inplace(
            self._state.rho_filtered,
            beta=beta,
            eta=cfg.beta_eta,
            out=self._dV_filtered_full,
        )
        self._dV_filtered_full *= self._dV_projected_full
        np.copyto(
            self._dV_projected_full,
            self.filter.gradient(self._dV_filtered_full)
        )
        np.copyto(
            self._dV_chain_design,
            self._dV_projected_full[tsk.design_elements]
        )

        volume = np.sum(
            rho_projected[tsk.design_elements] * elements_volume_design
        ) / elements_volume_design_sum
        vol_error = volume - vol_frac

        penalty = cfg.mu_p * vol_error
        if iter_num > 1:
            self.lambda_v = cfg.lambda_decay * self.lambda_v + \
                (1.0 - cfg.lambda_decay) * penalty
        else:
            self.lambda_v = penalty
        self.lambda_v = np.clip(
            self.lambda_v, cfg.lambda_lower, cfg.lambda_upper
        )
        constraint_coeff = self.lambda_v + \
            cfg.augmented_lagrangian_mu * vol_error

        np.copyto(self._dL_buffer, self._dC_raw_buffer)
        self._dL_buffer += constraint_coeff * self._dV_chain_design

        scale_percentile = cfg.lagrangian_percentile
        if isinstance(percentile, float):
            scale_percentile = percentile
        scale = np.percentile(np.abs(self._dL_buffer), scale_percentile)
        scale = max(scale, cfg.lagrangian_scale_floor)
        self.running_scale = 0.2 * self.running_scale + \
            (1.0 - 0.2) * scale if iter_num > 1 else scale
        logger.info(f"running_scale: {self.running_scale}")
        self._dL_buffer /= self.running_scale

        mask_int = (
            (rho_design_eles > cfg.rho_min + 1e-6) &
            (rho_design_eles < cfg.rho_max - 1e-6)
        )
        if np.any(mask_int):
            self.kkt_residual = float(
                np.linalg.norm(self._dL_buffer[mask_int], ord=np.inf)
            )
        else:
            self.kkt_residual = 0.0

        self.recorder.feed_data("lambda_v", self.lambda_v)
        self.recorder.feed_data("constraint_coeff", constraint_coeff)
        self.recorder.feed_data("vol_error", vol_error)
        self.recorder.feed_data("-dC", -self._dC_raw_buffer)
        self.recorder.feed_data("dL", self._dL_buffer)
        self.recorder.feed_data("kkt_residual", self.kkt_residual)

        lagrangian_log_update(
            rho_design_eles,
            self._dL_buffer,
            scaling_rate,
            eta,
            move_limit,
            rho_clip_lower,
            rho_clip_upper,
            cfg.rho_min,
            cfg.rho_max,
            cfg.lagrangian_clip,
        )


if __name__ == '__main__':
    import argparse
    from sktopt.mesh import toy_problem

    parser = argparse.ArgumentParser(description='')
    parser = misc.add_common_arguments(parser)
    parser.add_argument(
        '--mu_p', '-MUP', type=float, default=5.0, help=''
    )
    parser.add_argument(
        '--augmented_lagrangian_mu', '-ALM', type=float, default=0.0, help=''
    )
    parser.add_argument(
        '--lambda_v', '-LV', type=float, default=0.1, help=''
    )
    parser.add_argument(
        '--lambda_decay', '-LD', type=float, default=0.90, help=''
    )
    parser.add_argument(
        '--lagrangian_clip', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--lagrangian_percentile', type=float, default=95.0, help=''
    )
    parser.add_argument(
        '--lagrangian_scale_floor', type=float, default=1e-8, help=''
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

    cfg = LogMOC_Config.from_defaults(
        **misc.args2OC_Config_dict(vars(args))
    )
    optimizer = LogMOC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()
