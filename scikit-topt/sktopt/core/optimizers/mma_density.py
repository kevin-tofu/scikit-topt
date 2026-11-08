"""MMA density update integrated with the shared density optimizer loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sktopt.core import projection
from sktopt.core.optimizers import common_density


@dataclass
class MMA_Config(common_density.DensityMethod_OC_Config):
    """Settings for the Method of Moving Asymptotes density update."""

    mma_maxeval: int = 50
    mma_xtol_rel: float = 1e-5
    volume_constraint_tolerance: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if self.mma_maxeval < 1:
            raise ValueError("mma_maxeval must be positive")
        if self.mma_xtol_rel <= 0.0:
            raise ValueError("mma_xtol_rel must be positive")
        if self.volume_constraint_tolerance < 0.0:
            raise ValueError("volume_constraint_tolerance cannot be negative")


class MMA_Optimizer(common_density.DensityMethod):
    """Density optimizer using NLopt's MMA algorithm for each design update.

    The shared density driver evaluates compliance and its filtered gradient.
    MMA minimizes the first-order compliance model subject to the physical,
    filtered-and-projected volume constraint.
    """

    def __init__(self, cfg: MMA_Config, tsk):
        super().__init__(cfg, tsk)
        self.recorder = self.add_recorder(tsk)

    def rho_update(
        self,
        iter_num: int,
        rho_design_eles: np.ndarray,
        rho_projected: np.ndarray,
        dC_drho_design_eles: np.ndarray,
        u_dofs: np.ndarray,
        energy_mean: np.ndarray,
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
        del (
            iter_num,
            rho_projected,
            u_dofs,
            energy_mean,
            scaling_rate,
            eta,
            rho_clip_lower,
            rho_clip_upper,
            percentile,
        )
        try:
            import nlopt
        except ImportError as error:
            raise ImportError(
                "MMA_Optimizer requires NLopt; install it with `poetry add nlopt`."
            ) from error

        state = self._state
        if state is None:
            raise RuntimeError("Optimizer state is not initialized.")

        cfg = self.cfg
        design = self.tsk.design_elements
        x0 = np.asarray(rho_design_eles, dtype=float).copy()
        objective_gradient = np.asarray(
            dC_drho_design_eles, dtype=float
        ).copy()
        lower = np.maximum(cfg.rho_min, x0 - move_limit)
        upper = np.minimum(cfg.rho_max, x0 + move_limit)
        if np.any(lower > upper):
            raise ValueError("MMA move limits produced inconsistent bounds")

        rho_full = state.rho.copy()
        rho_filtered = np.empty_like(rho_full)
        rho_projected_candidate = np.empty_like(rho_full)
        projection_derivative = np.empty_like(rho_full)
        volume_gradient_full = np.empty_like(rho_full)

        def volume_constraint(x, grad):
            rho_full[design] = x
            np.copyto(rho_filtered, self.filter.forward(rho_full))
            projection.heaviside_projection_inplace(
                rho_filtered,
                beta=beta,
                eta=cfg.beta_eta,
                out=rho_projected_candidate,
            )
            volume = np.sum(
                rho_projected_candidate[design] * elements_volume_design
            ) / elements_volume_design_sum

            if grad.size:
                projection.heaviside_projection_derivative_inplace(
                    rho_filtered,
                    beta=beta,
                    eta=cfg.beta_eta,
                    out=projection_derivative,
                )
                volume_gradient_full.fill(0.0)
                volume_gradient_full[design] = (
                    elements_volume_design / elements_volume_design_sum
                )
                np.multiply(
                    volume_gradient_full,
                    projection_derivative,
                    out=volume_gradient_full,
                )
                # The Helmholtz sensitivity filter clips positive inputs to
                # zero because compliance sensitivities are nonpositive.
                # Negating before and after filtering preserves the positive
                # volume-constraint gradient through that API.
                volume_gradient = -self.filter.gradient(
                    -volume_gradient_full
                )
                grad[:] = volume_gradient[design]
            return float(volume - vol_frac)

        def objective(x, grad):
            if grad.size:
                grad[:] = objective_gradient
            return float(np.dot(objective_gradient, x - x0))

        initial_volume_gradient = np.empty_like(x0)
        initial_volume_error = volume_constraint(
            x0, initial_volume_gradient
        )
        denominator = float(np.dot(
            initial_volume_gradient, initial_volume_gradient
        ))
        lagrange_multiplier = (
            max(
                0.0,
                -float(np.dot(objective_gradient, initial_volume_gradient))
                / denominator,
            )
            if initial_volume_error >= 0.0 and denominator > 0.0
            else 0.0
        )
        stationarity = (
            objective_gradient
            + lagrange_multiplier * initial_volume_gradient
        )
        stationarity[(x0 <= lower + 1e-8) & (stationarity > 0.0)] = 0.0
        stationarity[(x0 >= upper - 1e-8) & (stationarity < 0.0)] = 0.0

        opt = nlopt.opt(nlopt.LD_MMA, x0.size)
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)
        opt.set_min_objective(objective)
        opt.add_inequality_constraint(
            volume_constraint, cfg.volume_constraint_tolerance
        )
        opt.set_maxeval(cfg.mma_maxeval)
        opt.set_xtol_rel(cfg.mma_xtol_rel)
        x_new = opt.optimize(x0)
        rho_design_eles[:] = np.clip(x_new, lower, upper)

        final_volume_error = volume_constraint(rho_design_eles, np.empty(0))
        self.recorder.feed_data("vol_error", final_volume_error)
        self.kkt_residual = float(np.linalg.norm(stationarity, ord=np.inf))
        self.recorder.feed_data("kkt_residual", self.kkt_residual)


__all__ = ["MMA_Config", "MMA_Optimizer"]
