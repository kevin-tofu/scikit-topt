"""Density optimization adapter for the standalone :mod:`mmapy` solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike
from mmapy import MMA, MMAOptions, MMAResult

from sktopt.core import projection
from sktopt.core.optimizers import common_density


class _MMASolverProtocol(Protocol):
    """MMA step contract consumed by the topology-optimization adapter."""

    def step(
        self,
        x: ArrayLike,
        objective: float,
        objective_gradient: ArrayLike,
        constraints: ArrayLike,
        constraint_jacobian: ArrayLike,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        *,
        move_limit: float | ArrayLike | None = None,
    ) -> MMAResult: ...


@dataclass
class MMA_Config(common_density.DensityMethod_OC_Config):
    """Settings for the Method of Moving Asymptotes density update."""

    mma_dual_tolerance: float = 1e-9
    mma_dual_max_iterations: int = 200
    mma_asymptote_initial: float = 0.5
    mma_asymptote_increase: float = 1.2
    mma_asymptote_decrease: float = 0.7
    mma_asymptote_minimum: float = 0.01
    mma_asymptote_maximum: float = 10.0
    mma_regularization: float = 1e-5
    volume_constraint_tolerance: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if self.volume_constraint_tolerance < 0.0:
            raise ValueError("volume_constraint_tolerance cannot be negative")


class MMA_Optimizer(common_density.DensityMethod):
    """Density optimizer backed by the reusable, stateful ``mmapy.MMA``."""

    exact_filter_gradient = True

    def __init__(self, cfg: MMA_Config, tsk):
        super().__init__(cfg, tsk)
        self.recorder = self.add_recorder(tsk)
        self._mma: _MMASolverProtocol = MMA(MMAOptions(
            initial_asymptote=cfg.mma_asymptote_initial,
            asymptote_increase=cfg.mma_asymptote_increase,
            asymptote_decrease=cfg.mma_asymptote_decrease,
            asymptote_minimum=cfg.mma_asymptote_minimum,
            asymptote_maximum=cfg.mma_asymptote_maximum,
            regularization=cfg.mma_regularization,
            dual_tolerance=cfg.mma_dual_tolerance,
            dual_max_iterations=cfg.mma_dual_max_iterations,
        ))

    def _volume_value_gradient(
        self,
        x: np.ndarray,
        state_density: np.ndarray,
        design: np.ndarray,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        beta: float,
        vol_frac: float,
    ) -> tuple[float, np.ndarray]:
        rho_full = state_density.copy()
        rho_full[design] = x
        rho_filtered = self.filter.forward(rho_full)
        rho_projected = np.empty_like(rho_full)
        projection.heaviside_projection_inplace(
            rho_filtered,
            beta=beta,
            eta=self.cfg.beta_eta,
            out=rho_projected,
        )
        value = float(
            np.sum(rho_projected[design] * elements_volume_design)
            / elements_volume_design_sum
            - vol_frac
        )
        derivative = np.empty_like(rho_full)
        projection.heaviside_projection_derivative_inplace(
            rho_filtered,
            beta=beta,
            eta=self.cfg.beta_eta,
            out=derivative,
        )
        volume_gradient = np.zeros_like(rho_full)
        volume_gradient[design] = (
            elements_volume_design / elements_volume_design_sum
        )
        volume_gradient *= derivative
        return value, self.filter.adjoint(volume_gradient)[design]

    def _volume_value(
        self,
        x: np.ndarray,
        state_density: np.ndarray,
        design: np.ndarray,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        beta: float,
        vol_frac: float,
    ) -> float:
        rho_full = state_density.copy()
        rho_full[design] = x
        rho_filtered = self.filter.forward(rho_full)
        rho_projected = np.empty_like(rho_full)
        projection.heaviside_projection_inplace(
            rho_filtered,
            beta=beta,
            eta=self.cfg.beta_eta,
            out=rho_projected,
        )
        return float(
            np.sum(rho_projected[design] * elements_volume_design)
            / elements_volume_design_sum
            - vol_frac
        )

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
            eta,
            rho_clip_lower,
            rho_clip_upper,
            percentile,
        )
        state = self._state
        if state is None:
            raise RuntimeError("Optimizer state is not initialized.")

        cfg = self.cfg
        design = self.tsk.design_elements
        x0 = np.asarray(rho_design_eles, dtype=float).copy()
        objective_gradient = np.asarray(
            dC_drho_design_eles, dtype=float
        ).copy()
        constraint_value, constraint_gradient = self._volume_value_gradient(
            x0,
            state.rho,
            design,
            elements_volume_design,
            elements_volume_design_sum,
            beta,
            vol_frac,
        )
        result = self._mma.step(
            x=x0,
            objective=0.0,
            objective_gradient=objective_gradient,
            constraints=[constraint_value],
            constraint_jacobian=constraint_gradient[None, :],
            lower_bounds=np.full_like(x0, cfg.rho_min),
            upper_bounds=np.full_like(x0, cfg.rho_max),
            move_limit=move_limit,
        )
        x_new = result.x

        current_error = constraint_value
        candidate_error = self._volume_value(
            x_new,
            state.rho,
            design,
            elements_volume_design,
            elements_volume_design_sum,
            beta,
            vol_frac,
        )
        # Correct curvature error from the nonlinear filter/projection along
        # the MMA step when the current design is close to active volume.
        if (
            current_error > -cfg.volume_constraint_tolerance
            and candidate_error < -cfg.volume_constraint_tolerance
        ):
            target_error = min(
                current_error, cfg.volume_constraint_tolerance
            )
            correction_tolerance = max(
                0.1 * cfg.volume_constraint_tolerance, 1e-12
            )
            alpha_lower, alpha_upper = 0.0, 1.0
            alpha_solution = None
            for _ in range(40):
                alpha = 0.5 * (alpha_lower + alpha_upper)
                trial = x0 + alpha * (result.x - x0)
                error = self._volume_value(
                    trial,
                    state.rho,
                    design,
                    elements_volume_design,
                    elements_volume_design_sum,
                    beta,
                    vol_frac,
                )
                if error > target_error:
                    alpha_lower = alpha
                else:
                    alpha_upper = alpha
                if (
                    error <= target_error
                    and target_error - error <= correction_tolerance
                ):
                    alpha_solution = alpha
                    break
            if alpha_solution is None:
                alpha_solution = alpha_upper
            x_new = x0 + alpha_solution * (result.x - x0)

        rho_design_eles[:] = x_new
        np.divide(
            x_new,
            x0,
            out=scaling_rate,
            where=np.abs(x0) > np.finfo(float).eps,
        )
        scaling_rate[np.abs(x0) <= np.finfo(float).eps] = 1.0
        physical_volume_error, final_volume_gradient = self._volume_value_gradient(
            x_new,
            state.rho,
            design,
            elements_volume_design,
            elements_volume_design_sum,
            beta,
            vol_frac,
        )
        multiplier = result.multipliers[0] if result.multipliers.size else 0.0
        stationarity = objective_gradient + multiplier * final_volume_gradient
        stationarity[x_new <= cfg.rho_min + 1e-8] = np.minimum(
            stationarity[x_new <= cfg.rho_min + 1e-8], 0.0
        )
        stationarity[x_new >= cfg.rho_max - 1e-8] = np.maximum(
            stationarity[x_new >= cfg.rho_max - 1e-8], 0.0
        )
        self.kkt_residual = float(max(
            np.linalg.norm(stationarity, ord=np.inf),
            max(physical_volume_error, 0.0),
            abs(multiplier * physical_volume_error),
        ))
        self.recorder.feed_data("vol_error", physical_volume_error)
        self.recorder.feed_data("kkt_residual", self.kkt_residual)


__all__ = ["MMA_Config", "MMA_Optimizer"]
