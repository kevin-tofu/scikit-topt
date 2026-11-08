"""Moving-asymptote density update for the shared density optimizer loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sktopt.core import projection
from sktopt.core.optimizers import common_density


@dataclass
class MMA_Config(common_density.DensityMethod_OC_Config):
    """Settings for the Method of Moving Asymptotes density update."""

    mma_dual_tolerance: float = 1e-9
    mma_dual_max_iterations: int = 100
    mma_asymptote_initial: float = 0.5
    mma_asymptote_increase: float = 1.2
    mma_asymptote_decrease: float = 0.7
    mma_asymptote_minimum: float = 0.01
    mma_asymptote_maximum: float = 10.0
    mma_regularization: float = 1e-5
    volume_constraint_tolerance: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if self.mma_dual_tolerance <= 0.0:
            raise ValueError("mma_dual_tolerance must be positive")
        if self.mma_dual_max_iterations < 1:
            raise ValueError("mma_dual_max_iterations must be positive")
        if self.mma_asymptote_initial <= 0.0:
            raise ValueError("mma_asymptote_initial must be positive")
        if self.mma_asymptote_increase < 1.0:
            raise ValueError("mma_asymptote_increase must be at least one")
        if not 0.0 < self.mma_asymptote_decrease <= 1.0:
            raise ValueError("mma_asymptote_decrease must be in (0, 1]")
        if not 0.0 < self.mma_asymptote_minimum < self.mma_asymptote_maximum:
            raise ValueError("invalid MMA asymptote distance limits")
        if self.mma_regularization <= 0.0:
            raise ValueError("mma_regularization must be positive")
        if self.volume_constraint_tolerance < 0.0:
            raise ValueError("volume_constraint_tolerance cannot be negative")


@dataclass
class _MMAHistory:
    """Design and asymptote history retained between outer FEM iterations."""

    previous: np.ndarray | None = None
    previous_previous: np.ndarray | None = None
    lower_asymptote: np.ndarray | None = None
    upper_asymptote: np.ndarray | None = None

    def asymptotes(
        self,
        x: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        cfg: MMA_Config,
    ) -> tuple[np.ndarray, np.ndarray]:
        span = upper_bound - lower_bound
        if self.lower_asymptote is None or self.upper_asymptote is None:
            distance = cfg.mma_asymptote_initial * span
            return x - distance, x + distance

        factor = np.ones_like(x)
        if self.previous is not None and self.previous_previous is not None:
            trend = (x - self.previous) * (self.previous - self.previous_previous)
            factor[trend > 0.0] = cfg.mma_asymptote_increase
            factor[trend < 0.0] = cfg.mma_asymptote_decrease

        lower = x - factor * (self.previous - self.lower_asymptote)
        upper = x + factor * (self.upper_asymptote - self.previous)
        min_distance = cfg.mma_asymptote_minimum * span
        max_distance = cfg.mma_asymptote_maximum * span
        lower = np.minimum(x - min_distance, np.maximum(x - max_distance, lower))
        upper = np.maximum(x + min_distance, np.minimum(x + max_distance, upper))
        return lower, upper

    def advance(
        self,
        x: np.ndarray,
        lower_asymptote: np.ndarray,
        upper_asymptote: np.ndarray,
    ) -> None:
        if self.previous is not None:
            self.previous_previous = self.previous.copy()
        self.previous = x.copy()
        self.lower_asymptote = lower_asymptote.copy()
        self.upper_asymptote = upper_asymptote.copy()


def _reciprocal_coefficients(
    gradient: np.ndarray,
    x: np.ndarray,
    lower_asymptote: np.ndarray,
    upper_asymptote: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the convex reciprocal approximation matching value gradient."""
    upper_distance = upper_asymptote - x
    lower_distance = x - lower_asymptote
    positive = np.maximum(gradient, 0.0)
    negative = np.maximum(-gradient, 0.0)
    p = (positive + regularization) * upper_distance**2
    q = (negative + regularization) * lower_distance**2
    return p, q


def _approximation_value(
    x: np.ndarray,
    x0: np.ndarray,
    f0: float,
    p: np.ndarray,
    q: np.ndarray,
    lower_asymptote: np.ndarray,
    upper_asymptote: np.ndarray,
) -> float:
    """Evaluate the MMA approximation anchored at ``(x0, f0)``."""
    return float(
        f0
        + np.sum(p * (
            1.0 / (upper_asymptote - x)
            - 1.0 / (upper_asymptote - x0)
        ))
        + np.sum(q * (
            1.0 / (x - lower_asymptote)
            - 1.0 / (x0 - lower_asymptote)
        ))
    )


def _solve_one_constraint_subproblem(
    *,
    x0: np.ndarray,
    objective_gradient: np.ndarray,
    constraint_value: float,
    constraint_gradient: np.ndarray,
    lower_asymptote: np.ndarray,
    upper_asymptote: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    regularization: float,
    dual_tolerance: float,
    dual_max_iterations: int,
) -> tuple[np.ndarray, float, float]:
    """Solve the separable MMA subproblem with one inequality constraint.

    The primal variables have a closed form for a given nonnegative dual
    multiplier.  A bracketed dual bisection then enforces the volume
    constraint.  All function evaluations inside this solve use the current
    convex MMA approximations; the FEM model is reevaluated by the outer loop.
    """
    p0, q0 = _reciprocal_coefficients(
        objective_gradient, x0, lower_asymptote, upper_asymptote,
        regularization,
    )
    p1, q1 = _reciprocal_coefficients(
        constraint_gradient, x0, lower_asymptote, upper_asymptote,
        regularization,
    )

    def primal(dual: float) -> np.ndarray:
        p = np.maximum(p0 + dual * p1, np.finfo(float).tiny)
        q = np.maximum(q0 + dual * q1, np.finfo(float).tiny)
        sqrt_p = np.sqrt(p)
        sqrt_q = np.sqrt(q)
        candidate = (
            sqrt_q * upper_asymptote + sqrt_p * lower_asymptote
        ) / (sqrt_p + sqrt_q)
        return np.clip(candidate, lower_bound, upper_bound)

    def constraint_approximation(x: np.ndarray) -> float:
        return _approximation_value(
            x, x0, constraint_value, p1, q1,
            lower_asymptote, upper_asymptote,
        )

    x_unconstrained = primal(0.0)
    if constraint_approximation(x_unconstrained) <= 0.0:
        return x_unconstrained, 0.0, constraint_approximation(x_unconstrained)

    dual_lower = 0.0
    dual_upper = 1.0
    x_upper = primal(dual_upper)
    while (
        constraint_approximation(x_upper) > 0.0
        and dual_upper < 1e30
    ):
        dual_upper *= 2.0
        x_upper = primal(dual_upper)

    if constraint_approximation(x_upper) > 0.0:
        return x_upper, dual_upper, constraint_approximation(x_upper)

    for _ in range(dual_max_iterations):
        dual_mid = 0.5 * (dual_lower + dual_upper)
        x_mid = primal(dual_mid)
        value_mid = constraint_approximation(x_mid)
        if value_mid > 0.0:
            dual_lower = dual_mid
        else:
            dual_upper = dual_mid
            x_upper = x_mid
        if dual_upper - dual_lower <= dual_tolerance * max(1.0, dual_upper):
            break
    return x_upper, dual_upper, constraint_approximation(x_upper)


class MMA_Optimizer(common_density.DensityMethod):
    """Density optimizer using a stateful moving-asymptote subproblem."""

    exact_filter_gradient = True

    def __init__(self, cfg: MMA_Config, tsk):
        super().__init__(cfg, tsk)
        self.recorder = self.add_recorder(tsk)
        self._mma_history = _MMAHistory()

    def _volume_constraint_value_gradient(
        self,
        x: np.ndarray,
        rho_full_base: np.ndarray,
        design: np.ndarray,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        beta: float,
        vol_frac: float,
    ) -> tuple[float, np.ndarray]:
        rho_full = rho_full_base.copy()
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
        np.multiply(volume_gradient, derivative, out=volume_gradient)
        gradient = self.filter.gradient_exact(volume_gradient)
        return value, gradient[design]

    def _volume_constraint_value(
        self,
        x: np.ndarray,
        rho_full_base: np.ndarray,
        design: np.ndarray,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        beta: float,
        vol_frac: float,
    ) -> float:
        rho_full = rho_full_base.copy()
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
        global_lower = np.full_like(x0, cfg.rho_min)
        global_upper = np.full_like(x0, cfg.rho_max)
        lower_asymptote, upper_asymptote = self._mma_history.asymptotes(
            x0, global_lower, global_upper, cfg
        )

        move = np.broadcast_to(np.asarray(move_limit, dtype=float), x0.shape)
        lower = np.maximum.reduce((
            global_lower,
            lower_asymptote + 0.1 * (x0 - lower_asymptote),
            x0 - move,
        ))
        upper = np.minimum.reduce((
            global_upper,
            upper_asymptote - 0.1 * (upper_asymptote - x0),
            x0 + move,
        ))
        if np.any(lower >= upper):
            raise ValueError("MMA bounds leave no feasible density step")

        constraint_value, constraint_gradient_full = (
            self._volume_constraint_value_gradient(
                x0,
                state.rho,
                design,
                elements_volume_design,
                elements_volume_design_sum,
                beta,
                vol_frac,
            )
        )
        x_new, dual, approximated_constraint = (
            _solve_one_constraint_subproblem(
                x0=x0,
                objective_gradient=objective_gradient,
                constraint_value=constraint_value,
                constraint_gradient=constraint_gradient_full,
                lower_asymptote=lower_asymptote,
                upper_asymptote=upper_asymptote,
                lower_bound=lower,
                upper_bound=upper,
                regularization=cfg.mma_regularization,
                dual_tolerance=cfg.mma_dual_tolerance,
                dual_max_iterations=cfg.mma_dual_max_iterations,
            )
        )
        current_physical_constraint = self._volume_constraint_value(
            x0,
            state.rho,
            design,
            elements_volume_design,
            elements_volume_design_sum,
            beta,
            vol_frac,
        )
        candidate_physical_constraint = self._volume_constraint_value(
            x_new,
            state.rho,
            design,
            elements_volume_design,
            elements_volume_design_sum,
            beta,
            vol_frac,
        )
        # The convex reciprocal model linearizes the projected volume at x0.
        # For a large move, correct its nonlinear projection error along the
        # feasible segment so the actual FEM design respects the requested
        # volume fraction.
        if (
            current_physical_constraint > -cfg.volume_constraint_tolerance
            and candidate_physical_constraint
            < -cfg.volume_constraint_tolerance
        ):
            target_constraint = min(
                current_physical_constraint,
                cfg.volume_constraint_tolerance,
            )
            alpha_lower, alpha_upper = 0.0, 1.0
            for _ in range(40):
                alpha = 0.5 * (alpha_lower + alpha_upper)
                trial = x0 + alpha * (x_new - x0)
                trial_constraint = self._volume_constraint_value(
                    trial,
                    state.rho,
                    design,
                    elements_volume_design,
                    elements_volume_design_sum,
                    beta,
                    vol_frac,
                )
                if trial_constraint > target_constraint:
                    alpha_lower = alpha
                else:
                    alpha_upper = alpha
            x_new = x0 + alpha_upper * (x_new - x0)
        rho_design_eles[:] = x_new
        np.divide(
            x_new,
            x0,
            out=scaling_rate,
            where=np.abs(x0) > np.finfo(float).eps,
        )
        scaling_rate[np.abs(x0) <= np.finfo(float).eps] = 1.0
        physical_volume_error = self._volume_constraint_value(
            rho_design_eles,
            state.rho,
            design,
            elements_volume_design,
            elements_volume_design_sum,
            beta,
            vol_frac,
        )
        self._mma_history.advance(
            x0, lower_asymptote, upper_asymptote
        )

        f_approx_gradient = (
            _reciprocal_coefficients(
                objective_gradient, x0, lower_asymptote,
                upper_asymptote, cfg.mma_regularization,
            )[0] / (upper_asymptote - x_new) ** 2
            - _reciprocal_coefficients(
                objective_gradient, x0, lower_asymptote,
                upper_asymptote, cfg.mma_regularization,
            )[1] / (x_new - lower_asymptote) ** 2
        )
        g_coefficients = _reciprocal_coefficients(
            constraint_gradient_full,
            x0,
            lower_asymptote,
            upper_asymptote,
            cfg.mma_regularization,
        )
        g_approx_gradient = (
            g_coefficients[0] / (upper_asymptote - x_new) ** 2
            - g_coefficients[1] / (x_new - lower_asymptote) ** 2
        )
        stationarity = f_approx_gradient + dual * g_approx_gradient
        active_lower = x_new <= lower + 1e-8
        active_upper = x_new >= upper - 1e-8
        stationarity[active_lower] = np.minimum(stationarity[active_lower], 0.0)
        stationarity[active_upper] = np.maximum(stationarity[active_upper], 0.0)
        self.kkt_residual = float(max(
            np.linalg.norm(stationarity, ord=np.inf),
            max(approximated_constraint, 0.0),
            abs(dual * approximated_constraint),
        ))
        self.recorder.feed_data("vol_error", physical_volume_error)
        self.recorder.feed_data("kkt_residual", self.kkt_residual)


__all__ = ["MMA_Config", "MMA_Optimizer"]
