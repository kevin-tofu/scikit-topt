from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import scipy

from sktopt.fea import composer
from sktopt.fea.solver_elastic import (
    LinearSolverConfig,
    compute_compliance_basis as _compute_compliance_basis,
    compute_compliance_basis_multi_load as _compute_compliance_basis_multi_load,
    solve_multi_load as _solve_multi_load,
    solve_u as _solve_u,
)


def _normalize_solver_arg(
    chosen_solver: Literal["auto", "cg_jacobi", "spsolve", "cg_pyamg"] | LinearSolverConfig,
    rtol: float = 1e-5,
    maxiter: int | None = None,
) -> LinearSolverConfig:
    if isinstance(chosen_solver, LinearSolverConfig):
        return chosen_solver
    return LinearSolverConfig(
        solver=chosen_solver,
        rtol=rtol,
        maxiter=maxiter,
        allow_fallback_to_spsolve=(chosen_solver == "auto"),
    )


def compute_compliance_simp_basis(
    basis,
    free_dofs,
    dirichlet_dofs,
    force,
    E0,
    Emin,
    p,
    nu0,
    rho,
) -> tuple[float, np.ndarray]:
    return _compute_compliance_basis(
        basis,
        free_dofs,
        dirichlet_dofs,
        force,
        E0,
        Emin,
        p,
        nu0,
        rho,
        elem_func=composer.simp_interpolation,
    )


def solve_u(
    K_cond: scipy.sparse.csc_matrix,
    F_cond: np.ndarray,
    chosen_solver: Literal["auto", "cg_jacobi", "spsolve", "cg_pyamg"] | LinearSolverConfig = "auto",
    rtol: float = 1e-8,
    maxiter: int | None = None,
) -> np.ndarray:
    return _solve_u(
        K_cond,
        F_cond,
        chosen_solver=_normalize_solver_arg(chosen_solver, rtol=rtol, maxiter=maxiter),
    )


def compute_compliance_basis(
    basis,
    free_dofs,
    dirichlet_dofs,
    force,
    E0,
    Emin,
    p,
    nu0,
    rho,
    elem_func: Callable = composer.simp_interpolation,
    solver: Literal["auto", "cg_jacobi", "spsolve", "cg_pyamg"] = "auto",
    rtol: float = 1e-5,
    maxiter: int | None = None,
) -> tuple[float, np.ndarray]:
    return _compute_compliance_basis(
        basis,
        free_dofs,
        dirichlet_dofs,
        force,
        E0,
        Emin,
        p,
        nu0,
        rho,
        elem_func=elem_func,
        solver=solver,
        rtol=rtol,
        maxiter=maxiter,
    )


def compute_compliance_basis_numba(
    basis,
    free_dofs,
    dirichlet_dofs,
    force,
    E0,
    Emin,
    p,
    nu0,
    rho,
    elem_func: Callable = composer.simp_interpolation,
    solver: Literal["auto", "cg_jacobi", "spsolve", "cg_pyamg"] = "auto",
    rtol: float = 1e-5,
    maxiter: int | None = None,
) -> tuple[float, np.ndarray]:
    return compute_compliance_basis(
        basis,
        free_dofs,
        dirichlet_dofs,
        force,
        E0,
        Emin,
        p,
        nu0,
        rho,
        elem_func=elem_func,
        solver=solver,
        rtol=rtol,
        maxiter=maxiter,
    )


def solve_multi_load(
    basis,
    free_dofs,
    dirichlet_dofs,
    force_list,
    E0,
    Emin,
    p,
    nu0,
    rho,
    u_all,
    solver: Literal["auto", "cg_jacobi", "spsolve", "cg_pyamg"] = "auto",
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int | None = None,
    n_joblib: int = 1,
):
    return _solve_multi_load(
        basis,
        free_dofs,
        dirichlet_dofs,
        force_list,
        E0,
        Emin,
        p,
        nu0,
        rho,
        u_all,
        solver=solver,
        elem_func=elem_func,
        rtol=rtol,
        maxiter=maxiter,
        n_joblib=n_joblib,
    )


def compute_compliance_basis_multi_load(
    basis,
    free_dofs,
    dirichlet_dofs,
    force_list,
    E0,
    Emin,
    p,
    nu0,
    rho,
    u_all,
    solver: Literal["auto", "cg_jacobi", "spsolve", "cg_pyamg"] = "auto",
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int | None = None,
    n_joblib: int = 1,
):
    return _compute_compliance_basis_multi_load(
        basis,
        free_dofs,
        dirichlet_dofs,
        force_list,
        E0,
        Emin,
        p,
        nu0,
        rho,
        u_all,
        solver=solver,
        elem_func=elem_func,
        rtol=rtol,
        maxiter=maxiter,
        n_joblib=n_joblib,
    )


__all__ = [
    "compute_compliance_simp_basis",
    "solve_u",
    "compute_compliance_basis",
    "compute_compliance_basis_numba",
    "solve_multi_load",
    "compute_compliance_basis_multi_load",
]
