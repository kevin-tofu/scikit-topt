from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal
from contextlib import contextmanager

import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator


import skfem
from skfem.models.elasticity import lame_parameters
from skfem import Functional
from skfem.helpers import ddot, sym_grad, trace, eye
from skfem.helpers import transpose
import pyamg

from sktopt.fea import composer
from sktopt.fea._petsc_compat import (
    PETScOptions,
    petsc_options_for_solver,
    solve_u_petsc,
    solve_u_petsc_multi,
)
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)

ElasticSolver = Literal[
    'cg_jacobi', 'spsolve', 'cg_pyamg', 'petsc', 'petsc_spdirect'
]
SolverSelector = Literal['auto'] | ElasticSolver


@dataclass(frozen=True)
class LinearSolverConfig:
    solver: SolverSelector = "spsolve"
    rtol: float = 1e-8
    maxiter: int | None = None
    petsc_options: PETScOptions | None = None
    allow_fallback_to_spsolve: bool = False


def normalize_linear_solver_config(
    solver: SolverSelector | LinearSolverConfig,
    rtol: float = 1e-8,
    maxiter: int | None = None,
    petsc_options: PETScOptions | None = None,
    allow_fallback_to_spsolve: bool = False,
) -> LinearSolverConfig:
    if isinstance(solver, LinearSolverConfig):
        return solver
    return LinearSolverConfig(
        solver=solver,
        rtol=rtol,
        maxiter=maxiter,
        petsc_options=petsc_options,
        allow_fallback_to_spsolve=allow_fallback_to_spsolve,
    )


def solve_u(
    K_cond: scipy.sparse.spmatrix,
    F_cond: np.ndarray,
    chosen_solver: ElasticSolver | LinearSolverConfig = 'spsolve',
    rtol: float = 1e-8,
    maxiter: int = None,
    petsc_options: PETScOptions | None = None,
    allow_fallback_to_spsolve: bool = False,
) -> np.ndarray:
    solver_cfg = normalize_linear_solver_config(
        chosen_solver,
        rtol=rtol,
        maxiter=maxiter,
        petsc_options=petsc_options,
        allow_fallback_to_spsolve=allow_fallback_to_spsolve,
    )
    chosen_solver = solver_cfg.solver
    rtol = solver_cfg.rtol
    maxiter = solver_cfg.maxiter
    petsc_options = solver_cfg.petsc_options
    allow_fallback_to_spsolve = solver_cfg.allow_fallback_to_spsolve

    try:
        if chosen_solver == 'cg_jacobi':
            M_diag = K_cond.diagonal()
            M_inv = 1.0 / M_diag
            M = LinearOperator(K_cond.shape, matvec=lambda x: M_inv * x)

            u_c, info = scipy.sparse.linalg.cg(
                A=K_cond, b=F_cond, M=M, rtol=rtol, maxiter=maxiter
            )
            logger.info(f"CG (diag preconditioner) solver info: {info}")

        elif chosen_solver == 'cg_pyamg':
            ml = pyamg.smoothed_aggregation_solver(K_cond)
            M = ml.aspreconditioner()

            # u_c, info = scipy.sparse.linalg.cg(
            #     A=K_cond, b=F_cond, M=M, tol=rtol, maxiter=maxiter
            # )
            u_c, info = scipy.sparse.linalg.cg(
                A=K_cond, b=F_cond, M=M, rtol=rtol, maxiter=maxiter
            )
            logger.info(f"CG (AMG preconditioner) solver info: {info}")

        elif chosen_solver == 'spsolve':
            u_c = scipy.sparse.linalg.spsolve(K_cond, F_cond)
            info = 0
            logger.info("Direct solver used: spsolve")

        elif chosen_solver == 'petsc':
            u_c, info = solve_u_petsc(
                K_cond,
                F_cond,
                rtol=rtol,
                maxiter=maxiter,
                petsc_options=petsc_options,
            )
            logger.info(f"PETSc CG (GAMG) solver info: {info}")

        elif chosen_solver == 'petsc_spdirect':
            u_c, info = solve_u_petsc(
                K_cond,
                F_cond,
                rtol=rtol,
                maxiter=maxiter,
                petsc_options=petsc_options_for_solver(
                    chosen_solver,
                    petsc_options,
                ),
            )
            logger.info(f"PETSc sparse direct solver info: {info}")

        else:
            raise ValueError(f"Unknown solver: {chosen_solver}")

    except Exception as e:
        if not allow_fallback_to_spsolve:
            raise
        logger.warning(f"Solver exception - {e}, falling back to spsolve.")
        u_c = scipy.sparse.linalg.spsolve(K_cond, F_cond)

    return u_c


def compute_compliance_basis(
    basis, free_dofs, dirichlet_dofs, force,
    E0, Emin, p, nu0,
    rho,
    elem_func: Callable = composer.simp_interpolation,
    solver_config: LinearSolverConfig | None = None,
    solver: SolverSelector | LinearSolverConfig = 'auto',
    rtol: float = 1e-5,
    maxiter: int = None,
    petsc_options: PETScOptions | None = None,
    timer=None,
) -> tuple:
    """
    Assemble and solve a single-load compliance problem.

    Parameters
    ----------
    solver_config : LinearSolverConfig, optional
        Preferred solver configuration. When provided, it takes precedence
        over ``solver``, ``rtol``, ``maxiter``, and ``petsc_options``.
    solver, rtol, maxiter, petsc_options : optional
        Legacy convenience inputs retained for backward compatibility.
        These are normalized into a :class:`LinearSolverConfig` internally.
    """
    def section(name: str):
        if timer:
            return timer.section(name)
        @contextmanager
        def _noop():
            yield
        return _noop()
    with section("assemble"):
        K = composer.assemble_stiffness_matrix(
            basis, rho, E0, Emin, p, nu0, elem_func
        )
    n_dof = K.shape[0]
    # Solver auto-selection
    solver_cfg = (
        solver_config if solver_config is not None else
        normalize_linear_solver_config(
            solver, rtol=rtol, maxiter=maxiter, petsc_options=petsc_options
        )
    )
    if solver_cfg.solver == 'auto':
        if n_dof < 1000:
            chosen_solver = 'spsolve'
        elif n_dof < 30000:
            # chosen_solver = 'cg_jacobi'
            chosen_solver = 'cg_pyamg'
        else:
            chosen_solver = 'cg_pyamg'
            # chosen_solver = 'cg_jacobi'
    else:
        chosen_solver = solver_cfg.solver

    _maxiter = (
        min(1000, max(300, n_dof // 5))
        if solver_cfg.maxiter is None else solver_cfg.maxiter
    )
    K_csr = K.tocsr()
    all_dofs = np.arange(K_csr.shape[0])
    free_dofs = np.setdiff1d(all_dofs, dirichlet_dofs, assume_unique=True)

    # enforce
    with section("enforce_bc"):
        K_e, F_e = skfem.enforce(K_csr, force, D=dirichlet_dofs)
    with section("solve"):
        u = solve_u(
            K_e,
            F_e,
            chosen_solver=LinearSolverConfig(
                solver=chosen_solver,
                rtol=solver_cfg.rtol,
                maxiter=_maxiter,
                petsc_options=solver_cfg.petsc_options,
                allow_fallback_to_spsolve=(solver_cfg.solver == 'auto'),
            ),
        )

    # condense
    # K_c, F_c, U_c, I = skfem.condense(K, F, D=fixed_dofs)
    # K_c = K_csr[free_dofs, :][:, free_dofs]
    # F_c = force[free_dofs]
    # u_free = solve_u(
    #     K_c, F_c, chosen_solver=chosen_solver, rtol=rtol, maxiter=_maxiter
    # )
    # u = np.zeros_like(force)
    # u[free_dofs] = u_free
    # f_free = force[free_dofs]
    # compliance = f_free @ u[free_dofs]
    compliance = F_e[free_dofs] @ u[free_dofs]
    return (float(compliance), u)


def solve_multi_load(
    basis: skfem.CellBasis,
    free_dofs: np.ndarray,
    dirichlet_dofs: np.ndarray,
    force_list: list[np.ndarray],
    E0: float, Emin: float, p: float, nu0: float,
    rho: np.ndarray,
    u_all: np.ndarray,
    solver: SolverSelector | LinearSolverConfig = 'auto',
    solver_config: LinearSolverConfig | None = None,
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int | None = None,
    petsc_options: PETScOptions | None = None,
    timer=None,
) -> np.ndarray:
    """
    Solve a shared-stiffness linear elasticity problem for multiple load cases.

    The stiffness matrix K(ρ) is assembled once, and Dirichlet boundary
    conditions are enforced via `skfem.enforce`. All load cases share the
    same Dirichlet DOF set; only the right-hand side (Neumann loads) differs.

    This function uses a single LU factorization of the enforced matrix K_e
    and solves all right-hand sides in one call:

        u_all = lu.solve(F_stack)

    which is typically the fastest and most robust approach for
    shared-stiffness multi-load problems.

    Parameters
    ----------
    basis : skfem.CellBasis
        Finite element basis for the displacement field.
    free_dofs : np.ndarray
        Array of free DOF indices (currently unused; kept for API compatibility).
    dirichlet_dofs : np.ndarray
        DOF indices with Dirichlet boundary conditions, shared across all loads.
    force_list : list of (n_dof,) ndarray
        List of global load vectors, one per load case.
    E0, Emin : float
        Maximum and minimum Young's modulus values for the SIMP interpolation.
    p : float
        SIMP penalization exponent.
    nu0 : float
        Poisson's ratio (assumed constant).
    rho : (n_elem,) ndarray
        Element-wise density field.
    u_all : (n_dof, n_loads) ndarray
        Output array to store displacement solutions; each column is one load case.
    solver_config : LinearSolverConfig, optional
        Preferred solver configuration for the shared stiffness system.
        When provided, it takes precedence over ``solver``, ``rtol``,
        ``maxiter``, and ``petsc_options``.
    solver : {'auto', 'cg_jacobi', 'spsolve', 'cg_pyamg', 'petsc', 'petsc_spdirect'}, optional
        Legacy convenience selector for the shared stiffness system.
        - 'auto' or 'spsolve': direct LU factorization (splu) is used.
        - 'petsc': solve each RHS with a reused PETSc KSP object.
        - 'petsc_spdirect': solve each RHS with PETSc sparse direct solve.
        - 'cg_jacobi', 'cg_pyamg': currently not implemented for multi-load.
    elem_func : Callable, optional
        Density interpolation function, e.g. SIMP or RAMP.
    rtol, maxiter, petsc_options : optional
        Legacy convenience inputs retained for backward compatibility.
        These are normalized into ``solver_config`` internally.

    Returns
    -------
    F_stack : (n_dof, n_loads) ndarray
        Stack of enforced right-hand sides for each load case.
    """

    # fall back to single-load path to honor chosen solver/timer
    solver_cfg = (
        solver_config if solver_config is not None else
        normalize_linear_solver_config(
            solver, rtol=rtol, maxiter=maxiter, petsc_options=petsc_options
        )
    )

    if len(force_list) == 1:
        compliance, u = compute_compliance_basis(
            basis, free_dofs, dirichlet_dofs, force_list[0],
            E0, Emin, p, nu0,
            rho,
            elem_func=elem_func,
            solver_config=solver_cfg,
            timer=timer,
        )
        u_all[:, 0] = u
        return np.array([compliance])

    # Multi-load currently supports direct LU or PETSc.
    if solver_cfg.solver not in ('auto', 'spsolve', 'petsc', 'petsc_spdirect'):
        raise NotImplementedError(
            "solve_multi_load currently supports only direct LU "
            "(solver='auto' or 'spsolve') or PETSc "
            "(solver='petsc' or 'petsc_spdirect'). "
            "Other iterative solvers for multi-load are not implemented."
        )

    def section(name: str):
        if timer:
            return timer.section(name)
        @contextmanager
        def _noop():
            yield
        return _noop()

    # Assemble the global stiffness matrix with SIMP interpolation
    with section("assemble"):
        K = composer.assemble_stiffness_matrix(
            basis, rho, E0, Emin, p, nu0, elem_func
        )
    K_csr = K.tocsr()

    # Enforce Dirichlet BCs once to obtain the enforced matrix K_e.
    # The pattern of K_e depends only on the Dirichlet DOF set, not on
    # the particular load vector, so we can use the first load as a template.
    with section("enforce_bc"):
        K_e, _ = skfem.enforce(K_csr, force_list[0], D=dirichlet_dofs)

    # Build all enforced right-hand sides and stack them column-wise.
    with section("build_rhs"):
        F_stack = np.column_stack([
            skfem.enforce(K_csr, f, D=dirichlet_dofs)[1]
            for f in force_list
        ])  # shape: (n_dof, n_loads)

    if solver_cfg.solver in ("petsc", "petsc_spdirect"):
        _petsc_options = petsc_options_for_solver(
            solver_cfg.solver, solver_cfg.petsc_options
        )
        with section("solve"):
            u_sol, infos = solve_u_petsc_multi(
                K_e,
                F_stack,
                rtol=solver_cfg.rtol,
                maxiter=solver_cfg.maxiter,
                petsc_options=_petsc_options,
            )
        logger.info(f"PETSc multi-load solver info: {infos}")
    else:
        # Perform a single LU factorization of K_e
        with section("factorize"):
            lu = scipy.sparse.linalg.splu(K_e.tocsc())

        # Solve all load cases in one call (multi-RHS solve)
        with section("solve"):
            u_sol = lu.solve(F_stack)      # shape: (n_dof, n_loads)

    # Store the solutions in the provided array
    u_all[:, :] = u_sol

    return F_stack


def compute_compliance_basis_multi_load(
    basis: skfem.CellBasis,
    free_dofs: np.ndarray,
    dirichlet_dofs: np.ndarray,
    force_list: list[np.ndarray],
    E0: float, Emin: float, p: float, nu0: float,
    rho: np.ndarray,
    u_all: np.ndarray,
    solver: SolverSelector | LinearSolverConfig = 'auto',
    solver_config: LinearSolverConfig | None = None,
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int = None,
    petsc_options: PETScOptions | None = None,
    timer=None,
) -> np.ndarray:
    """
    Compute compliance values for one or more load cases.

    Parameters
    ----------
    solver_config : LinearSolverConfig, optional
        Preferred solver configuration. When provided, it takes precedence
        over ``solver``, ``rtol``, ``maxiter``, and ``petsc_options``.
    solver, rtol, maxiter, petsc_options : optional
        Legacy convenience inputs retained for backward compatibility.
    """

    solver_cfg = (
        solver_config if solver_config is not None else
        normalize_linear_solver_config(
            solver, rtol=rtol, maxiter=maxiter, petsc_options=petsc_options
        )
    )

    # Single-load: use the iterative-capable path and keep timer granularity.
    if len(force_list) == 1:
        compliance, u = compute_compliance_basis(
            basis, free_dofs, dirichlet_dofs, force_list[0],
            E0, Emin, p, nu0,
            rho,
            elem_func=elem_func,
            solver_config=solver_cfg,
            timer=timer,
        )
        u_all[:, 0] = u
        return np.array([compliance])

    # Multi-load currently supports only direct LU factorization.
    F_stack = solve_multi_load(
        basis, free_dofs, dirichlet_dofs, force_list,
        E0, Emin, p, nu0,
        rho,
        u_all,
        solver_config=solver_cfg,
        elem_func=elem_func,
        timer=timer,
    )

    if F_stack.ndim == 1:
        F_stack = F_stack[:, None]
    if u_all.ndim == 1:
        u_view = u_all[:, None]
    else:
        u_view = u_all

    # compliance for each load: fᵢ · uᵢ
    compliance_each = np.einsum('ij,ij->j', F_stack, u_view)

    return compliance_each


@skfem.Functional
def _strain_energy_density_(w):
    grad = w['uh'].grad  # shape: (3, 3, nelems, nqp)
    symgrad = 0.5 * (grad + transpose(grad))  # same shape
    tr = trace(symgrad)
    I_mat = eye(tr, symgrad.shape[0])  # shape: (3, 3, nelems, nqp)
    # mu, lam の shape: (nqp, nelems) → transpose to (nelems, nqp)
    mu = w['mu_elem'].T  # shape: (nelems, nqp)
    lam = w['lam_elem'].T  # shape: (nelems, nqp)
    # reshape to enable broadcasting
    mu = mu[None, None, :, :]  # → shape (1, 1, nelems, nqp)
    lam = lam[None, None, :, :]  # same

    stress = 2. * mu * symgrad + lam * I_mat  # shape-compatible now
    return 0.5 * ddot(stress, symgrad)


def strain_energy_skfem(
    basis: skfem.Basis,
    rho: np.ndarray, u: np.ndarray,
    E0: float, Emin: float, p: float, nu: float,
    elem_func: Callable = composer.simp_interpolation
) -> np.ndarray:
    uh = basis.interpolate(u)
    E_elem = elem_func(rho, E0, Emin, p)
    # shape: (nelements,)
    lam_elem, mu_elem = lame_parameters(E_elem, nu)
    n_qp = basis.X.shape[1]
    # shape: (n_qp, n_elements)
    lam_elem = np.tile(lam_elem, (n_qp, 1))
    mu_elem = np.tile(mu_elem, (n_qp, 1))
    elem_energy = _strain_energy_density_.elemental(
        basis, uh=uh, lam_elem=lam_elem, mu_elem=mu_elem
    )
    return elem_energy


def strain_energy_skfem_multi(
    basis: skfem.Basis,
    rho: np.ndarray,
    U: np.ndarray,  # shape: (n_dof, n_loads)
    E0: float, Emin: float, p: float, nu: float,
    elem_func: Callable = composer.simp_interpolation
) -> np.ndarray:
    """
    Compute strain energy density for multiple displacement fields.

    Returns:
        elem_energy_all: (n_elements, n_loads)
    """
    n_dof, n_loads = U.shape
    n_elements = basis.mesh.nelements

    E_elem = elem_func(rho, E0, Emin, p)
    lam_elem, mu_elem = lame_parameters(E_elem, nu)
    n_qp = basis.X.shape[1]
    lam_elem = np.tile(lam_elem, (n_qp, 1))  # (n_qp, n_elements)
    mu_elem = np.tile(mu_elem, (n_qp, 1))

    elem_energy_all = np.zeros((n_elements, n_loads))
    for i in range(n_loads):
        uh = basis.interpolate(U[:, i])  # scalar/vector field per load case
        elem_energy = _strain_energy_density_.elemental(
            basis, uh=uh, lam_elem=lam_elem, mu_elem=mu_elem
        )
        elem_energy_all[:, i] = elem_energy

    return elem_energy_all  # shape: (n_elements, n_loads)


class FEM_SimpLinearElasticity():
    """
    Finite Element solver for linear elasticity using SIMP interpolation.

    This class performs linear elastic FEM analysis where the Young's modulus
    is interpolated based on material density (ρ) using a SIMP-type
    interpolation function. It is intended for density-based topology
    optimization workflows, where element stiffness is expressed as:

        E(ρ) = E_min + (E_max - E_min) * f(ρ)

    where `f(ρ)` is typically ρᵖ for SIMP.

    Parameters
    ----------
    task : LinearElasticity
        Predefined linear elasticity problem that includes mesh, material
        constants (E, ν), boundary conditions, load vectors, and basis
        definitions.
    E_min_coeff : float
        Ratio defining the minimum Young's modulus as:
            E_min = task.E * E_min_coeff
        Used to avoid singular stiffness matrices during optimization.
    density_interpolation : Callable, optional
        A function f(ρ) that returns an interpolated stiffness multiplier.
        Defaults to `composer.simp_interpolation` (ρᵖ). Any custom
        interpolation function following SIMP/RAMP/etc. can be used.
    solver_config : LinearSolverConfig, optional
        Preferred normalized solver configuration.
    solver_option : {"spsolve", "cg_pyamg", "petsc", "petsc_spdirect"}, optional
        Legacy convenience backend selector used only when
        ``solver_config`` is not provided.
        - "spsolve": direct SciPy sparse solver (robust, slower for large DOF)
        - "cg_pyamg": Conjugate Gradient with PyAMG multigrid preconditioner
            (fast for large problems)
        - "petsc": PETSc CG + GAMG (requires petsc4py; supports multi-load
          by reusing the same KSP across RHS vectors)
        - "petsc_spdirect": PETSc sparse direct solve via KSP/PC
          preonly+lu

    Attributes
    ----------
    task : LinearElasticity
        The underlying elasticity problem definition.
    E_max : float
        Maximum Young's modulus (equal to `task.E`).
    E_min : float
        Minimum Young's modulus used for void regions.
    density_interpolation : Callable
        The SIMP / RAMP interpolation function used to compute material
        stiffness.
    solver_option : str
        Selected linear solver backend derived from ``solver_config``.
    petsc_options : PETScOptions or None
        PETSc-specific options derived from ``solver_config``.
    solver_config : LinearSolverConfig
        Normalized linear solver configuration used for state solves.

    Notes
    -----
    - This class does **not** update densities; it only evaluates the FEM
        response for a given density field.
    - Designed to integrate with OC/MMA/ADMM-based topology optimization
        frameworks.
    - The stiffness matrix assembly depends on interpolated Young's modulus
        at each element.
    - `E_min_coeff` should typically be small (1e−3 ~ 1e−9), but not zero.

    Examples
    --------
    >>> fem = FEM_SimpLinearElasticity(
    ...     task=my_task,
    ...     E_min_coeff=1e-3,
    ...     density_interpolation=composer.simp_interpolation,
    ...     solver_config=LinearSolverConfig(solver="spsolve"),
    ... )
    >>> u = fem.objectives_multi_load(rho)  # FEM compliaance given density
    """

    def __init__(
        self, task: "LinearElasticity",
        E_min_coeff: float,
        density_interpolation: Callable = composer.simp_interpolation,
        solver_config: LinearSolverConfig | None = None,
        solver_option: Literal[
            "spsolve", "cg_pyamg", "petsc", "petsc_spdirect"
        ] = "spsolve",
        petsc_options: PETScOptions | None = None,
    ):
        self.task = task
        self.E_max = task.E * 1.0
        self.E_min = task.E * E_min_coeff
        self.density_interpolation = density_interpolation
        self.solver_config = (
            solver_config if solver_config is not None else
            normalize_linear_solver_config(
                solver_option,
                petsc_options=petsc_options,
            )
        )
        self.solver_option = self.solver_config.solver
        self.petsc_options = self.solver_config.petsc_options

    def objectives_multi_load(
        self,
        rho: np.ndarray, p: float,
        u_dofs: np.ndarray,
        timer=None,
        force_scale: float = 1.0
    ) -> np.ndarray:

        force_list = self.task.neumann_linear if isinstance(
            self.task.neumann_linear, list
        ) else [self.task.neumann_linear]
        force_list = [f * force_scale for f in force_list]

        compliance_array = compute_compliance_basis_multi_load(
            self.task.basis, self.task.free_dofs, self.task.dirichlet_dofs,
            force_list,
            self.E_max, self.E_min, p, self.task.nu,
            rho,
            u_dofs,
            elem_func=self.density_interpolation,
            solver_config=self.solver_config,
            timer=timer,
        )
        return compliance_array

    def energy_multi_load(
        self,
        rho: np.ndarray, p: float, u_dofs: np.ndarray
    ) -> np.ndarray:
        return strain_energy_skfem_multi(
            self.task.basis, rho, u_dofs,
            self.E_max, self.E_min, p, self.task.nu,
            elem_func=self.density_interpolation
        )
