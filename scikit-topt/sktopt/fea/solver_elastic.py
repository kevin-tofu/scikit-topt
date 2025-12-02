from __future__ import annotations
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
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


def solve_u(
    K_cond: scipy.sparse.spmatrix,
    F_cond: np.ndarray,
    chosen_solver: Literal['cg_jacobi', 'spsolve', 'cg_pyamg'] = 'spsolve',
    rtol: float = 1e-8,
    maxiter: int = None,
) -> np.ndarray:
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

        else:
            raise ValueError(f"Unknown solver: {chosen_solver}")

    except Exception as e:
        logger.warning(f"Solver exception - {e}, falling back to spsolve.")
        u_c = scipy.sparse.linalg.spsolve(K_cond, F_cond)

    return u_c


def compute_compliance_basis(
    basis, free_dofs, dirichlet_dofs, force,
    E0, Emin, p, nu0,
    rho,
    elem_func: Callable = composer.simp_interpolation,
    solver: Literal['auto', 'cg_jacobi', 'spsolve', 'cg_pyamg'] = 'auto',
    rtol: float = 1e-5,
    maxiter: int = None,
    timer=None,
) -> tuple:
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
    if solver == 'auto':
        if n_dof < 1000:
            chosen_solver = 'spsolve'
        elif n_dof < 30000:
            # chosen_solver = 'cg_jacobi'
            chosen_solver = 'cg_pyamg'
        else:
            chosen_solver = 'cg_pyamg'
            # chosen_solver = 'cg_jacobi'
    else:
        chosen_solver = solver

    _maxiter = min(1000, max(300, n_dof // 5)) if maxiter is None else maxiter
    K_csr = K.tocsr()
    all_dofs = np.arange(K_csr.shape[0])
    free_dofs = np.setdiff1d(all_dofs, dirichlet_dofs, assume_unique=True)

    # enforce
    with section("enforce_bc"):
        K_e, F_e = skfem.enforce(K_csr, force, D=dirichlet_dofs)
    with section("solve"):
        u = solve_u(
            K_e, F_e, chosen_solver=chosen_solver,
            rtol=rtol, maxiter=_maxiter
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
    solver: Literal['auto', 'cg_jacobi', 'spsolve', 'cg_pyamg'] = 'auto',
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int | None = None,
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
    solver : {'auto', 'cg_jacobi', 'spsolve', 'cg_pyamg'}, optional
        Solver selector. For multi-load, only 'auto' and 'spsolve' are supported.
        - 'auto' or 'spsolve': direct LU factorization (splu) is used.
        - 'cg_jacobi', 'cg_pyamg': currently not implemented for multi-load.
    elem_func : Callable, optional
        Density interpolation function, e.g. SIMP or RAMP.
    rtol : float, optional
        Relative tolerance (unused for direct LU; kept for API compatibility).
    maxiter : int or None, optional
        Maximum number of iterations (unused for direct LU).

    Returns
    -------
    F_stack : (n_dof, n_loads) ndarray
        Stack of enforced right-hand sides for each load case.
    """

    # fall back to single-load path to honor chosen solver/timer
    if len(force_list) == 1:
        compliance, u = compute_compliance_basis(
            basis, free_dofs, dirichlet_dofs, force_list[0],
            E0, Emin, p, nu0,
            rho,
            elem_func=elem_func,
            solver=solver,
            rtol=rtol,
            maxiter=maxiter,
            timer=timer,
        )
        u_all[:, 0] = u
        return np.array([compliance])

    # Multi-load currently supports only direct LU factorization.
    if solver not in ('auto', 'spsolve'):
        raise NotImplementedError(
            "solve_multi_load currently supports only direct LU (solver "
            "='auto' or 'spsolve'). Iterative solvers for multi-load are "
            "not implemented."
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
    solver: Literal['auto', 'cg_jacobi', 'spsolve', 'cg_pyamg'] = 'auto',
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int = None,
    timer=None,
) -> np.ndarray:

    # Single-load: use the iterative-capable path and keep timer granularity.
    if len(force_list) == 1:
        compliance, u = compute_compliance_basis(
            basis, free_dofs, dirichlet_dofs, force_list[0],
            E0, Emin, p, nu0,
            rho,
            elem_func=elem_func,
            solver=solver,
            rtol=rtol,
            maxiter=maxiter,
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
        solver=solver,
        elem_func=elem_func,
        rtol=rtol,
        maxiter=maxiter,
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


@Functional
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
    solver_option : {"spsolve", "cg_pyamg"}, optional
        Linear solver backend.
        - "spsolve": direct SciPy sparse solver (robust, slower for large DOF)
        - "cg_pyamg": Conjugate Gradient with PyAMG multigrid preconditioner
            (fast for large problems)

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
        Selected linear solver backend.

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
    ...     solver_option="spsolve",
    ... )
    >>> u = fem.objectives_multi_load(rho)  # FEM compliaance given density
    """

    def __init__(
        self, task: "LinearElasticity",
        E_min_coeff: float,
        density_interpolation: Callable = composer.simp_interpolation,
        solver_option: Literal["spsolve", "cg_pyamg"] = "spsolve",
    ):
        self.task = task
        self.E_max = task.E * 1.0
        self.E_min = task.E * E_min_coeff
        self.density_interpolation = density_interpolation
        self.solver_option = solver_option

    def objectives_multi_load(
        self,
        rho: np.ndarray, p: float,
        u_dofs: np.ndarray,
        timer=None
    ) -> np.ndarray:

        compliance_array = compute_compliance_basis_multi_load(
            self.task.basis, self.task.free_dofs, self.task.dirichlet_dofs,
            self.task.neumann_linear,
            self.E_max, self.E_min, p, self.task.nu,
            rho,
            u_dofs,
            elem_func=self.density_interpolation,
            solver=self.solver_option,
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
