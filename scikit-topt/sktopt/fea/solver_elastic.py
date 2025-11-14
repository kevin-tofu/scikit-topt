from __future__ import annotations
from typing import Callable, Literal

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


def compute_compliance_simp_basis(
    basis, free_dofs, dirichlet_dofs, force,
    E0, Emin, p, nu0,
    rho,
) -> tuple:
    K = composer.assemble_stiffness_matrix(
        basis, rho, E0,
        Emin, p, nu0
    )
    K_e, F_e = skfem.enforce(K, force, D=dirichlet_dofs)
    # u = scipy.sparse.linalg.spsolve(K_e, F_e)
    u = skfem.solve(K_e, F_e)
    f_free = force[free_dofs]
    compliance = f_free @ u[free_dofs]
    return (compliance, u)


def solve_u(
    K_cond: scipy.sparse.csc_matrix,
    F_cond: np.ndarray,
    chosen_solver: Literal['cg_jacobi', 'spsolve', 'cg_pyamg'] = 'auto',
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
        print(f"Solver exception - {e}, falling back to spsolve.")
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
) -> tuple:
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
    K_e, F_e = skfem.enforce(K_csr, force, D=dirichlet_dofs)
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
    maxiter: int = None,
    n_joblib: int = 1
) -> float:
    solver = 'spsolve' if solver == 'auto' else solver
    n_dof = basis.N
    assert u_all.shape == (n_dof, len(force_list))

    K = composer.assemble_stiffness_matrix(
        basis, rho, E0, Emin, p, nu0, elem_func
    )
    _maxiter = min(1000, max(300, n_dof // 5)) if maxiter is None else maxiter
    K_csr = K.tocsr()

    # all_dofs = np.arange(n_dof)
    # free_dofs = np.setdiff1d(all_dofs, dirichlet_dofs, assume_unique=True)
    # K_e = K_csr[free_dofs, :][:, free_dofs]
    # K_e, _ = skfem.enforce(
    #     K_csr[free_dofs, :][:, free_dofs], force_list[0],
    #     D=dirichlet_dofs
    # )
    # K_c, _, _, _ = skfem.condense(K_csr, force_list[0], D=dirichlet_dofs)
    K_e, _ = skfem.enforce(K_csr, force_list[0], D=dirichlet_dofs)
    F_stack = np.column_stack([
        skfem.enforce(K_csr, f, D=dirichlet_dofs)[1] for f in force_list
    ])
    compliance_total = 0.0
    u_all[:, :] = 0.0
    if solver == 'spsolve':
        try:
            from joblib import Parallel, delayed, parallel_backend
            if n_joblib > 1:
                lu = scipy.sparse.linalg.splu(K_e.tocsc())

                def solve_system(F_stack):
                    return lu.solve(F_stack)

                with parallel_backend("threading"):
                    u_all[:, :] = np.column_stack(
                        Parallel(n_jobs=n_joblib)(
                            delayed(solve_system)(F_stack[:, i]) for i in range(
                                F_stack.shape[1]
                            )
                        )
                    )
                return F_stack
        except ModuleNotFoundError as e:
            logger.info(f"ModuleNotFoundError: {e}")
            n_joblib = -1

        lu = scipy.sparse.linalg.splu(K_e.tocsc())
        u_all[:, :] = np.column_stack(
            [lu.solve(F_stack[:, i]) for i in range(F_stack.shape[1])]
        )

    else:
        # choose preconditioner if needed
        if solver == 'cg_jacobi':
            M_diag = K_e.diagonal()
            M_inv = 1.0 / M_diag
            M = LinearOperator(K_e.shape, matvec=lambda x: M_inv * x)
        elif solver == 'cg_pyamg':
            ml = pyamg.smoothed_aggregation_solver(K_e)
            M = ml.aspreconditioner()
        else:
            raise ValueError(f"Unknown solver: {solver}")

        for i, _ in enumerate(force_list):
            F_e = F_stack[:, i]
            # _, F_e = skfem.enforce(K_csr, force, D=dirichlet_dofs)
            u_e, info = scipy.sparse.linalg.cg(
                K_e, F_e, M=M, rtol=rtol, maxiter=_maxiter
            )
            if info != 0:
                logger.info(
                    f"[warning] \
                        CG did not converge for load case {i}: info = {info}"
                )
            u_all[:, i] = u_e
            # compliance_total += F_e[free_dofs] @ u_e[free_dofs]

    # compliance_total = np.sum(np.einsum('ij,ij->j', F_stack, u_all))
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
    n_joblib: int = 1
) -> np.ndarray:
    F_stack = solve_multi_load(
        basis, free_dofs, dirichlet_dofs, force_list,
        E0, Emin, p, nu0,
        rho,
        u_all,
        solver=solver, elem_func=elem_func, rtol=rtol,
        maxiter=maxiter, n_joblib=n_joblib
    )

    # compliance_total = np.sum(np.einsum('ij,ij->j', F_stack, u_all))
    compliance_each = np.einsum('ij,ij->j', F_stack, u_all)
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
    n_joblib : float, optional
        Number of parallel jobs for element-wise computations. If 1, no
        parallelization is used.

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
    n_joblib : float
        Number of parallel jobs for element computations.

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
        n_joblib: float = 1
    ):
        self.task = task
        self.E_max = task.E * 1.0
        self.E_min = task.E * E_min_coeff
        self.density_interpolation = density_interpolation
        self.solver_option = solver_option
        self.n_joblib = n_joblib

    def objectives_multi_load(
        self,
        rho: np.ndarray, p: float,
        u_dofs: np.ndarray
    ) -> np.ndarray:

        compliance_array = compute_compliance_basis_multi_load(
            self.task.basis, self.task.free_dofs, self.task.dirichlet_dofs,
            self.task.neumann_linear,
            self.E_max, self.E_min, p, self.task.nu,
            rho,
            u_dofs,
            elem_func=self.density_interpolation,
            solver=self.solver_option,
            n_joblib=self.n_joblib
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
