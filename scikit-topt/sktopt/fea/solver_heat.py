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

from sktopt.mesh import LinearHeatConduction
from sktopt.fea import composer
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


def solve_multi_load(
    basis: skfem.CellBasis,
    free_dofs: np.ndarray,
    dirichlet_dofs: np.ndarray,
    heat_source_list: list[np.ndarray],
    k0: float, kmin: float, p: float,
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
    assert u_all.shape == (n_dof, len(heat_source_list))

    K = composer.assemble_conduction_matrix(
        basis, rho, k0, kmin, p, elem_func
    )
    _maxiter = min(1000, max(300, n_dof // 5)) if maxiter is None else maxiter
    K_csr = K.tocsr()
    K_e, _ = skfem.enforce(K_csr, heat_source_list[0], D=dirichlet_dofs)
    heat_source_stack = np.column_stack([
        skfem.enforce(K_csr, f, D=dirichlet_dofs)[1] for f in heat_source_list
    ])
    # compliance_total = 0.0
    u_all[:, :] = 0.0
    if solver == 'spsolve':
        if n_joblib > 1:
            from joblib import Parallel, delayed, parallel_backend
            lu = scipy.sparse.linalg.splu(K_e.tocsc())

            def solve_system(heat_source_stack):
                return lu.solve(heat_source_stack)

            with parallel_backend("threading"):
                u_all[:, :] = np.column_stack(
                    Parallel(n_jobs=n_joblib)(
                        delayed(solve_system)(heat_source_stack[:, i]) for i in range(
                            heat_source_stack.shape[1]
                        )
                    )
                )

        else:
            lu = scipy.sparse.linalg.splu(K_e.tocsc())
            u_all[:, :] = np.column_stack(
                [lu.solve(heat_source_stack[:, i]) for i in range(heat_source_stack.shape[1])]
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

        for i, _ in enumerate(heat_source_list):
            F_e = heat_source_stack[:, i]
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
    return heat_source_stack


def compute_compliance_basis_multi_load(
    basis: skfem.CellBasis,
    free_dofs: np.ndarray,
    dirichlet_dofs: np.ndarray,
    force_list: list[np.ndarray],
    E0: float, Emin: float, p: float,
    rho: np.ndarray,
    u_all: np.ndarray,
    solver: Literal['auto', 'cg_jacobi', 'spsolve', 'cg_pyamg'] = 'auto',
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int = None,
    n_joblib: int = 1
) -> np.ndarray:
    heat_source_stack = solve_multi_load(
        basis, free_dofs, dirichlet_dofs, force_list,
        E0, Emin, p,
        rho,
        u_all,
        solver=solver, elem_func=elem_func, rtol=rtol,
        maxiter=maxiter, n_joblib=n_joblib
    )

    # compliance_total = np.sum(np.einsum('ij,ij->j', F_stack, u_all))
    compliance_each = np.einsum('ij,ij->j', heat_source_stack, u_all)
    return compliance_each



def thermal_energy_skfem_multi(
    basis: skfem.Basis,
    rho: np.ndarray,
    T: np.ndarray,  # shape: (n_dof, n_loads)
    k0: float, kmin: float, p: float,
    elem_func: Callable,
) -> np.ndarray:
    """
    Compute element-wise thermal energy density for multiple temperature fields.

    Equivalent to 0.5 * ∫Ω k(ρ) |∇T|² dΩ per element.

    Parameters:
        basis : skfem.Basis
            Scalar FEM basis (e.g., ElementTriP1, ElementTetP1).
        rho : np.ndarray
            Element-wise density (0 = void, 1 = solid).
        T : np.ndarray
            Nodal temperature fields for multiple load cases, shape (n_dof, n_loads).
        k0, kmin : float
            Conductivity of solid and void material.
        p : float
            Penalization power (SIMP exponent).
        elem_func : Callable
            Material interpolation function, e.g., SIMP.

    Returns:
        elem_energy_all : np.ndarray
            Element-wise thermal energy for each load case, shape (n_elements, n_loads).
    """

    n_dof, n_loads = T.shape
    n_elements = basis.mesh.nelements

    # --- 1. Conductivity interpolation (SIMP)
    k_elem = elem_func(rho, k0, kmin, p)
    n_qp = basis.X.shape[1]
    k_elem = np.tile(k_elem, (n_qp, 1))  # (n_qp, n_elements)

    # --- 2. Initialize results
    elem_energy_all = np.zeros((n_elements, n_loads))

    # --- 3. Loop over load cases (temperature fields)
    for i in range(n_loads):
        Th = basis.interpolate(T[:, i])  # interpolate nodal temperatures
        # compute ∇T at quadrature points
        gradT = Th.grad  # shape (dim, n_qp, n_elements)
        # |∇T|² per quadrature point and element
        gradT_sq = np.sum(gradT**2, axis=0)  # (n_qp, n_elements)
        # 0.5 * k * |∇T|², integrated over element
        elem_energy = 0.5 * np.sum(k_elem * gradT_sq * basis.dx)  # scalar sum over quadrature
        elem_energy_all[:, i] = elem_energy

    return elem_energy_all  # shape: (n_elements, n_loads)


class FEM_SimpLinearHeatConduction():
    def __init__(
        self, task: LinearHeatConduction,
        E_min_coeff: float,
        density_interpolation: Callable = composer.simp_interpolation,
        solver_option: Literal["spsolve", "cg_pyamg"] = "spsolve",
        n_joblib: float = 1
    ):
        self.task = task
        self.k_max = task.k * 1.0
        self.k_min = task.k * E_min_coeff
        self.density_interpolation = density_interpolation
        self.solver_option = solver_option
        self.n_joblib = n_joblib

    def compute_compliance_multi_load(
        self,
        rho: np.ndarray, p: float, u_dofs: np.ndarray
    ) -> np.ndarray:

        compliance_array = compute_compliance_basis_multi_load(
            self.task.basis, self.task.free_dofs, self.task.dirichlet_dofs,
            self.task.neumann_linear,
            self.k_max, self.k_min, p,
            rho,
            u_dofs,
            elem_func=self.density_interpolation,
            solver=self.solver_option,
            n_joblib=self.n_joblib
        )
        return compliance_array

    def strain_energy_skfem_multi_load(
        self,
        rho: np.ndarray, p: float, u_dofs: np.ndarray
    ) -> np.ndarray:
        return thermal_energy_skfem_multi(
            self.task.basis, rho, u_dofs,
            self.k_max, self.k_min, p,
            elem_func=self.density_interpolation
        )
