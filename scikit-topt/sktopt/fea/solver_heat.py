from __future__ import annotations
from typing import Callable, Literal

import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

import skfem
from skfem.helpers import grad, dot
import pyamg

# from sktopt.mesh import LinearHeatConduction
from sktopt.fea import composer
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


def solve_multi_load(
    basis: skfem.Basis,
    free_dofs: np.ndarray,
    dirichlet_nodes_list: list[np.ndarray],
    dirichlet_values_list: list[float],
    robin_bilinear: scipy.sparse.csr_matrix | list[scipy.sparse.csr_matrix],
    robin_linear: np.ndarray | list[np.ndarray],
    k0: float, kmin: float, p: float,
    rho: np.ndarray,
    u_all: np.ndarray,
    solver: Literal['auto', 'spsolve'] = 'auto',
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int = None,
    n_joblib: int = 1
) -> float:
    solver = 'spsolve' if solver == 'auto' else solver
    n_dof = basis.N
    # assert u_all.shape == (n_dof, len(dirichlet_values))
    K = composer.assemble_conduction_matrix(
        basis, rho, k0, kmin, p, elem_func
    )
    if isinstance(robin_bilinear, scipy.sparse.csr_matrix):
        K = K + robin_bilinear
    elif isinstance(robin_bilinear, list):
        for loop in robin_bilinear:
            K += loop

    emit = np.zeros([K.shape[0]])
    if isinstance(robin_linear, np.ndarray):
        emit = robin_linear
    elif isinstance(robin_linear, list):
        for loop in robin_linear:
            emit += loop
    K_csr = K.tocsr()
    # compliance_total = 0.0
    u_all[:, :] = 0.0
    dirichlet_dofs_list = [
        basis.get_dofs(nodes=loop) for loop in dirichlet_nodes_list
    ]

    def solve_system(
        dirichlet_dofs_loop, dirichlet_values_loop
    ):
        T_sol = skfem.solve(
            *skfem.condense(
                K_csr, emit, D=dirichlet_dofs_loop,
                x=np.full(dirichlet_dofs_loop.N, dirichlet_values_loop)
            )
        )
        return T_sol

    if solver == 'spsolve':
        if n_joblib > 1:
            from joblib import Parallel, delayed, parallel_backend
            with parallel_backend("threading"):
                u_all[:, :] = np.column_stack(
                    Parallel(n_jobs=n_joblib)(
                        delayed(solve_system)(
                            dirichlet_dofs_list[i], dirichlet_values_list[i]
                        ) for i in range(len(dirichlet_values_list))
                    )
                )

        else:
            u_all[:, :] = np.column_stack(
                [
                    solve_system(
                        dirichlet_dofs_list[i], dirichlet_values_list[i]
                    ) for i in range(len(dirichlet_values_list))
                ]
            )

    compliance_list = list()
    for i in range(u_all.shape[1]):
        T_i = u_all[:, i]
        compliance_list.append(float(T_i @ (K_csr @ T_i)))
    return compliance_list


def compute_compliance_basis_multi_load(
    basis: skfem.CellBasis,
    free_dofs: np.ndarray,
    dirichlet_nodes: list[np.ndarray],
    dirichlet_values: list[float],
    robin_bilinear: np.ndarray,
    robin_linear: np.ndarray,
    E0: float, Emin: float, p: float,
    rho: np.ndarray,
    u_all: np.ndarray,
    solver: Literal['auto', 'cg_jacobi', 'spsolve', 'cg_pyamg'] = 'auto',
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int = None,
    n_joblib: int = 1
) -> np.ndarray:
    compliance_list = solve_multi_load(
        basis, free_dofs, dirichlet_nodes, dirichlet_values,
        robin_bilinear, robin_linear,
        E0, Emin, p,
        rho,
        u_all,
        solver=solver, elem_func=elem_func, rtol=rtol,
        maxiter=maxiter, n_joblib=n_joblib
    )
    return compliance_list


@skfem.Functional
def _heat_energy_density_(w):
    gradT = w['Th'].grad  # shape: (3, nqp, nelems)
    k_elem = w['k_elem'].T  # (nelems, nqp)
    k_elem = k_elem[None, :, :]  # (1, nqp, nelems)
    return k_elem * dot(gradT, gradT)


def heat_energy_skfem(
    basis: skfem.Basis,
    rho: np.ndarray,
    T: np.ndarray,
    k0: float, kmin: float, p: float,
    elem_func: Callable = composer.simp_interpolation
) -> np.ndarray:
    Th = basis.interpolate(T)
    k_elem = elem_func(rho, k0, kmin, p)  # shape: (n_elem,)
    n_qp = basis.X.shape[1]
    k_elem = np.tile(k_elem, (n_qp, 1))   # shape: (n_qp, n_elem)

    elem_energy = _heat_energy_density_.elemental(
        basis, Th=Th, k_elem=k_elem
    )
    return elem_energy


def heat_energy_skfem_multi(
    basis: skfem.Basis,
    rho: np.ndarray,
    T_all: np.ndarray,  # shape: (n_dof, n_loads)
    k0: float, kmin: float, p: float,
    elem_func: Callable = composer.simp_interpolation
) -> np.ndarray:
    n_dof, n_loads = T_all.shape
    n_elem = basis.mesh.nelements

    k_elem = elem_func(rho, k0, kmin, p)
    n_qp = basis.X.shape[1]
    k_elem = np.tile(k_elem, (n_qp, 1))

    elem_energy_all = np.zeros((n_elem, n_loads))
    for i in range(n_loads):
        Th = basis.interpolate(T_all[:, i])
        elem_energy = _heat_energy_density_.elemental(
            basis, Th=Th, k_elem=k_elem
        )
        elem_energy_all[:, i] = elem_energy
    return elem_energy_all


class FEM_SimpLinearHeatConduction():
    def __init__(
        self, task: "LinearHeatConduction",
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

    def compliance_multi_load(
        self,
        rho: np.ndarray, p: float, u_dofs: np.ndarray
    ) -> np.ndarray:
        dirichlet_nodes_list = self.task.dirichlet_nodes if isinstance(
            self.task.dirichlet_nodes, list
        ) else [self.task.dirichlet_nodes]
        dirichlet_values_list = self.task.dirichlet_values if isinstance(
            self.task.dirichlet_values, list
        ) else [self.task.dirichlet_values]

        compliance_list = compute_compliance_basis_multi_load(
            self.task.basis, self.task.free_dofs,
            dirichlet_nodes_list,
            dirichlet_values_list,
            self.task.robin_bilinear, self.task.robin_linear,
            self.k_max, self.k_min, p,
            rho,
            u_dofs,
            elem_func=self.density_interpolation,
            solver=self.solver_option,
            n_joblib=self.n_joblib
        )
        return np.array(compliance_list)

    def energy_multi_load(
        self,
        rho: np.ndarray, p: float, u_dofs: np.ndarray
    ) -> np.ndarray:
        return heat_energy_skfem_multi(
            self.task.basis, rho, u_dofs,
            self.k_max, self.k_min, p,
            elem_func=self.density_interpolation
        )
