from __future__ import annotations
from typing import Callable, Literal

import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

import skfem
from skfem.helpers import grad, dot
import pyamg
# import joblib

# from sktopt.mesh import LinearHeatConduction
from sktopt.fea import composer
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


def solve_scipy(
    K_csr, emit,
    dirichlet_dofs_list, dirichlet_values_list,
    u_all,
    n_joblib: int = 1
):
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

    try:
        from joblib import Parallel, delayed, parallel_backend
        if n_joblib > 1:
            with parallel_backend("threading"):
                u_all[:, :] = np.column_stack(
                    Parallel(n_jobs=n_joblib)(
                        delayed(solve_system)(
                            dirichlet_dofs_list[i], dirichlet_values_list[i]
                        ) for i in range(len(dirichlet_values_list))
                    )
                )
            return

    except ModuleNotFoundError as e:
        logger.info(f"ModuleNotFoundError: {e}")
        n_joblib = -1

    u_all[:, :] = np.column_stack(
        [
            solve_system(
                dirichlet_dofs_list[i], dirichlet_values_list[i]
            ) for i in range(len(dirichlet_values_list))
        ]
    )


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
    n_joblib: int = 1,
    objective: Literal["compliance", "averaged_temp"] = "averaged_temp"
) -> list:
    solver = 'spsolve' if solver == 'auto' else solver
    # n_dof = basis.N
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

    if solver == "spsolve":
        solve_scipy(
            K_csr, emit,
            dirichlet_dofs_list, dirichlet_values_list,
            u_all, n_joblib=n_joblib
        )
    else:
        raise NotImplementedError("")

    objective_list = list()
    if objective == "compliance":
        for i in range(u_all.shape[1]):
            T_i = u_all[:, i]
            objective_list.append(float(T_i @ (K_csr @ T_i)))

        # λ_all = None
        λ_all = - 2 * u_all
    elif objective == "averaged_temp":
        objective_list = [
            float(np.sum(u_all[:, i])) for i in range(u_all.shape[1])
        ]
        λ_all = np.zeros_like(u_all)
        solve_scipy(
            K_csr, np.ones_like(emit),
            dirichlet_dofs_list, dirichlet_values_list,
            λ_all, n_joblib=n_joblib
        )
    else:
        raise ValueError("")
    return objective_list, λ_all


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
    n_joblib: int = 1,
    objective: Literal["compliance", "averaged_temp"] = "averaged_temp"
) -> list:
    solver = 'spsolve' if solver == 'auto' else solver
    # n_dof = basis.N
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

    if solver == "spsolve":
        solve_scipy(
            K_csr, emit,
            dirichlet_dofs_list, dirichlet_values_list,
            u_all, n_joblib=n_joblib
        )
    else:
        raise NotImplementedError("")
    return K_csr, emit, dirichlet_dofs_list


def compute_objective_multi_load(
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
    n_joblib: int = 1,
    objective: Literal["compliance", "averaged_temp"] = "averaged_temp"
) -> list:
    K_csr, emit, dirichlet_dofs_list = solve_multi_load(
        basis, free_dofs, dirichlet_nodes, dirichlet_values,
        robin_bilinear, robin_linear,
        E0, Emin, p,
        rho,
        u_all,
        solver=solver, elem_func=elem_func, rtol=rtol,
        maxiter=maxiter, n_joblib=n_joblib
    )

    n_loads = u_all.shape[1]
    objective_list: list[float] = []

    if objective == "compliance":
        # J_i = u_i^T K u_i
        for i in range(n_loads):
            T_i = u_all[:, i]
            objective_list.append(float(T_i @ (K_csr @ T_i)))
        λ_all = -2.0 * u_all

    elif objective == "averaged_temp":
        # J_i = sum_j T_{i,j}
        for i in range(n_loads):
            objective_list.append(float(np.sum(u_all[:, i])))

        # 随伴場 λ を解く場合（平均温度目的用）
        λ_all = np.zeros_like(u_all)
        ones_emit = np.ones_like(emit)
        solve_scipy(
            K_csr, ones_emit,
            dirichlet_dofs_list, dirichlet_values,
            λ_all, n_joblib=n_joblib
        )

    else:
        raise ValueError(f"Unknown objective: {objective}")

    return objective_list, λ_all



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


@skfem.Functional
def _hx_num_density_(w):
    """
    Numerator of heat-exchange objective:
        J_num = -∫_Γh T_env * h_eff * (T - T_env) dΓ
    approximated via:
        -∫_Ω T_env * h_eff * (Th - T_env) * |∇ρ| dΩ
    """
    Th = w['Th']                # Temperature field
    T_env = w['T_env']          # Ambient temperature
    h_eff = w['h_eff']          # Effective Robin coefficient

    grad_rho = w['rho'].grad
    interface = np.sqrt(np.sum(grad_rho**2, axis=0))

    return -T_env * h_eff * (Th - T_env) * interface


@skfem.Functional
def _hx_den_density_(w):
    """
    Denominator: ∫_Γh dΓ ≈ ∫_Ω |∇ρ| dΩ
    """
    grad_rho = w['rho'].grad
    interface = np.sqrt(np.sum(grad_rho**2, axis=0))
    return interface


def heat_exchange_objective(
    cbasis: skfem.CellBasis,
    T: np.ndarray,        # nodal temperature
    rho_nodal: np.ndarray,
    T_env: float,
    h_eff_scalar: float,  # 基本は self.task.robin_coefficient を入れる
) -> float:
    """
    Heat-exchange objective:
        J = J_num / J_den
    """

    Th = cbasis.interpolate(T)
    rhoh = cbasis.interpolate(rho_nodal)

    T_env_field = cbasis.interpolate(np.full(cbasis.N, T_env))
    h_eff_field = cbasis.interpolate(np.full(cbasis.N, h_eff_scalar))

    num_e = _hx_num_density_.elemental(
        cbasis, Th=Th, rho=rhoh, T_env=T_env_field, h_eff=h_eff_field
    )
    den_e = _hx_den_density_.elemental(cbasis, rho=rhoh)

    J_num = num_e.sum()
    J_den = den_e.sum()

    if J_den <= 1e-16:
        return 0.0

    return J_num / J_den


def avg_temp_skfem(
    basis: skfem.Basis,
    T: np.ndarray,
    T_env: float
) -> np.ndarray:
    """
    Compute elementwise contributions to the average temperature functional:
        J = ∫_Ω T(x) dΩ
    """
    Th = basis.interpolate(T)
    # elem_avgT = _avg_temp_density_.elemental(
    #     basis, Th=Th
    # )
    elem_avgT = _hx_boundary_density_.elemental(
        basis, Th=Th, T_env=T_env
    )
    return elem_avgT


def avg_temp_skfem_multi(
    basis: skfem.Basis,
    T_all: np.ndarray,
    T_env: float
) -> np.ndarray:
    """
    Compute elementwise average temperature functional for multiple load cases.
    Each column of T_all corresponds to one load condition.
    """
    n_dof, n_loads = T_all.shape
    n_elem = basis.mesh.nelements
    elem_energy_all = np.zeros((n_elem, n_loads))
    for i in range(n_loads):
        Th = basis.interpolate(T_all[:, i])
        elem_energy = _avg_temp_density_.elemental(
            basis, Th=Th, T_env=T_env
        )
        elem_energy_all[:, i] = elem_energy
    return elem_energy_all


@skfem.Functional
def _avg_temp_grad_density_(w):
    gradT = w['Th'].grad
    gradL = w['λh'].grad
    return dot(gradT, gradL)


def avg_temp_grad_density_multi(
    basis: skfem.Basis,
    T_all: np.ndarray,
    λ_all: np.ndarray,  # adjoint field(s)
) -> np.ndarray:
    n_dof, n_loads = T_all.shape
    n_elem = basis.mesh.nelements
    elem_energy_all = np.zeros((n_elem, n_loads))
    for i in range(n_loads):
        Th = basis.interpolate(T_all[:, i])
        λh = basis.interpolate(λ_all[:, i])
        elem_energy = _avg_temp_grad_density_.elemental(basis, Th=Th, λh=λh)
        elem_energy_all[:, i] = elem_energy
    return elem_energy_all


def get_robin_virtual(
    h: float, T_env: float,
    p: float, q: float
):
    @skfem.BilinearForm
    def robin_virtual_bilinear(u, v, w):
        rho = w['rho']
        grad_rho = w['rho'].grad
        interface = np.sqrt(np.sum(grad_rho**2, axis=0))
        h_eff = h * rho**p * (1 - rho)**q
        return h_eff * interface * u * v

    @skfem.LinearForm
    def robin_virtual_linear(v, w):
        rho = w['rho']
        grad_rho = w['rho'].grad
        interface = np.sqrt(np.sum(grad_rho**2, axis=0))
        h_eff = h * rho**p * (1 - rho)**q
        return h_eff * interface * T_env * v  # ← T_env*v

    return robin_virtual_bilinear, robin_virtual_linear


class FEM_SimpLinearHeatConduction():
    """
    Finite element solver for SIMP-based linear heat conduction problems
    with support for multi-load objectives and virtual Robin boundaries.

    This class evaluates thermal objectives (e.g., average temperature,
    thermal compliance, or user-defined metrics) for a given density field
    using SIMP interpolation of the conductivity.
    It also assembles additional "virtual" Robin boundaries, which enables
    topology optimization of boundary-dependent heat-transfer behavior.

    Parameters
    ----------
    task : LinearHeatConduction
        Problem configuration containing mesh, basis, boundary conditions,
        load cases, and objective type.
    E_min_coeff : float
        Minimum conductivity ratio used in the SIMP model. The actual minimum
        conductivity is ``task.k * E_min_coeff``.
    density_interpolation : Callable, optional
        Interpolation function for SIMP (or RAMP) mapping ρ → k(ρ).
        Defaults to ``composer.simp_interpolation``.
    solver_option : {"spsolve", "cg_pyamg"}, optional
        Linear solver to use for the state equation of each load case.
    q : int, optional
        Exponent for boundary interpolation in the virtual Robin model
        (often used to sharpen on/off behavior of boundary heat transfer).
    n_joblib : int, optional
        Number of parallel jobs for solving multiple load cases.

    Attributes
    ----------
    k_max : float
        Conductivity of solid material.
    k_min : float
        Conductivity of void (or weak) material.
    λ_all : np.ndarray or None
        Stored adjoint fields for all load cases, computed during the last
        call to :meth:`objectives_multi_load`.

    Notes
    -----
    - The class converts element-wise densities into nodal densities by
      averaging over connected elements. This ensures smoother interpolation
      when assembling virtual Robin terms.
    - When ``task.design_robin_boundary`` is True, the actual boundary
      condition is updated based on the given density field, enabling
      optimization over Robin boundaries.
    - For multi-load problems, each load case is solved independently and
      the objectives are returned as a list (e.g., one compliance value per
      thermal load case).

    """

    def __init__(
        self, task: "LinearHeatConduction",
        E_min_coeff: float,
        density_interpolation: Callable = composer.simp_interpolation,
        solver_option: Literal["spsolve"] = "spsolve",
        q: int = 4,
        n_joblib: int = 1
    ):
        self.task = task
        self.k_max = task.k * 1.0
        self.k_min = task.k * E_min_coeff
        self.density_interpolation = density_interpolation
        self.solver_option = solver_option
        self.n_joblib = n_joblib
        self.λ_all = None
        self.q = q

    def objectives_multi_load(
        self,
        rho: np.ndarray, p: float, u_dofs: np.ndarray
    ) -> np.ndarray:
        dirichlet_nodes_list = self.task.dirichlet_nodes if isinstance(
            self.task.dirichlet_nodes, list
        ) else [self.task.dirichlet_nodes]
        dirichlet_values_list = self.task.dirichlet_values if isinstance(
            self.task.dirichlet_values, list
        ) else [self.task.dirichlet_values]

        robin_virtual_bilinear, robin_virtual_linear = get_robin_virtual(
            self.task.robin_coefficient, self.task.robin_bc_value,
            p, self.q
        )
        basis = self.task.basis

        t = basis.mesh.t
        nvert = basis.mesh.nvertices
        k = t.shape[0]

        rho_nodal = np.zeros(nvert, dtype=float)
        count = np.zeros(nvert, dtype=float)
        np.add.at(rho_nodal, t.ravel(), np.repeat(rho, k))
        np.add.at(count,     t.ravel(), 1)
        rho_nodal /= np.maximum(count, 1.0)
        # to fem field
        rho_field = basis.interpolate(rho_nodal)
        K_virtual = robin_virtual_bilinear.assemble(
            self.task.basis, rho=rho_field
        )
        f_virtual = robin_virtual_linear.assemble(
            self.task.basis, rho=rho_field
        )

        if self.task.design_robin_boundary is True:
            self.task.update_robin_bc(rho, p)
        # else:
        #     robin_bilinear = self.task.robin_bilinear
        #     robin_linear = self.task.robin_linear

        objective_list, λ_all = compute_objective_multi_load(
            self.task.basis, self.task.free_dofs,
            dirichlet_nodes_list,
            dirichlet_values_list,
            self.task.robin_bilinear+[K_virtual],
            self.task.robin_linear+[f_virtual],
            self.k_max, self.k_min, p,
            rho,
            u_dofs,
            elem_func=self.density_interpolation,
            solver=self.solver_option,
            n_joblib=self.n_joblib,
            objective=self.task.objective
            # objective=(
            #     "averaged_temp"
            #     if self.task.objective == "heat_exchange"
            #     else self.task.objective
            # )
        )

        if self.task.objective == "heat_exchange":
            cbasis = skfem.CellBasis(
                self.task.basis.mesh,
                self.task.basis.elem
            )
            J_list = []
            for i in range(u_dofs.shape[1]):
                T_i = u_dofs[:, i]
                J_i = heat_exchange_objective(
                    cbasis,
                    T_i,
                    rho_nodal,
                    self.task.robin_bc_value,      # T_env
                    self.task.robin_coefficient    # h (or h_eff scalar)
                )
                J_list.append(J_i)
            objective_list = J_list

        self.λ_all = λ_all
        return np.array(objective_list)

    def energy_multi_load(
        self,
        rho: np.ndarray, p: float, u_dofs: np.ndarray
    ) -> np.ndarray:
        # if self.task.objective == "compliance":
        return heat_energy_skfem_multi(
            self.task.basis, rho, u_dofs,
            self.k_max, self.k_min, p,
            elem_func=self.density_interpolation
        )
        # elif self.task.objective == "averaged_temp":
        #     return avg_temp_grad_density_multi(
        #         self.task.basis,
        #         u_dofs,
        #         self.λ_all
        #     )
