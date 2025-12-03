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


def solve_scipy_heat_multi_enforce(
    K_csr: scipy.sparse.csr_matrix,
    emit: np.ndarray,
    dirichlet_dofs_list: list[skfem.Dofs],
    dirichlet_values_list: list[float],
    u_all: np.ndarray,
) -> None:
    """
    General multi-load heat solver, no joblib.

    - For each load i, applies enforce(K, emit, D_i, x_i) and solves.
    - Does NOT assume all Dirichlet DOFs are identical across loads.
    """

    n_loads = len(dirichlet_values_list)
    assert u_all.shape[1] == n_loads

    for i in range(n_loads):
        D_i = dirichlet_dofs_list[i]
        val_i = dirichlet_values_list[i]

        K_e_i, f_e_i = skfem.enforce(
            K_csr, emit,
            D=D_i,
            x=np.full(D_i.N, val_i),
        )
        lu_i = scipy.sparse.linalg.splu(K_e_i.tocsc())
        u_all[:, i] = lu_i.solve(f_e_i)


def solve_scipy(
    K_csr,
    emit,
    dirichlet_dofs_list,
    dirichlet_values_list,
    u_all,
):
    """
    Multi-load heat solver (no joblib, enforce-based).

    Assumptions
    -----------
    - All load cases share the *same* Dirichlet DOF set.
      (i.e., dirichlet_dofs_list[i] are identical objects)
    - Only the Dirichlet *values* differ across load cases.
    - The RHS contribution "emit" is common to all load cases.

    Method
    ------
    1. Build the enforced stiffness matrix K_e using the first load case.
       (K_e depends on the Dirichlet DOF set D, not on the actual values)
    2. Factorize K_e only once (LU).
    3. For each load case, build only the enforced RHS f_e
       using skfem.enforce(K, emit, D, x=value).
    4. Stack all RHS vectors column-wise into F_stack.
    5. Solve all RHS simultaneously using lu.solve(F_stack).
       This is significantly faster than solving each load separately.

    Parameters
    ----------
    K_csr : csr_matrix
        Global conduction matrix.
    emit : (n_dof,) ndarray
        Load / emission vector (common to all load cases).
    dirichlet_dofs_list : list of skfem.Dofs
        List of Dirichlet DOF sets (all must be identical).
    dirichlet_values_list : list of float
        Dirichlet temperature values for each load case.
    u_all : (n_dof, n_loads) ndarray
        Output array to store all temperature solutions.
    """
    n_loads = len(dirichlet_values_list)
    if n_loads == 0:
        return

    # Use the first Dirichlet DOF set as the representative
    D0 = dirichlet_dofs_list[0]
    val0 = dirichlet_values_list[0]

    # Build the enforced stiffness matrix K_e (depends on D, not on value)
    K_e, _ = skfem.enforce(
        K_csr,
        emit,
        D=D0,
        x=np.full(D0.N, val0),
    )

    # Construct each load case RHS and stack them
    F_cols = []
    for D_i, val_i in zip(dirichlet_dofs_list, dirichlet_values_list):
        # Only Dirichlet value differs; DOF set is the same
        _, f_e_i = skfem.enforce(
            K_csr,
            emit,
            D=D_i,
            x=np.full(D_i.N, val_i),
        )
        F_cols.append(f_e_i)

    # Column-wise stack => shape (n_dof, n_loads)
    F_stack = np.column_stack(F_cols)

    # Perform LU factorization only once
    lu = scipy.sparse.linalg.splu(K_e.tocsc())

    # Solve all load cases in one call (fastest)
    u_all[:, :] = lu.solve(F_stack)


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
    maxiter: int = None
) -> list:
    solver = 'spsolve' if solver == 'auto' else solver

    K = composer.assemble_conduction_matrix(
        basis, rho, k0, kmin, p, elem_func
    )

    if isinstance(robin_bilinear, scipy.sparse.csr_matrix):
        K = K + robin_bilinear
    elif isinstance(robin_bilinear, list):
        for loop in robin_bilinear:
            K += loop

    emit = np.zeros(K.shape[0])
    if isinstance(robin_linear, np.ndarray):
        emit = robin_linear
    elif isinstance(robin_linear, list):
        for loop in robin_linear:
            emit += loop

    K_csr = K.tocsr()
    u_all[:, :] = 0.0

    dirichlet_dofs_list = [
        basis.get_dofs(nodes=loop) for loop in dirichlet_nodes_list
    ]

    if solver == "spsolve":
        solve_scipy(
            K_csr, emit,
            dirichlet_dofs_list, dirichlet_values_list,
            u_all
        )
    else:
        raise NotImplementedError("")

    return K_csr, emit, dirichlet_dofs_list


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
def _heat_exchange_grad_density_(w):
    """
    Element-wise contribution to the heat-exchange objective gradient
    coming from the PDE (conduction) part, i.e. proportional to
        ∂k/∂ρ * ∇T · ∇λ.

    Note:
        This captures only the implicit dependence of J on ρ via the state
        equation (K(ρ) T = f). The explicit dependence of J on ρ through
        h_eff(ρ) and |∇ρ| should be added separately if needed.
    """
    gradT = w['Th'].grad   # ∇T
    gradL = w['λh'].grad   # ∇λ
    return dot(gradT, gradL)


def heat_exchange_grad_density_multi(
    basis: skfem.Basis,
    T_all: np.ndarray,    # shape: (n_dof, n_loads), state field(s)
    λ_all: np.ndarray,    # shape: (n_dof, n_loads), adjoint field(s)
) -> np.ndarray:
    """
    Compute elementwise gradient density for the heat-exchange objective
    (PDE contribution) for multiple load cases.

    Parameters
    ----------
    basis : skfem.Basis
        FEM basis used for the conduction problem.
    T_all : (n_dof, n_loads) ndarray
        State temperatures for each load case (columns).
    λ_all : (n_dof, n_loads) ndarray
        Adjoint fields corresponding to the heat-exchange objective.

    Returns
    -------
    elem_grad_all : (n_elem, n_loads) ndarray
        Elementwise gradient densities for each load case. These are
        proportional to ∇T · ∇λ and should be combined with the SIMP
        derivative ∂k/∂ρ and any filter/projection derivatives to obtain
        dJ/dρ on the design variables.
    """
    n_dof, n_loads = T_all.shape
    n_elem = basis.mesh.nelements

    elem_grad_all = np.zeros((n_elem, n_loads))
    for i in range(n_loads):
        Th = basis.interpolate(T_all[:, i])
        λh = basis.interpolate(λ_all[:, i])
        elem_grad = _heat_exchange_grad_density_.elemental(
            basis, Th=Th, λh=λh
        )
        elem_grad_all[:, i] = elem_grad

    return elem_grad_all


@skfem.Functional
def _hx_den_density_(w):
    """
    Denominator: ∫_Γh dΓ ≈ ∫_Ω |∇ρ| dΩ
    """
    grad_rho = w['rho'].grad
    interface = np.sqrt(np.sum(grad_rho**2, axis=0))
    return interface


@skfem.LinearForm
def _hx_adj_rhs_(v, w):
    """
    Right-hand side density for the adjoint equation of the
    heat-exchange objective.

        dJ/dT = (- T_env * h_eff * |∇ρ| / J_den)

    This LinearForm integrates
        - T_env * h_eff * |∇ρ| * v
    before dividing by Jden
    """
    T_env = w['T_env']
    h_eff = w['h_eff']
    grad_rho = w['rho'].grad
    interface = np.sqrt(np.sum(grad_rho**2, axis=0))
    return - T_env * h_eff * interface * v


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


@skfem.Functional
def _avg_temp_density_(w):
    """
    Element-wise density for the average-temperature objective:
        J = ∫_Ω (T - T_env) dΩ

    Up to a constant shift, this is equivalent to ∫_Ω T dΩ,
    so it induces the same sensitivity with respect to ρ.
    """
    Th = w['Th']       # interpolated temperature
    T_env = w['T_env'] # ambient temperature field
    return Th - T_env


def avg_temp_skfem(
    basis: skfem.Basis,
    T: np.ndarray,
    T_env: float
) -> np.ndarray:
    """
    Compute elementwise contributions to the average temperature functional:
        J = ∫_Ω (T(x) - T_env) dΩ

    Parameters
    ----------
    basis : skfem.Basis
        FEM basis.
    T : (n_dof,) ndarray
        Nodal temperature vector.
    T_env : float
        Ambient/reference temperature.

    Returns
    -------
    elem_avgT : (n_elem,) ndarray
        Elementwise contributions to the average-temperature functional.
    """
    Th = basis.interpolate(T)
    T_env_field = basis.interpolate(np.full(basis.N, T_env))

    elem_avgT = _avg_temp_density_.elemental(
        basis, Th=Th, T_env=T_env_field
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

    Parameters
    ----------
    basis : skfem.Basis
        FEM basis.
    T_all : (n_dof, n_loads) ndarray
        Nodal temperatures for each load case (columns).
    T_env : float
        Ambient/reference temperature.

    Returns
    -------
    elem_avg_all : (n_elem, n_loads) ndarray
        Elementwise contributions to the average-temperature functional
        for each load case.
    """
    n_dof, n_loads = T_all.shape
    n_elem = basis.mesh.nelements
    elem_avg_all = np.zeros((n_elem, n_loads))

    T_env_field = basis.interpolate(np.full(basis.N, T_env))

    for i in range(n_loads):
        Th = basis.interpolate(T_all[:, i])
        elem_avg = _avg_temp_density_.elemental(
            basis, Th=Th, T_env=T_env_field
        )
        elem_avg_all[:, i] = elem_avg

    return elem_avg_all


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
        q: int = 4
    ):
        self.task = task
        self.k_max = task.k * 1.0
        self.k_min = task.k * E_min_coeff
        self.density_interpolation = density_interpolation
        self.solver_option = solver_option
        # self.n_joblib = n_joblib
        self.λ_all = None
        self.q = q

    def objectives_multi_load(
        self,
        rho: np.ndarray, p: float, u_dofs: np.ndarray, timer=None
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

        K_csr, emit, dirichlet_dofs_list = solve_multi_load(
            basis, self.task.free_dofs,
            dirichlet_nodes_list, dirichlet_values_list,
            self.task.robin_bilinear+[K_virtual],
            self.task.robin_linear+[f_virtual],
            self.k_max, self.k_min, p,
            rho,
            u_dofs,
            solver=self.solver_option,
            elem_func=self.density_interpolation,
            # n_joblib=self.n_joblib,
        )

        n_loads = u_dofs.shape[1]
        J_list: list[float] = []
        λ_all: np.ndarray | None = None

        objective = self.task.objective

        if objective == "compliance":
            for i in range(n_loads):
                T_i = u_dofs[:, i]
                J_list.append(float(T_i @ (K_csr @ T_i)))
            λ_all = -2.0 * u_dofs

        elif objective == "heat_exchange":
            cbasis = skfem.CellBasis(basis.mesh, basis.elem)
            rhoh = cbasis.interpolate(rho_nodal)

            # J_den = ∫|∇ρ| dΩ
            den_e = _hx_den_density_.elemental(cbasis, rho=rhoh)
            J_den = den_e.sum()

            if J_den <= 1e-16:
                # the boundary is almost zero -> objective/grad = 0
                J_list = [0.0 for _ in range(n_loads)]
                λ_all = np.zeros_like(u_dofs)
                self.λ_all = λ_all
                return np.array(J_list)

            J_list: list[float] = list()
            for i in range(n_loads):
                T_i = u_dofs[:, i]
                J_i = heat_exchange_objective(
                    cbasis,
                    T_i,
                    rho_nodal,
                    self.task.robin_bc_value,      # T_env
                    self.task.robin_coefficient    # h
                )
                J_list.append(J_i)

            T_env_field = cbasis.interpolate(
                np.full(cbasis.N, self.task.robin_bc_value)
            )
            # h_eff_nodal = h * ρ^p (1-ρ)^q
            h = self.task.robin_coefficient
            h_eff_nodal = h * (rho_nodal**p) * (1.0 - rho_nodal)**self.q
            h_eff_field = cbasis.interpolate(h_eff_nodal)
            rhs_vec = _hx_adj_rhs_.assemble(
                cbasis, rho=rhoh, T_env=T_env_field, h_eff=h_eff_field
            )
            rhs_vec /= J_den  # ∂J/∂T

            # solve adjoint K λ = ∂J/∂T
            λ_all = np.zeros_like(u_dofs)
            solve_scipy(
                K_csr, rhs_vec,
                dirichlet_dofs_list, dirichlet_values_list,
                λ_all
            )

        elif objective == "averaged_temp":
            for i in range(n_loads):
                J_list.append(float(np.sum(u_dofs[:, i])))

            λ_all = np.zeros_like(u_dofs)
            ones_emit = np.ones_like(emit)
            solve_scipy(
                K_csr, ones_emit,
                dirichlet_dofs_list, dirichlet_values_list,
                λ_all
            )

        else:
            raise ValueError(f"Unknown objective: {objective}")

        self.λ_all = λ_all
        return np.array(J_list)

    def energy_multi_load(
        self,
        rho: np.ndarray, p: float, u_dofs: np.ndarray
    ) -> np.ndarray:
        if self.task.objective == "compliance":
            return heat_energy_skfem_multi(
                self.task.basis, rho, u_dofs,
                self.k_max, self.k_min, p,
                elem_func=self.density_interpolation
            )

        elif self.task.objective == "heat_exchange":
            if self.λ_all is None:
                raise RuntimeError("adjoint field λ_all is not computed.")
            return heat_exchange_grad_density_multi(
                self.task.basis,
                u_dofs,
                self.λ_all,
            )

        elif self.task.objective == "averaged_temp":
            if self.λ_all is None:
                raise RuntimeError("adjoint field λ_all is not computed.")
            return avg_temp_grad_density_multi(
                self.task.basis,
                u_dofs,
                self.λ_all
            )
        else:
            raise ValueError(f"Unknown objective: {self.task.objective}")
