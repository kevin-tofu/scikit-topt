from typing import Callable
from collections import defaultdict

import scipy
from numba import njit, prange

import skfem
from skfem import Basis, asm
from skfem.helpers import ddot, sym_grad, trace, eye
from skfem.assembly import BilinearForm
from skfem.helpers import transpose
from skfem.models.elasticity import lame_parameters
from skfem import Functional
# from skfem import asm, LinearForm

import numpy as np


@njit
def simp_interpolation(rho, E0, Emin, p):
    E_elem = Emin + (E0 - Emin) * (rho ** p)
    return E_elem


@njit
def ramp_interpolation(rho, E0, Emin, p):
    """
    ram: E(rho) = Emin + (E0 - Emin) * [rho / (1 + p(1 - rho))]
    Parameters:
      rho  : array of densities in [0,1]
      E0   : maximum Young's modulus
      Emin : minimum Young's modulus
      p    : ram parameter
    Returns:
      array of element-wise Young's moduli
    """
    # avoid division by zero
    E_elem = Emin + (E0 - Emin) * (rho / (1.0 + p*(1.0 - rho)))
    return E_elem


simp_interpolation_numba = simp_interpolation
ramp_interpolation_numba = ramp_interpolation


@njit
def lam_mu(E, nu):
    lam = (nu * E) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def assemble_stiffness_matrix(
    basis: Basis,
    rho: np.ndarray,
    E0: float, Emin: float, p: float, nu: float,
    elem_func: Callable = simp_interpolation
):
    """
    Assemble the global stiffness matrix
    for 3D linear elasticity with SIMP material interpolation.

    Parameters:
        basis : skfem Basis for the mesh
                (built with ElementVector(ElementTetP1) on MeshTet).
        rho   : 1D array of length n_elements
                with density values for each element.
        E0    : Young's modulus of solid material (for rho = 1).
        Emin  : Minimum Young's modulus for void material
                (for rho = 0, ensures numerical stability).
        p     : Penalization power for SIMP
                (typically >= 1, e.g., 3 for standard topology optimization).
        nu    : Poisson's ratio (assumed constant for all elements).

    Returns:
        Sparse stiffness matrix (scipy.sparse.csr_matrix) assembled
        for the given density distribution.
    """
    # 1. Compute Young's modulus for each element using SIMP / RAMP
    E_elem = elem_func(rho, E0, Emin, p)  # array of size [n_elements]
    # 2. Compute Lamé parameters for each element
    lam = (nu * E_elem) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_elem / (2.0 * (1.0 + nu))
    # lam, mu = lam_mu(E_elem, nu)
    lam = lam.reshape(-1, 1)
    mu = mu.reshape(-1, 1)

    @BilinearForm
    def stiffness_form(u, v, w):
        # sym_grad(u) is the strain tensor ε(u) at integration points
        # trace(sym_grad(u)) is the volumetric strain (divergence of u)
        strain_u = sym_grad(u)
        strain_v = sym_grad(v)
        # λ * tr(ε(u)) * tr(ε(v))
        term_volumetric = lam * trace(strain_u) * trace(strain_v)
        # 2μ * (ε(u) : ε(v))
        term_dev = 2.0 * mu * ddot(strain_u, strain_v)
        return term_volumetric + term_dev  # integrand for stiffness
    # 4. Assemble the stiffness matrix using the basis
    K = asm(stiffness_form, basis)
    return K


@njit(parallel=True)
def _get_elements_volume_tet_numba(t_conn, p_coords) -> np.ndarray:
    n_elements = t_conn.shape[1]
    elements_volume = np.zeros(n_elements)
    for e in prange(n_elements):
        n0, n1, n2, n3 = t_conn[:, e]
        v1 = p_coords[:, n1] - p_coords[:, n0]
        v2 = p_coords[:, n2] - p_coords[:, n0]
        v3 = p_coords[:, n3] - p_coords[:, n0]
        vol = np.dot(np.cross(v1, v2), v3) / 6.0
        elements_volume[e] = vol

    return elements_volume


def _get_elements_volume_tet(t_conn, p_coords) -> np.ndarray:
    n_elements = t_conn.shape[1]
    elements_volume = np.zeros(n_elements)
    # for e in prange(n_elements):
    for e in range(n_elements):
        n0, n1, n2, n3 = t_conn[:, e]
        v1 = p_coords[:, n1] - p_coords[:, n0]
        v2 = p_coords[:, n2] - p_coords[:, n0]
        v3 = p_coords[:, n3] - p_coords[:, n0]
        vol = np.dot(np.cross(v1, v2), v3) / 6.0

        if vol < -1e-12:
            print("Element", e, "has negative volume:", vol)
            raise ValueError("!!!")
        elements_volume[e] = vol

    return elements_volume


@njit
def _tet_volume_numba(p0, p1, p2, p3):
    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0
    return abs(np.dot(np.cross(v1, v2), v3)) / 6.0


def _get_elements_volume_hex(t_conn, p_coords) -> np.ndarray:
    """
    Compute volume of Hex elements by decomposing into 6 tetrahedra.

    Parameters
    ----------
    t_conn : (8, n_elem) int
        Hexahedral element connectivity
    p_coords : (3, n_nodes) float
        Node coordinates

    Returns
    -------
    elements_volume : (n_elem,) float
        Approximate volumes
    """
    n_elements = t_conn.shape[1]
    elements_volume = np.zeros(n_elements)

    for e in prange(n_elements):
        n = t_conn[:, e]

        #   7--------6
        #  /|       /|
        # 4--------5 |
        # | |      | |
        # | 3------|-2
        # |/       |/
        # 0--------1
        vol = 0.0
        vol += _tet_volume_numba(
            p_coords[:, n[0]], p_coords[:, n[1]],
            p_coords[:, n[3]], p_coords[:, n[4]]
        )
        vol += _tet_volume_numba(
            p_coords[:, n[1]], p_coords[:, n[2]],
            p_coords[:, n[3]], p_coords[:, n[6]]
        )
        vol += _tet_volume_numba(
            p_coords[:, n[1]], p_coords[:, n[5]],
            p_coords[:, n[6]], p_coords[:, n[4]]
        )
        vol += _tet_volume_numba(
            p_coords[:, n[3]], p_coords[:, n[6]],
            p_coords[:, n[7]], p_coords[:, n[4]]
        )
        vol += _tet_volume_numba(
            p_coords[:, n[1]], p_coords[:, n[3]],
            p_coords[:, n[6]], p_coords[:, n[4]]
        )
        vol += _tet_volume_numba(
            p_coords[:, n[1]], p_coords[:, n[6]],
            p_coords[:, n[5]], p_coords[:, n[4]]
        )

        elements_volume[e] = vol

    return elements_volume


@njit(parallel=True)
def _get_elements_volume_hex_numba(t_conn, p_coords) -> np.ndarray:
    """
    Compute volume of Hex elements by decomposing into 6 tetrahedra.

    Parameters
    ----------
    t_conn : (8, n_elem) int
        Hexahedral element connectivity
    p_coords : (3, n_nodes) float
        Node coordinates

    Returns
    -------
    elements_volume : (n_elem,) float
        Approximate volumes
    """
    n_elements = t_conn.shape[1]
    elements_volume = np.zeros(n_elements)

    for e in prange(n_elements):
        n = t_conn[:, e]

        # Tetra 6 division
        #   7--------6
        #  /|       /|
        # 4--------5 |
        # | |      | |
        # | 3------|-2
        # |/       |/
        # 0--------1

        vol = 0.0
        vol += _tet_volume_numba(
            p_coords[:, n[0]], p_coords[:, n[1]],
            p_coords[:, n[3]], p_coords[:, n[4]]
        )
        vol += _tet_volume_numba(
            p_coords[:, n[1]], p_coords[:, n[2]],
            p_coords[:, n[3]], p_coords[:, n[6]]
        )
        vol += _tet_volume_numba(
            p_coords[:, n[1]], p_coords[:, n[5]],
            p_coords[:, n[6]], p_coords[:, n[4]]
        )
        vol += _tet_volume_numba(
            p_coords[:, n[3]], p_coords[:, n[6]],
            p_coords[:, n[7]], p_coords[:, n[4]]
        )
        vol += _tet_volume_numba(
            p_coords[:, n[1]], p_coords[:, n[3]],
            p_coords[:, n[6]], p_coords[:, n[4]]
        )
        vol += _tet_volume_numba(
            p_coords[:, n[1]], p_coords[:, n[6]],
            p_coords[:, n[5]], p_coords[:, n[4]]
        )

        elements_volume[e] = vol

    return elements_volume


def get_elements_volume(
    mesh: skfem.Mesh
) -> np.ndarray:
    if isinstance(mesh, skfem.MeshTet):
        return _get_elements_volume_tet(mesh.t, mesh.p)
    elif isinstance(mesh, skfem.MeshHex):
        return _get_elements_volume_hex(mesh.t, mesh.p)
    else:
        raise NotImplementedError("skfem.MeshTet or skfem.MeshHex")


def assemble_stiffness_matrix_simp(
    basis: Basis,
    rho: np.ndarray,
    E0: float, Emin: float, p: float, nu: float
):
    return assemble_stiffness_matrix(
        basis,
        rho,
        E0, Emin, p, nu,
        elem_func=simp_interpolation
    )


def assemble_stiffness_matrix_ramp(
    basis: Basis,
    rho: np.ndarray,
    E0: float, Emin: float, p: float, nu: float
):
    return assemble_stiffness_matrix(
        basis,
        rho,
        E0, Emin, p, nu,
        elem_func=ramp_interpolation
    )


def adjacency_matrix(mesh: skfem.MeshTet):
    n_elements = mesh.t.shape[1]
    face_to_elements = defaultdict(list)
    for i in range(n_elements):
        tet = mesh.t[:, i]
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for face in faces:
            face_to_elements[face].append(i)

    adjacency = [[] for _ in range(n_elements)]
    for elems in face_to_elements.values():
        if len(elems) == 2:
            i, j = elems
            adjacency[i].append(j)
            adjacency[j].append(i)
    return adjacency


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
    elem_func: Callable = simp_interpolation
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
    elem_func: Callable = simp_interpolation
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


@Functional
def compute_element_stress_tensor(w):
    """
    Return stress tensor per element per quadrature point.
    Output shape: (3, 3, n_elem, n_qp)
    """
    # shape: (3, 3, nelems, nqp)
    grad = w['uh'].grad
    # shape: same
    symgrad = 0.5 * (grad + transpose(grad))

    # shape: (nelems, nqp)
    tr = trace(symgrad)
    # shape: (3, 3, nelems, nqp)
    I_mat = eye(tr, symgrad.shape[0])

    # shape: (1, 1, nelems, nqp)
    mu = w['mu_elem'].T[None, None, :, :]
    # shape: same
    lam = w['lam_elem'].T[None, None, :, :]
    # shape: (3, 3, nelems, nqp)
    stress = 2. * mu * symgrad + lam * I_mat
    return stress


def stress_tensor_skfem(
    basis: skfem.Basis,
    rho: np.ndarray, u: np.ndarray,
    E0: float, Emin: float, p: float, nu: float,
    elem_func: Callable = simp_interpolation
):
    uh = basis.interpolate(u)
    E_elem = elem_func(rho, E0, Emin, p)
    lam_elem, mu_elem = lame_parameters(E_elem, nu)  # shape: (nelements,)
    n_qp = basis.X.shape[1]

    lam_elem = np.tile(lam_elem, (n_qp, 1))  # shape: (n_qp, n_elements)
    mu_elem = np.tile(mu_elem, (n_qp, 1))

    stress_tensor = compute_element_stress_tensor.elemental(
        basis,
        uh=uh,
        # uh=u,
        lam_elem=lam_elem, mu_elem=mu_elem
    )
    return stress_tensor  # shape: (n_elem, n_qp, 3, 3)


def von_mises_from_stress_tensor(stress_tensor: np.ndarray) -> np.ndarray:
    """
    Compute Von Mises stress from full stress tensor.

    Parameters:
        stress_tensor: ndarray of shape (3, 3, n_elem, n_qp)

    Returns:
        von_mises: ndarray of shape (n_elem, n_qp)
    """
    s = stress_tensor
    s_xx = s[0, 0]
    s_yy = s[1, 1]
    s_zz = s[2, 2]
    s_xy = s[0, 1]
    s_yz = s[1, 2]
    s_zx = s[2, 0]

    return np.sqrt(
        0.5 * (
            (s_xx - s_yy)**2 +
            (s_yy - s_zz)**2 +
            (s_zz - s_xx)**2 +
            6 * (s_xy**2 + s_yz**2 + s_zx**2)
        )
    )


if __name__ == '__main__':
    import time
    import meshio
    import pyvista as pv
    from memory_profiler import profile

    @profile
    def test_1():
        from sktopt.mesh import toy_problem

        tsk = toy_problem.toy_msh("plate-0.2.msh")
        rho = np.ones(tsk.all_elements.shape)

        K1 = assemble_stiffness_matrix(
            tsk.basis, rho, tsk.E0, 0.0, 1.0, tsk.nu0
        )
        lam, mu = lame_parameters(tsk.E0, tsk.nu0)

        def C(T):
            return 2. * mu * T + lam * eye(trace(T), T.shape[0])

        @skfem.BilinearForm
        def stiffness(u, v, w):
            return ddot(C(sym_grad(u)), sym_grad(v))

        _F = tsk.force
        K2 = stiffness.assemble(tsk.basis)

        # print("tsk.dirichlet_nodes", tsk.dirichlet_nodes)
        K1_e, F1_e = skfem.enforce(K1, _F, D=tsk.dirichlet_nodes)
        K2_e, F2_e = skfem.enforce(K2, _F, D=tsk.dirichlet_nodes)

        U1_e = scipy.sparse.linalg.spsolve(K1_e, F1_e)
        U2_e = scipy.sparse.linalg.spsolve(K2_e, F2_e)

        print("U1_e:", np.average(U1_e))
        print("U2_e:", np.average(U2_e))

        sf = 1.0
        m1 = tsk.mesh.translated(sf * U1_e[tsk.basis.nodal_dofs])
        m1.save('K1.vtk')
        m2 = tsk.mesh.translated(sf * U2_e[tsk.basis.nodal_dofs])
        m2.save('K2.vtk')

        K1_e, F1_e = skfem.enforce(K1, _F, D=tsk.dirichlet_nodes)
        # K1_e_np = K1_e.toarray()
        U1_e = scipy.sparse.linalg.spsolve(K1_e, F1_e)
        u = U1_e
        K = K1_e.toarray()
        U_global = 0.5 * u @ (K @ u)
        print("Global:", U_global)

        print(tsk.basis.element_dofs.shape, rho.shape)
        U_elementwise1 = strain_energy_hdcode(
            u, tsk.basis.element_dofs,
            tsk.basis,
            rho,
            tsk.E0,
            tsk.Emin,
            1.0,
            tsk.nu0,
        ).sum()
        element_dofs = tsk.basis.element_dofs[:, tsk.design_elements]
        rho_design = rho[tsk.design_elements]
        print(element_dofs.shape, rho_design.shape)

        t0 = time.time()
        U_elementwise2 = strain_energy_hdcode(
            u, element_dofs,
            tsk.basis,
            rho_design,
            tsk.E0,
            tsk.Emin,
            1.0,
            tsk.nu0,
        ).sum()
        t1 = time.time()

        U_elementwise2_numba1 = strain_energy_hdcode_numba(
            u, element_dofs,
            tsk.mesh.p,
            rho_design,
            tsk.E0,
            tsk.Emin,
            1.0,
            tsk.nu0,
        ).sum()
        t2 = time.time()
        U_elementwise2_numba2 = strain_energy_hdcode_numba(
            u, element_dofs,
            tsk.mesh.p,
            rho_design,
            tsk.E0,
            tsk.Emin,
            1.0,
            tsk.nu0,
        ).sum()
        t3 = time.time()

        print("numpy", t1 - t0)
        print("numba-1st", t2 - t1)
        print("numba-2nd", t3 - t2)
        element_dofs = tsk.basis.element_dofs[:, tsk.free_nodes]
        rho_design = rho[tsk.free_nodes]
        print(element_dofs.shape, rho_design.shape)
        U_elementwise3 = strain_energy_hdcode(
            u, element_dofs,
            tsk.basis,
            rho_design,
            tsk.E0,
            tsk.Emin,
            1.0,
            tsk.nu0,
        ).sum()
        print("Sum over elements all:", U_elementwise1)
        print("Sum over elements design:", U_elementwise2)
        print("Sum over elements design:", U_elementwise3)
        print("error", np.sum(U_elementwise2 - U_elementwise2_numba1))
        print("error", np.sum(U_elementwise2 - U_elementwise2_numba2))

    @profile
    def test_2():
        import time
        from sktopt.mesh import toy_problem
        # tsk = toy_problem.toy1()
        tsk = toy_problem.toy_msh("plate-0.2.msh")
        rho = np.ones(tsk.all_elements.shape)
        p = 1
        t0 = time.time()
        K0 = assemble_stiffness_matrix_numba(
            tsk.basis,
            rho,
            tsk.E0, tsk.Emin, p, tsk.nu0
        )
        t1 = time.time()
        print("numba - 1st", t1 - t0, "sec")
        t0 = time.time()
        K0 = assemble_stiffness_matrix_numba(
            tsk.basis,
            rho,
            tsk.E0, tsk.Emin, p, tsk.nu0
        )
        t1 = time.time()
        print("numba - 2nd (firster than 1st time)", t1 - t0, "sec")

        K1 = assemble_stiffness_matrix(
            tsk.basis,
            rho,
            tsk.E0, tsk.Emin, p, tsk.nu0
        )
        print("numpy", time.time() - t1)
        print("err:", np.sum(K0 - K1))

    @profile
    def test_3():
        import sktopt
        from sktopt.mesh import toy_problem

        tsk = toy_problem.toy_msh("plate-0.2.msh")
        rho = np.ones(tsk.all_elements.shape)
        K0 = assemble_stiffness_matrix(
            tsk.basis, rho, tsk.E0, tsk.Emin, 1.0, tsk.nu0
        )
        # K1 = assemble_stiffness_matrix_numba(
        #     tsk.basis, rho, tsk.E0, tsk.Emin, 1.0, tsk.nu0
        # )
        lam, mu = lame_parameters(tsk.E0, tsk.nu0)

        def C(T):
            return 2. * mu * T + lam * eye(trace(T), T.shape[0])

        @skfem.BilinearForm
        def stiffness(u, v, w):
            return ddot(C(sym_grad(u)), sym_grad(v))

        _F = tsk.force
        K2 = stiffness.assemble(tsk.basis)

        # print("tsk.dirichlet_nodes", tsk.dirichlet_nodes)
        K0_e, F0_e = skfem.enforce(K0, _F, D=tsk.dirichlet_nodes)
        # K1_e, F1_e = skfem.enforce(K1, _F, D=tsk.dirichlet_nodes)
        K2_e, F2_e = skfem.enforce(K2, _F, D=tsk.dirichlet_nodes)

        # U1_e = scipy.sparse.linalg.spsolve(K1_e, F1_e)
        # U2_e = scipy.sparse.linalg.spsolve(K2_e, F2_e)
        U0_e = sktopt.fea.solver.solve_u(K0_e, F0_e, chosen_solver="cg_pyamg")
        # U1_e = sktopt.fea.solver.solve_u(K1_e, F1_e, chosen_solver="cg_pyamg")
        U2_e = sktopt.fea.solver.solve_u(K2_e, F2_e, chosen_solver="cg_pyamg")

        print("U0_e ave :", np.average(U0_e))
        # print("U1_e ave :", np.average(U1_e))
        print("U2_e ave:", np.average(U2_e))
        print("U0_e max :", np.max(U0_e))
        # print("U1_e max :", np.max(U1_e))
        print("U2_e max:", np.max(U2_e))
        print("U0_e min :", np.min(U0_e))
        # print("U1_e min :", np.min(U1_e))
        print("U2_e min:", np.min(U2_e))

        if isinstance(tsk.mesh, skfem.MeshTet):
            mesh_type = "tetra"
        elif isinstance(tsk.mesh, skfem.MeshHex):
            mesh_type = "hexahedron"
        else:
            raise ValueError("")

        sf = 1.0
        # m1 = tsk.mesh.translated(sf * U1_e[tsk.basis.nodal_dofs])
        # m1.save('K1.vtk')
        m2 = tsk.mesh.translated(sf * U2_e[tsk.basis.nodal_dofs])
        m2.save('K2.vtk')

        # K1_e, F1_e = skfem.enforce(K1, _F, D=tsk.dirichlet_nodes)
        # U1_e = scipy.sparse.linalg.spsolve(K1_e, F1_e)
        # u = U1_e
        # u = tsk.basis.interpolate(U0_e)
        compliance, u_compliance = sktopt.fea.solver.compute_compliance_basis(
            tsk.basis, tsk.free_nodes, tsk.dirichlet_nodes, _F,
            tsk.E0, tsk.Emin, 1.0, tsk.nu0,
            rho,
            elem_func=simp_interpolation,
            # solver="spsolve"
        )
        strain = strain_energy_skfem(
            tsk.basis, rho, u_compliance,
            tsk.E0, tsk.Emin, 1.0, tsk.nu0,
            elem_func=simp_interpolation
        )

        print(np.average(np.abs(U0_e)))
        print(np.average(np.abs(u_compliance)))
        print("u diff :", np.sum((U0_e - u_compliance)**2))
        strain_min_max = (strain.max()/2, strain.max())
        print(f"strain_min_max: {strain_min_max}")
        mesh_path = "strain.vtu"
        cell_outputs = dict()
        # cell_outputs["strain"] = [np.linalg.norm(u, axis=0)]
        cell_outputs["strain"] = [strain]
        meshio_mesh = meshio.Mesh(
            points=tsk.mesh.p.T,
            cells=[(mesh_type, tsk.mesh.t.T)],
            cell_data=cell_outputs
        )
        meshio.write(mesh_path, meshio_mesh)

        strain_image_title = "strain"
        strain_image_path = "strain.jpg"
        pv.start_xvfb()
        mesh = pv.read(mesh_path)
        scalar_name = "strain"
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            cmap="turbo",
            clim=(
                cell_outputs["strain"][0].min(),
                cell_outputs["strain"][0].max()
            ),
            opacity=0.3,
            show_edges=False,
            scalar_bar_args={"title": scalar_name}
        )
        plotter.add_text(
            strain_image_title, position="upper_left",
            font_size=12, color="black"
        )
        plotter.screenshot(strain_image_path)
        plotter.close()

    @profile
    def test_4():
        from skfem.helpers import ddot
        import sktopt
        from sktopt.mesh import toy_problem

        tsk = toy_problem.toy_msh("plate-0.2.msh")
        rho = np.ones(tsk.all_elements.shape)

        K0 = assemble_stiffness_matrix(
            tsk.basis, rho, tsk.E0, 0.0, 1.0, tsk.nu0
        )
        _F = tsk.force
        K_e, F_e = skfem.enforce(K0, _F, D=tsk.dirichlet_nodes)
        u = sktopt.fea.solver.solve_u(K_e, F_e, chosen_solver="cg_pyamg")
        print(
            "np.sum(u[tsk.dirichlet_nodes]):", np.sum(u[tsk.dirichlet_nodes])
        )
        lam, mu = lame_parameters(tsk.E0, tsk.nu0)

        def C(strain):
            return 2.0 * mu * strain + lam * eye(
                trace(strain), strain.shape[0]
            )

        @Functional
        def strain_energy_density(w):
            grad = w['uh'].grad  # shape: (ndim, nqp, nelements)
            symgrad = 0.5 * (grad + transpose(grad))
            return 0.5 * ddot(C(symgrad), symgrad)

        uh = tsk.basis.interpolate(u)
        total_U = strain_energy_density.assemble(tsk.basis, uh=uh)
        element_U = strain_energy_density.elemental(tsk.basis, uh=uh)
        print(f"Total Strain Energy = {total_U}")
        # print("The Strain Energy Each =", element_U)
        elem_energy_simp = strain_energy_skfem(
            tsk.basis, np.ones(tsk.mesh.nelements), u,
            1.0, 0.0, 1.0, 0.3
        )
        # print("The Strain Energy Each =", elem_energy_simp)
        print("Difference =", np.sum((elem_energy_simp - element_U)**2))

        stress = stress_tensor_skfem(
            tsk.basis, np.ones(tsk.mesh.nelements), u,
            1.0, 0.0, 1.0, 0.3
        )
        print("stress:", stress.shape)
        von_mises = von_mises_from_stress_tensor(stress)
        print("von_mises:", von_mises.shape)

    # test_1()
    # test_2()
    test_3()
    test_4()
