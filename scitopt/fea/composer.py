from typing import Callable
from collections import defaultdict

import scipy
from numba import njit, prange
import numpy as npu

import skfem
from skfem import Basis, asm
from skfem.helpers import ddot, sym_grad, eye, trace, eye
from skfem.models.elasticity import lame_parameters
from skfem.assembly import BilinearForm
from skfem import asm, Basis
from skfem import BilinearForm
from skfem.assembly import BilinearForm

import numpy as np
from skfem import BilinearForm, asm, Basis
from skfem.helpers import sym_grad, ddot, trace


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
    elem_func: Callable=simp_interpolation
):
    """
    Assemble the global stiffness matrix for 3D linear elasticity with SIMP material interpolation.
    
    Parameters:
        basis : skfem Basis for the mesh (built with ElementVector(ElementTetP1) on MeshTet).
        rho   : 1D array of length n_elements with density values for each element.
        E0    : Young's modulus of solid material (for rho = 1).
        Emin  : Minimum Young's modulus for void material (for rho = 0, ensures numerical stability).
        p     : Penalization power for SIMP (typically >= 1, e.g., 3 for standard topology optimization).
        nu    : Poisson's ratio (assumed constant for all elements).
    
    Returns:
        Sparse stiffness matrix (scipy.sparse.csr_matrix) assembled for the given density distribution.
    """
    # 1. Compute Young's modulus for each element using SIMP
    E_elem = elem_func(rho, E0, Emin, p)  # array of size [n_elements]
    
    # 2. Compute Lamé parameters for each element
    lam = (nu * E_elem) / ((1.0 + nu) * (1.0 - 2.0 * nu))   # first Lamé parameter λ_e per element
    mu  = E_elem / (2.0 * (1.0 + nu))                      # second Lamé parameter (shear modulus) μ_e per element
    # lam, mu = lam_mu(E_elem, nu)
    
    # Reshape to allow broadcasting over integration points (each as [n_elem, 1] column vectors)
    lam = lam.reshape(-1, 1)
    mu  = mu.reshape(-1, 1)
    
    # 3. Define the bilinear form for elasticity (integrand of stiffness entries)
    @BilinearForm
    def stiffness_form(u, v, w):
        # sym_grad(u) is the strain tensor ε(u) at integration points
        # trace(sym_grad(u)) is the volumetric strain (divergence of u)
        # ddot(A, B) computes the double-dot (Frobenius) product of two matrices A and B
        strain_u = sym_grad(u)
        strain_v = sym_grad(v)
        # Apply Lamé parameters for each element (w corresponds to integration context)
        # lam and mu are arrays of shape [n_elem, 1], broadcasting to [n_elem, n_quad] with strain arrays
        term_volumetric = lam * trace(strain_u) * trace(strain_v)      # λ * tr(ε(u)) * tr(ε(v))
        term_dev = 2.0 * mu * ddot(strain_u, strain_v)                 # 2μ * (ε(u) : ε(v))
        return term_volumetric + term_dev  # integrand for stiffness
    
    # 4. Assemble the stiffness matrix using the basis
    K = asm(stiffness_form, basis)
    return K


@njit(parallel=True)
def _assemble_stiffness_matrix_numba(
    p_coords, t_conn, element_dofs, E0, Emin, nu, E_elem
):
    n_elements = t_conn.shape[1]
    data = np.zeros(n_elements * 144)  # 12x12 per element
    row = np.zeros_like(data, dtype=np.int32)
    col = np.zeros_like(data, dtype=np.int32)

    # Base elasticity matrix (for E=1.0, scaled later by E_eff)
    lam_base, mu_base = lam_mu(E0, nu)
    C0 = np.array([
        [1 - nu,    nu,       nu,       0,                   0,                   0                  ],
        [nu,        1 - nu,   nu,       0,                   0,                   0                  ],
        [nu,        nu,       1 - nu,   0,                   0,                   0                  ],
        [0,         0,        0,        (1 - 2*nu) / 2.0,    0,                   0                  ],
        [0,         0,        0,        0,                   (1 - 2*nu) / 2.0,    0                  ],
        [0,         0,        0,        0,                   0,                   (1 - 2*nu) / 2.0 ]
    ])
    C0 *= lam_base

    for e in prange(n_elements):
        nodes = t_conn[:, e]
        coords = p_coords[:, nodes]  # shape (3, 4)

        M = np.ones((4, 4))
        for i in range(4):
            M[i, :3] = coords[:, i]
        Minv = np.linalg.inv(M)
        grads = Minv[:3, :]  # shape (3, 4)

        B = np.zeros((6, 12))
        for j in range(4):
            dNdx, dNdy, dNdz = grads[0, j], grads[1, j], grads[2, j]
            B[0, 3*j    ] = dNdx
            B[1, 3*j + 1] = dNdy
            B[2, 3*j + 2] = dNdz
            B[3, 3*j    ] = dNdy
            B[3, 3*j + 1] = dNdx
            B[4, 3*j + 1] = dNdz
            B[4, 3*j + 2] = dNdy
            B[5, 3*j + 2] = dNdx
            B[5, 3*j    ] = dNdz

        vol = abs(np.linalg.det(M)) / 6.0
        E_eff = E_elem[e]
        C_e = C0 * (E_eff / E0)
        ke = B.T @ C_e @ B * vol

        dofs = element_dofs[:, e]
        for i in range(12):
            for j in range(12):
                idx = e * 144 + i * 12 + j
                data[idx] = ke[i, j]
                row[idx] = dofs[i]
                col[idx] = dofs[j]

    return data, (row, col)


def assemble_stiffness_matrix_numba(
    basis, rho, E0, Emin, pval, nu,
    elem_func: Callable=ramp_interpolation_numba
):
    p_coords = basis.mesh.p
    t_conn = basis.mesh.t
    element_dofs = basis.element_dofs
    E_elem = elem_func(rho, E0, Emin, pval)
    data, rowcol = _assemble_stiffness_matrix_numba(
        p_coords, t_conn, element_dofs, E0, Emin, nu, E_elem
    )
    
    ndof = basis.N
    return scipy.sparse.coo_matrix(
        (data, rowcol), shape=(ndof, ndof)
    ).tocsr()


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


def compute_strain_energy(
    u,
    element_dofs,
    basis,
    rho,
    E0,
    Emin, penal, nu0
):
    """Compute element-wise strain energy for a 3D tetrahedral mesh using SIMP material interpolation."""
    mesh = basis.mesh
    # Material constants for elasticity matrix
    lam_factor = lambda E: E / ((1.0 + nu0) * (1.0 - 2.0 * nu0))  # common factor for isotropic C
    mu_factor  = lambda E: E / (2.0 * (1.0 + nu0))               # shear modulus μ

    n_elems = element_dofs.shape[1]  # number of elements (columns of element_dofs)
    energies = np.zeros(n_elems)
    # Precompute base elasticity matrix for E0 (could also compute fresh each time scaled by E_e)
    C0 = lam_factor(E0) * np.array([
        [1 - nu0,    nu0,       nu0,       0,                   0,                   0                  ],
        [nu0,        1 - nu0,   nu0,       0,                   0,                   0                  ],
        [nu0,        nu0,       1 - nu0,   0,                   0,                   0                  ],
        [0,          0,         0,         (1 - 2*nu0) / 2.0,   0,                   0                  ],
        [0,          0,         0,         0,                   (1 - 2*nu0) / 2.0,   0                  ],
        [0,          0,         0,         0,                   0,                   (1 - 2*nu0) / 2.0 ]
    ])
    # Loop over each element in the design domain
    for idx in range(n_elems):
        # Global DOF indices for this element and extract their coordinates
        edofs = element_dofs[:, idx]                  # 12 DOF indices (3 per node for 4 nodes)
        # Infer the 4 node indices (each node has 3 DOFs). We assume DOFs are grouped by node.
        node_ids = [int(edofs[3*j] // 3) for j in range(4)]
        # Coordinates of the 4 nodes (3x4 matrix)
        coords = mesh.p[:, node_ids]
        # Build matrix M for shape function coefficient solve
        # Each row: [x_i, y_i, z_i, 1] for node i
        M = np.column_stack((coords.T, np.ones(4)))
        Minv = np.linalg.inv(M)
        # Gradients of shape functions (each column i gives grad(N_i) = [dN_i/dx, dN_i/dy, dN_i/dz])
        grads = Minv[:3, :]  # 3x4 matrix of gradients
        # Construct B matrix (6x12) for this element
        B = np.zeros((6, 12))
        for j in range(4):
            dNdx, dNdy, dNdz = grads[0, j], grads[1, j], grads[2, j]
            # Fill B for this node j
            B[0, 3*j    ] = dNdx
            B[1, 3*j + 1] = dNdy
            B[2, 3*j + 2] = dNdz
            B[3, 3*j    ] = dNdy
            B[3, 3*j + 1] = dNdx
            B[4, 3*j + 1] = dNdz
            B[4, 3*j + 2] = dNdy
            B[5, 3*j + 2] = dNdx
            B[5, 3*j    ] = dNdz
        # Compute volume of the tetrahedron (abs(det(M))/6)
        vol = abs(np.linalg.det(M)) / 6.0
        # Young's modulus for this element via SIMP
        E_eff = Emin + (rho[idx] ** penal) * (E0 - Emin)
        # Form elasticity matrix C_e (scale base matrix by E_eff/E0 since ν constant)
        C_e = C0 * (E_eff / E0)
        # Element nodal displacements
        u_e = u[edofs]
        # Compute strain = B * u_e
        strain = B.dot(u_e)
        # Strain energy density = 0.5 * strain^T * C_e * strain
        Ue = 0.5 * strain.dot(C_e.dot(strain)) * vol
        energies[idx] = Ue
    return energies


@njit
def compute_strain_energy_numba(
    u,
    element_dofs,
    node_coords,  # mesh.p
    rho,
    E0,
    Emin,
    penal,
    nu0
):
    lam_factor = lambda E: E / ((1.0 + nu0) * (1.0 - 2.0 * nu0))
    mu_factor  = lambda E: E / (2.0 * (1.0 + nu0))

    n_elems = element_dofs.shape[1]
    energies = np.zeros(n_elems)

    # create C0 in advance
    C0 = lam_factor(E0) * np.array([
        [1 - nu0,    nu0,       nu0,       0,                   0,                   0                  ],
        [nu0,        1 - nu0,   nu0,       0,                   0,                   0                  ],
        [nu0,        nu0,       1 - nu0,   0,                   0,                   0                  ],
        [0,          0,         0,         (1 - 2*nu0) / 2.0,   0,                   0                  ],
        [0,          0,         0,         0,                   (1 - 2*nu0) / 2.0,   0                  ],
        [0,          0,         0,         0,                   0,                   (1 - 2*nu0) / 2.0 ]
    ])

    for idx in range(n_elems):
        edofs = element_dofs[:, idx]

        node_ids = np.empty(4, dtype=np.int32)
        for j in range(4):
            node_ids[j] = int(edofs[3*j] // 3)

        coords = np.zeros((3, 4))
        for j in range(4):
            coords[:, j] = node_coords[:, node_ids[j]]

        M = np.ones((4, 4))
        for i in range(4):
            M[i, :3] = coords[:, i]

        Minv = np.linalg.inv(M)
        grads = Minv[:3, :]

        B = np.zeros((6, 12))
        for j in range(4):
            dNdx, dNdy, dNdz = grads[0, j], grads[1, j], grads[2, j]
            B[0, 3*j    ] = dNdx
            B[1, 3*j + 1] = dNdy
            B[2, 3*j + 2] = dNdz
            B[3, 3*j    ] = dNdy
            B[3, 3*j + 1] = dNdx
            B[4, 3*j + 1] = dNdz
            B[4, 3*j + 2] = dNdy
            B[5, 3*j + 2] = dNdx
            B[5, 3*j    ] = dNdz

        vol = abs(np.linalg.det(M)) / 6.0
        E_eff = Emin + (rho[idx] ** penal) * (E0 - Emin)
        C_e = C0 * (E_eff / E0)
        u_e = u[edofs]
        strain = B.dot(u_e)
        Ue = 0.5 * strain.dot(C_e.dot(strain)) * vol
        energies[idx] = Ue

    return energies


if __name__ == '__main__':
    
    import time
    from memory_profiler import profile

    @profile
    def test_1():
        from scitopt.mesh import toy_problem

        tsk = toy_problem.toy()
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


        # 
        K1_e, F1_e = skfem.enforce(K1, _F, D=tsk.dirichlet_nodes)
        # K1_e_np = K1_e.toarray()
        U1_e = scipy.sparse.linalg.spsolve(K1_e, F1_e)
        u = U1_e
        K = K1_e.toarray()
        U_global = 0.5 * u @ (K @ u)
        print("Global:", U_global)

        # 
        print(tsk.basis.element_dofs.shape, rho.shape)
        U_elementwise1 = compute_strain_energy(
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
        U_elementwise2 = compute_strain_energy(
            u, element_dofs,
            tsk.basis,
            rho_design,
            tsk.E0,
            tsk.Emin,
            1.0,
            tsk.nu0,
        ).sum()
        t1 = time.time()
        
        U_elementwise2_numba1 = compute_strain_energy_numba(
            u, element_dofs,
            tsk.mesh.p,
            rho_design,
            tsk.E0,
            tsk.Emin,
            1.0,
            tsk.nu0,
        ).sum()
        t2 = time.time()
        U_elementwise2_numba2 = compute_strain_energy_numba(
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
        U_elementwise3 = compute_strain_energy(
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

    @profile
    def test_2():
        import time
        from scitopt.mesh import toy_problem
        tsk = toy_problem.toy()
        
        rho = np.ones(tsk.all_elements.shape)
        p = 1
        
        t0 = time.time()
        K0 = assemble_stiffness_matrix_numba(
            tsk.basis,
            rho,
            tsk.E0, tsk.Emin, p, tsk.nu0
        )
        t1 = time.time()
        print("numba - 1st", t1 - t0)
        t0 = time.time()
        K0 = assemble_stiffness_matrix_numba(
            tsk.basis,
            rho,
            tsk.E0, tsk.Emin, p, tsk.nu0
        )
        t1 = time.time()
        print("numba - 2nd", t1 - t0)

        K1 = assemble_stiffness_matrix(
            tsk.basis,
            rho,
            tsk.E0, tsk.Emin, p, tsk.nu0
        )
        print("numpy", time.time() - t1)
        print("err:", np.sum(K0 - K1))

    test_1()
    test_2()
