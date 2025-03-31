from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import numpy as np
import scipy
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu, spsolve
import skfem


def adjacency_matrix_volume(mesh):
    n_elements = mesh.t.shape[1]
    volumes = np.zeros(n_elements)
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

        coords = mesh.p[:, tet]
        a = coords[:, 1] - coords[:, 0]
        b = coords[:, 2] - coords[:, 0]
        c = coords[:, 3] - coords[:, 0]
        volumes[i] = abs(np.dot(a, np.cross(b, c))) / 6.0

    adjacency = defaultdict(list)
    for face, elems in face_to_elements.items():
        if len(elems) == 2:
            i, j = elems
            adjacency[i].append(j)
            adjacency[j].append(i)
    return (adjacency, volumes)


def element_to_element_laplacian_tet(mesh, radius):
    adjacency, volumes = adjacency_matrix_volume(mesh)
    n_elements = mesh.t.shape[1]
    element_centers = np.mean(mesh.p[:, mesh.t], axis=1).T
    rows = []
    cols = []
    data = []
    for i in range(n_elements):
        diag = 0.0
        for j in adjacency[i]:
            dist = np.linalg.norm(element_centers[i] - element_centers[j])
            if dist < 1e-12:
                continue
            # w = 1.0 / (dist + 1e-5)
            w = np.exp(-dist**2 / (2 * radius**2)) 
            rows.append(i)
            cols.append(j)
            data.append(-w)
            diag += w
        rows.append(i)
        cols.append(i)
        data.append(diag)
    laplacian = coo_matrix((data, (rows, cols)), shape=(n_elements, n_elements)).tocsc()
    return laplacian, volumes


def helmholtz_filter_element_based_tet(rho_element: np.ndarray, mesh: skfem.Mesh, radius: float) -> np.ndarray:
    """
    """
    laplacian, volumes = element_to_element_laplacian_tet(mesh, radius)
    volumes_normalized = volumes / np.mean(volumes)

    M = csc_matrix(np.diag(volumes_normalized))
    A = M + radius**2 * laplacian
    rhs = M @ rho_element

    rho_filtered = spsolve(A, rhs)
    return rho_filtered


def compute_filter_gradient_matrix(mesh: skfem.Mesh, radius: float):
    """
    Compute the Jacobian of the Helmholtz filter: d(rho_filtered)/d(rho)
    """
    laplacian, volumes = element_to_element_laplacian_tet(mesh, radius)
    volumes_normalized = volumes / np.mean(volumes)

    M = csc_matrix(np.diag(volumes_normalized))
    A = M + radius**2 * laplacian

    # Solve: d(rho_filtered)/d(rho) = A^{-1} * M
    # You can precompute LU for efficiency
    A_solver = splu(A)

    def filter_grad_vec(v: np.ndarray) -> np.ndarray:
        """Applies Jacobian to vector v"""
        return A_solver.solve(M @ v)

    def filter_jacobian_matrix() -> np.ndarray:
        """Returns the full Jacobian matrix: A^{-1} @ M"""
        n = M.shape[0]
        I = np.eye(n)
        return np.column_stack([filter_grad_vec(I[:, i]) for i in range(n)])

    return filter_grad_vec, filter_jacobian_matrix


def prepare_helmholtz_filter(mesh: skfem.Mesh, radius: float):
    """
    Precompute and return the matrices and solver for Helmholtz filter.
    """
    laplacian, volumes = element_to_element_laplacian_tet(mesh, radius)
    volumes_normalized = volumes / np.mean(volumes)

    M = csc_matrix(np.diag(volumes_normalized))
    A = M + radius**2 * laplacian
    return A, M


def apply_helmholtz_filter(rho_element: np.ndarray, solver, M) -> np.ndarray:
    """
    Apply the Helmholtz filter using precomputed solver and M.
    """
    rhs = M @ rho_element
    rho_filtered = solver.solve(rhs)
    return rho_filtered


def apply_filter_gradient(v: np.ndarray, solver, M) -> np.ndarray:
    """
    Apply the Jacobian of the Helmholtz filter: d(rho_filtered)/d(rho) to a vector.
    """
    return solver.solve(M @ v)


@dataclass
class HelmholtzFilter():
    A: scipy.sparse.linalg.SuperLU
    M: csc_matrix
    A_solver: Optional[scipy.sparse.linalg.SuperLU]


    @classmethod
    def from_defaults(
        cls,
        mesh: skfem.Mesh,
        radius: float,
        dst_path: Optional[str]=None
    ):
        A, M = prepare_helmholtz_filter(mesh, radius)
        if isinstance(dst_path, str):
            scipy.sparse.save_npz(f"{dst_path}/M.npz", M)
            scipy.sparse.save_npz(f"{dst_path}/A.npz", A)

        # A_solver = splu(A)
        return cls(
            A, M, None
        )
    
    @classmethod
    def from_file(cls, dst_path: str):
        M = scipy.sparse.load_npz(f"{dst_path}/M.npz")
        A = scipy.sparse.load_npz(f"{dst_path}/A.npz")
        A_solver = splu(A)
        return cls(
            A, M, A_solver
        )
    
    def filter(self, rho_element: np.ndarray):
        return apply_helmholtz_filter(rho_element, self.A_solver, self.M)

    def gradient(self, v: np.ndarray):
        return apply_filter_gradient(v, self.A_solver, self.M)

    def create_solver(self):
        self.A_solver = splu(self.A)