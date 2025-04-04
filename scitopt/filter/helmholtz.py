from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import numpy as np
import scipy
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu, spsolve
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator, spilu
import skfem


def compute_tet_volumes(mesh):
    coords = mesh.p[:, mesh.t]  # (3, 4, n_elements)
    a = coords[:, 1, :] - coords[:, 0, :]
    b = coords[:, 2, :] - coords[:, 0, :]
    c = coords[:, 3, :] - coords[:, 0, :]
    return np.abs(np.einsum('ij,ij->j', a, np.cross(b, c, axis=0))) / 6.0


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


def adjacency_matrix_volume_fast(mesh):
    n_elements = mesh.t.shape[1]

    # -------------------------
    # 1. ベクトル化された体積計算
    # -------------------------
    coords = mesh.p[:, mesh.t]  # shape: (3, 4, n_elements)
    a = coords[:, 1, :] - coords[:, 0, :]
    b = coords[:, 2, :] - coords[:, 0, :]
    c = coords[:, 3, :] - coords[:, 0, :]
    volumes = np.abs(np.einsum('ij,ij->j', a, np.cross(b, c, axis=0))) / 6.0

    # -------------------------
    # 2. 高速face→要素マップ構築（sorted排除は安全のため残す）
    # -------------------------
    face_to_elements = defaultdict(list)

    t = mesh.t.T  # shape: (n_elements, 4)
    # 全要素に対して面を列挙
    for i in range(n_elements):
        tet = t[i]
        # 各面を小さい順にtuple化（一意なキー）
        face_to_elements[tuple(sorted((tet[0], tet[1], tet[2])))] += [i]
        face_to_elements[tuple(sorted((tet[0], tet[1], tet[3])))] += [i]
        face_to_elements[tuple(sorted((tet[0], tet[2], tet[3])))] += [i]
        face_to_elements[tuple(sorted((tet[1], tet[2], tet[3])))] += [i]

    # -------------------------
    # 3. 面共有から隣接情報作成
    # -------------------------
    adjacency = defaultdict(list)
    for face, elems in face_to_elements.items():
        if len(elems) == 2:
            i, j = elems
            adjacency[i].append(j)
            adjacency[j].append(i)

    return adjacency, volumes


def element_to_element_laplacian_tet(mesh, radius):
    adjacency, volumes = adjacency_matrix_volume_fast(mesh)
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
    # V = csc_matrix(np.diag(volumes_normalized))
    V = scipy.sparse.diags(volumes_normalized, format="csc")
    A = V + radius**2 * laplacian
    return A, V


def apply_helmholtz_filter_lu(
    rho_element: np.ndarray, solver: scipy.sparse.linalg.SuperLU,
    V: scipy.sparse._csc.csc_matrix
) -> np.ndarray:
    """
    Apply the Helmholtz filter using precomputed solver and M.
    """
    rhs = V @ rho_element
    rho_filtered = solver.solve(rhs)
    return rho_filtered


def apply_filter_gradient_lu(
    vec: np.ndarray, solver: scipy.sparse.linalg.SuperLU,
    V: scipy.sparse._csc.csc_matrix
) -> np.ndarray:
    """
    Apply the Jacobian of the Helmholtz filter: d(rho_filtered)/d(rho) to a vector.
    """
    return solver.solve(V @ vec)


def apply_helmholtz_filter_cg(
    rho_element: np.ndarray,
    A: scipy.sparse._csc.csc_matrix, V: scipy.sparse._csc.csc_matrix,
    M: Optional[LinearOperator] = None,
    rtol: float=1e-6,
    maxiter: Optional[int]=None
) -> np.ndarray:
    """
    Apply the Helmholtz filter using precomputed solver and M.
    """
    n_elements = A.shape
    _maxiter = min(1000, max(300, n_elements // 5)) if maxiter is None else maxiter
    rhs = V @ rho_element
    rho_filtered, info = cg(A, rhs, M=M, rtol=rtol, maxiter=_maxiter)
    print("helmholtz_filter_cg-info: ", info)
    if info > 0:
        raise RuntimeError("helmholtz_filter_cg does not converge")
    return rho_filtered


def apply_filter_gradient_cg(
    vec: np.ndarray,
    A: scipy.sparse._csc.csc_matrix,
    V: scipy.sparse._csc.csc_matrix,
    M: Optional[LinearOperator] = None,
    rtol: float=1e-6,
    maxiter: Optional[int]=None
) -> np.ndarray:
    """
    Apply the Jacobian of the Helmholtz filter: d(rho_filtered)/d(rho) to a vector.
    """
    n_elements = A.shape
    _maxiter = min(1000, max(300, n_elements // 5)) if maxiter is None else maxiter

    ret, info = cg(A, V @ vec, M=M, rtol=rtol, maxiter=_maxiter)
    print("filter_gradient_cg-info: ", info)
    if info > 0:
        raise RuntimeError("filter_gradient_cg does not converge")
    return ret


@dataclass
class HelmholtzFilter():
    A: scipy.sparse.linalg.SuperLU
    V: csc_matrix
    A_solver: Optional[scipy.sparse.linalg.SuperLU]=None
    M: Optional[LinearOperator]=None
    rtol: float=1e-5
    maxiter: int=1000


    @classmethod
    def from_defaults(
        cls,
        mesh: skfem.Mesh,
        radius: float,
        dst_path: Optional[str]=None
    ):
        A, V = prepare_helmholtz_filter(mesh, radius)
        if isinstance(dst_path, str):
            scipy.sparse.save_npz(f"{dst_path}/V.npz", V)
            scipy.sparse.save_npz(f"{dst_path}/A.npz", A)

        # A_solver = splu(A)
        return cls(
            A, V, None
        )
    
    @classmethod
    def from_file(cls, dst_path: str):
        V = scipy.sparse.load_npz(f"{dst_path}/V.npz")
        A = scipy.sparse.load_npz(f"{dst_path}/A.npz")
        # A_solver = splu(A)
        return cls(A, V)


    def filter(self, rho_element: np.ndarray):
        if self.A_solver is not None:
            return apply_helmholtz_filter_lu(rho_element, self.A_solver, self.V)
        else:
            return apply_helmholtz_filter_cg(
                rho_element, self.A, self.V, M=self.M,
                rtol=self.rtol,
                maxiter=self.maxiter
            )

        
    def gradient(self, v: np.ndarray):
        if self.A_solver is not None:
            return apply_filter_gradient_lu(v, self.A_solver, self.V)
        else:
            return apply_filter_gradient_cg(
                v, self.A, self.V,
                M=self.M,
                rtol=self.rtol,
                maxiter=self.maxiter
            )

    def create_solver(self):
        self.A_solver = splu(self.A)
    
    def create_LinearOperator(
        self,
        rtol: float=1e-5,
        maxiter: int=-1
    ):
        self.rtol = rtol
        n_dof = self.A.shape[0]
        self.maxiter = maxiter if maxiter > 0 else n_dof // 4
        
        # 
        M_inv = 1.0 / self.A.diagonal()
        def apply_M(x):
            return M_inv * x

        self.M = LinearOperator(
            self.A.shape, matvec=apply_M
        )
        
        # or
        # Algebraic Multigrid
        # import pyamg
        # ml = pyamg.ruge_stuben_solver(A)
        # x = ml.solve(b, tol=1e-8)