from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from typing import Literal
import numpy as np
import scipy
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
import pyamg
import skfem


def compute_tet_volumes(mesh):
    coords = mesh.p[:, mesh.t]  # (3, 4, n_elements)
    a = coords[:, 1, :] - coords[:, 0, :]
    b = coords[:, 2, :] - coords[:, 0, :]
    c = coords[:, 3, :] - coords[:, 0, :]
    return np.abs(np.einsum('ij,ij->j', a, np.cross(b, c, axis=0))) / 6.0


def adjacency_matrix_volume_tet(mesh):
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


def tetra_volume(p):
    # p: shape (3, 4)
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    c = p[:, 3] - p[:, 0]
    return abs(np.dot(a, np.cross(b, c))) / 6.0


def adjacency_matrix_volume_hex(mesh):
    n_elements = mesh.t.shape[1]
    volumes = np.zeros(n_elements)
    face_to_elements = defaultdict(list)

    hex_faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5],  # right
    ]

    for i in range(n_elements):
        hex_elem = mesh.t[:, i]
        for face in hex_faces:
            face_nodes = tuple(sorted(hex_elem[f] for f in face))
            face_to_elements[face_nodes].append(i)

        coords = mesh.p[:, hex_elem]  # shape (3, 8)

        # approximate by dividing a volume into 8 Tetrahedra
        v = 0.0
        v += tetra_volume(coords[:, [0, 1, 3, 4]])
        v += tetra_volume(coords[:, [1, 2, 3, 6]])
        v += tetra_volume(coords[:, [1, 5, 6, 4]])
        v += tetra_volume(coords[:, [3, 6, 7, 4]])
        v += tetra_volume(coords[:, [1, 3, 6, 4]])
        v += tetra_volume(coords[:, [1, 6, 5, 4]])
        volumes[i] = v

    adjacency = defaultdict(list)
    for face, elems in face_to_elements.items():
        if len(elems) == 2:
            i, j = elems
            adjacency[i].append(j)
            adjacency[j].append(i)

    return adjacency, volumes


def adjacency_matrix_volume_hex_fast(mesh):
    t = mesh.t  # shape (8, n_elements)
    n_elements = t.shape[1]

    # ----------------------------------
    # approximate by dividing a volume into 6 Tetrahedra
    # ----------------------------------
    coords = mesh.p[:, t]  # shape: (3, 8, n_elements)

    def tet_volume_vectorized(p0, p1, p2, p3):
        a = p1 - p0
        b = p2 - p0
        c = p3 - p0
        return np.abs(np.einsum('ij,ij->j', a, np.cross(b, c, axis=0))) / 6.0

    v0 = tet_volume_vectorized(
        coords[:, 0, :], coords[:, 1, :], coords[:, 3, :], coords[:, 4, :])
    v1 = tet_volume_vectorized(
        coords[:, 1, :], coords[:, 2, :], coords[:, 3, :], coords[:, 6, :])
    v2 = tet_volume_vectorized(
        coords[:, 1, :], coords[:, 5, :], coords[:, 6, :], coords[:, 4, :])
    v3 = tet_volume_vectorized(
        coords[:, 3, :], coords[:, 6, :], coords[:, 7, :], coords[:, 4, :])
    v4 = tet_volume_vectorized(
        coords[:, 1, :], coords[:, 3, :], coords[:, 6, :], coords[:, 4, :])
    v5 = tet_volume_vectorized(
        coords[:, 1, :], coords[:, 6, :], coords[:, 5, :], coords[:, 4, :])

    volumes = v0 + v1 + v2 + v3 + v4 + v5  # shape: (n_elements,)

    # ----------------------------------
    # 2. build adjancy by storing 4 vertices to dictonary
    # ----------------------------------
    face_to_elements = defaultdict(list)

    hex_faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5],  # right
    ]

    t_T = t.T  # shape: (n_elements, 8)

    for i in range(n_elements):
        hex_nodes = t_T[i]
        for face in hex_faces:
            face_nodes = tuple(sorted(hex_nodes[j] for j in face))
            face_to_elements[face_nodes].append(i)

    adjacency = defaultdict(list)
    for face, elems in face_to_elements.items():
        if len(elems) == 2:
            i, j = elems
            adjacency[i].append(j)
            adjacency[j].append(i)

    return adjacency, volumes


def adjacency_matrix_volume_tet_fast(mesh):

    n_elements = mesh.t.shape[1]
    coords = mesh.p[:, mesh.t]  # shape: (3, 4, n_elements)
    a = coords[:, 1, :] - coords[:, 0, :]
    b = coords[:, 2, :] - coords[:, 0, :]
    c = coords[:, 3, :] - coords[:, 0, :]
    volumes = np.abs(np.einsum('ij,ij->j', a, np.cross(b, c, axis=0))) / 6.0
    face_to_elements = defaultdict(list)

    t = mesh.t.T  # shape: (n_elements, 4)
    for i in range(n_elements):
        tet = t[i]
        face_to_elements[tuple(sorted((tet[0], tet[1], tet[2])))] += [i]
        face_to_elements[tuple(sorted((tet[0], tet[1], tet[3])))] += [i]
        face_to_elements[tuple(sorted((tet[0], tet[2], tet[3])))] += [i]
        face_to_elements[tuple(sorted((tet[1], tet[2], tet[3])))] += [i]

    adjacency = defaultdict(list)
    for face, elems in face_to_elements.items():
        if len(elems) == 2:
            i, j = elems
            adjacency[i].append(j)
            adjacency[j].append(i)

    return adjacency, volumes


def element_to_element_laplacian(
    mesh, radius
):
    if isinstance(mesh, skfem.MeshTet):
        adjacency, volumes = adjacency_matrix_volume_tet_fast(mesh)
    elif isinstance(mesh, skfem.MeshHex):
        adjacency, volumes = adjacency_matrix_volume_hex_fast(mesh)
    else:
        raise NotImplementedError("skfem.MeshTet or skfem.MeshHex")

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
            data.append(-radius**2 * w)
            diag += radius**2 * w

            # rows.append(i)
            # cols.append(j)
            # data.append(-w)
            # diag += w
        rows.append(i)
        cols.append(i)
        data.append(diag)
    laplacian = coo_matrix(
        (data, (rows, cols)), shape=(n_elements, n_elements)
    ).tocsc()
    return laplacian, volumes


def compute_filter_gradient_matrix(mesh: skfem.Mesh, radius: float):
    """
    Compute the Jacobian of the Helmholtz filter: d(rho_filtered)/d(rho)
    """
    laplacian, volumes = element_to_element_laplacian(mesh, radius)
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
        Imat = np.eye(n)
        return np.column_stack([filter_grad_vec(Imat[:, i]) for i in range(n)])

    return filter_grad_vec, filter_jacobian_matrix


def prepare_helmholtz_filter(
    mesh: skfem.Mesh, radius: float,
    design_elements_mask: Optional[np.ndarray] = None,
    exclude_nonadjacent: bool = False,
):
    """
    Precompute and return the matrices and solver for Helmholtz filter.
    """
    laplacian, volumes = element_to_element_laplacian(mesh, radius)
    if False:
        centroids = np.mean(mesh.p[:, mesh.t], axis=1).T
        tree = scipy.spatial.cKDTree(centroids)
        n_elements = mesh.t.shape[1]
        valid_mask = np.zeros(n_elements, dtype=bool)

        for i in range(n_elements):
            idx = tree.query_ball_point(centroids[i], r=radius)
            if np.any(design_elements_mask[idx]):
                valid_mask[i] = True

        laplacian = laplacian.tolil()
        for i in range(n_elements):
            if not valid_mask[i]:
                laplacian.rows[i] = []
                laplacian.data[i] = []
        laplacian = laplacian.tocsc()

        mean_volume = np.mean(volumes[valid_mask]) if np.any(valid_mask) \
            else 1.0
        volumes_normalized = volumes / mean_volume
        volumes_normalized[~valid_mask] = 0.0
    else:
        volumes_normalized = volumes / np.mean(volumes)
    # V = csc_matrix(np.diag(volumes_normalized))
    V = scipy.sparse.diags(volumes_normalized, format="csc")
    A = V + radius**2 * laplacian
    return A, V


def apply_helmholtz_filter_lu(
    rho_element: np.ndarray,
    solver: scipy.sparse.linalg.SuperLU,
    V: scipy.sparse._csc.csc_matrix
) -> np.ndarray:
    """
    Apply the Helmholtz filter
    using a precomputed SuperLU solver and mass matrix V.

    Parameters
    ----------
    rho_element : np.ndarray
        Raw (unfiltered) density values per element.
    solver : SuperLU
        Precomputed LU decomposition of (V + r^2 L).
    V : csc_matrix
        Mass matrix (typically diagonal, from normalized element volumes).

    Returns
    -------
    np.ndarray
        Filtered density values.
    """
    rhs = V @ rho_element
    rho_filtered = solver.solve(rhs)
    return rho_filtered


def apply_filter_gradient_lu(
    vec: np.ndarray, solver: scipy.sparse.linalg.SuperLU,
    V: scipy.sparse._csc.csc_matrix
) -> np.ndarray:
    """
    Apply the Jacobian of the Helmholtz filter:
    d(rho_filtered)/d(rho) to a vector.
    """
    return solver.solve(V @ vec)


def apply_helmholtz_filter_cg(
    rho_element: np.ndarray,
    A: scipy.sparse._csc.csc_matrix, V: scipy.sparse._csc.csc_matrix,
    M: Optional[LinearOperator] = None,
    rtol: float = 1e-6,
    maxiter: Optional[int] = None
) -> np.ndarray:
    """
    Apply the Helmholtz filter using precomputed solver and M.
    """
    n_elements = A.shape
    _maxiter = min(1000, max(300, n_elements // 5)) \
        if maxiter is None else maxiter
    rhs = V @ rho_element
    rho_filtered, info = cg(A, rhs, M=M, rtol=rtol, maxiter=_maxiter)
    print("helmholtz_filter_cg-info: ", info)
    if info > 0:
        raise RuntimeError("helmholtz_filter_cg does not converge")
    return rho_filtered


def apply_helmholtz_filter_amg(
    rho_element: np.ndarray,
    V: scipy.sparse.csc_matrix,
    ml: pyamg.multilevel.MultilevelSolver,
    tol: float = 1e-8
) -> np.ndarray:
    """
    Apply the Helmholtz filter using AMG (PyAMG) directly.

    Parameters
    ----------
    rho_element : ndarray
        Raw element-wise density values.
    A : sparse.csc_matrix
        Helmholtz system matrix: A = V + r^2 * L.
    V : sparse.csc_matrix
        Diagonal volume weight matrix.
    ml : pyamg.MultilevelSolver
        AMG solver preconstructed from A.
    tol : float
        Solver tolerance (default 1e-8).

    Returns
    -------
    rho_filtered : ndarray
        Filtered density.
    """
    rhs = V @ rho_element
    # rho_filtered = ml.solve(rhs, tol=tol)
    return ml.solve(rhs, tol=tol)


def apply_filter_gradient_cg(
    vec: np.ndarray,
    A: scipy.sparse._csc.csc_matrix,
    V: scipy.sparse._csc.csc_matrix,
    M: Optional[LinearOperator] = None,
    rtol: float = 1e-6,
    maxiter: Optional[int] = None
) -> np.ndarray:
    """
    Apply the Jacobian of the Helmholtz filter:
    d(rho_filtered)/d(rho) to a vector.
    """
    n_elements = A.shape
    _maxiter = min(1000, max(300, n_elements // 5)) \
        if maxiter is None else maxiter

    ret, info = cg(A, V @ vec, M=M, rtol=rtol, maxiter=_maxiter)
    print("filter_gradient_cg-info: ", info)
    if info > 0:
        raise RuntimeError("filter_gradient_cg does not converge")
    return ret


def apply_filter_gradient_amg(
    vec: np.ndarray,
    V: scipy.sparse.csc_matrix,
    ml: pyamg.multilevel.MultilevelSolver,
    tol: float = 1e-8
) -> np.ndarray:
    """
    Apply the Jacobian of the Helmholtz filter to a vector
    using AMG (i.e., solve A x = V @ vec).

    Parameters
    ----------
    vec : ndarray
        The input vector to which the Jacobian is applied.
    A : sparse.csc_matrix
        Helmholtz system matrix: A = V + r^2 * L.
    V : sparse.csc_matrix
        Diagonal volume weight matrix.
    ml : pyamg.MultilevelSolver
        Precomputed AMG multilevel solver.
    tol : float
        Solver tolerance.

    Returns
    -------
    x : ndarray
        Result of applying the Helmholtz filter's Jacobian to `vec`.
    """
    rhs = V @ vec
    # result = ml.solve(rhs, tol=tol)
    return ml.solve(rhs, tol=tol)


def _update_radius(
    mesh: skfem.Mesh,
    radius: float,
    dst_path: Optional[str] = None,
    design_mask: Optional[np.ndarray] = None
):
    exclude_nonadjacent = False if design_mask is None else True
    A, V = prepare_helmholtz_filter(
        mesh, radius,
        design_elements_mask=design_mask,
        exclude_nonadjacent=exclude_nonadjacent
    )
    if isinstance(dst_path, str):
        scipy.sparse.save_npz(f"{dst_path}/V.npz", V)
        scipy.sparse.save_npz(f"{dst_path}/A.npz", A)

    return A, V


@dataclass
class HelmholtzFilter():
    A: csc_matrix
    V: csc_matrix
    radius: float
    dst_path: Optional[str] = None
    solver_option: Literal["spsolve", "cg_jacobi", "cg_pyamg"] = "cg_jacobi"
    A_solver: Optional[scipy.sparse.linalg.SuperLU] = None
    M: Optional[LinearOperator] = None
    pyamg_solver: Optional[pyamg.multilevel.MultilevelSolver] = None
    rtol: float = 1e-5
    maxiter: int = 1000

    def update_radius(
        self,
        mesh: skfem.Mesh,
        radius: float,
        design_mask: Optional[np.ndarray] = None,
        solver_option="cg_pyamg"
    ):
        self.radius = radius
        self.A, self.V = _update_radius(
            mesh=mesh,
            radius=radius,
            dst_path=self.dst_path,
            design_mask=design_mask
        )
        self.preprocess(solver_option)

    @classmethod
    def from_defaults(
        cls,
        mesh: skfem.Mesh,
        radius: float=1.0,
        solver_option: Literal[
            "spsolve", "cg_jacobi", "cg_pyamg"] = "cg_pyamg",
        dst_path: Optional[str] = None,
        design_mask: Optional[np.ndarray] = None,
    ):
        A, V = _update_radius(
            mesh=mesh,
            radius=radius,
            dst_path=dst_path,
            design_mask=design_mask
        )
        ret = cls(
            A=A, V=V, radius=radius, dst_path=dst_path,
            solver_option=solver_option
        )
        print(f"preprocess : {solver_option}")
        ret.preprocess(solver_option)
        print(ret.solver_option)
        return ret

    @classmethod
    def from_file(cls, dst_path: str):
        V = scipy.sparse.load_npz(f"{dst_path}/V.npz")
        A = scipy.sparse.load_npz(f"{dst_path}/A.npz")
        # A_solver = splu(A)
        return cls(A, V, radius=-1)

    def filter(self, rho_element: np.ndarray):
        if self.solver_option == "spsolve":
            return apply_helmholtz_filter_lu(
                rho_element, self.A_solver, self.V
            )
        elif self.solver_option == "cg_jacobi":
            return apply_helmholtz_filter_cg(
                rho_element, self.A, self.V, M=self.M,
                rtol=self.rtol,
                maxiter=self.maxiter
            )
        elif self.solver_option == "cg_pyamg":
            return apply_helmholtz_filter_amg(
                rho_element, self.V, self.pyamg_solver,
                tol=self.rtol
            )

    def gradient(self, v: np.ndarray):
        if self.solver_option == "spsolve":
            return apply_filter_gradient_lu(v, self.A_solver, self.V)
        elif self.solver_option == "cg_jacobi":
            return apply_filter_gradient_cg(
                v, self.A, self.V,
                M=self.M,
                rtol=self.rtol,
                maxiter=self.maxiter
            )
        elif self.solver_option == "cg_pyamg":
            return apply_filter_gradient_amg(
                v, self.V,
                self.pyamg_solver,
                tol=self.rtol,
            )
        else:
            raise ValueError("solver_option is not set")

    def preprocess(
        self,
        solver_option: Optional[str] = None
    ):
        if isinstance(solver_option, str):
            if solver_option in ["cg_jacobi", "cg_pyamg", "spsolve"]:
                self.solver_option = solver_option
            else:
                raise ValueError("should be cg/pyamg/spsolve")

        if self.solver_option == "cg_pyamg":
            self.create_amgsolver()
        elif self.solver_option == "cg_jacobi":
            self.create_LinearOperator()
        elif self.solver_option == "spsolve":
            self.create_solver()

    def create_solver(self):
        self.A_solver = splu(self.A)

    def create_LinearOperator(
        self,
        rtol: float = 1e-5,
        maxiter: int = -1
    ):
        self.rtol = rtol
        n_dof = self.A.shape[0]
        self.maxiter = maxiter if maxiter > 0 else n_dof // 4
        eps = 1e-8
        M_inv = 1.0 / (self.A.diagonal() + eps)

        def apply_M(x):
            return M_inv * x

        self.M = LinearOperator(
            self.A.shape, matvec=apply_M
        )

    def create_amgsolver(self):
        self.A = self.A.tocsr()
        self.pyamg_solver = pyamg.smoothed_aggregation_solver(self.A)
        # or
        # Algebraic Multigrid
        # import pyamg
        # ml = pyamg.ruge_stuben_solver(A)
        # x = ml.solve(b, tol=1e-8)
