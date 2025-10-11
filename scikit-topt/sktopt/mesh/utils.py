from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange
import skfem


def get_points_in_range(x_rng, y_rng, z_rng):
    def in_range(x):
        return (
            (x[0] >= x_rng[0]) & (x[0] <= x_rng[1]) &
            (x[1] >= y_rng[0]) & (x[1] <= y_rng[1]) &
            (x[2] >= z_rng[0]) & (x[2] <= z_rng[1])
        )
    return in_range


def fix_hexahedron_orientation(t, p):
    """
    Ensures that each hexahedral element in the mesh has positive volume
    (i.e., right-handed orientation). Adjusts the order of nodes if needed.

    Parameters
    ----------
    t : (8, n_elem) int
        Hexahedral element connectivity (e.g., mesh.t)
    p : (3, n_nodes) float
        Node coordinates (e.g., mesh.p)

    Returns
    -------
    t_fixed : (8, n_elem) int
        Corrected node ordering for each element.
    """

    t_fixed = np.array(t, copy=True)
    n_elem = t.shape[1]

    for e in range(n_elem):
        hex_nodes = t_fixed[:, e]
        coords = p[:, hex_nodes]  # shape (3, 8)

        # Compute Jacobian determinant at natural coordinate center
        # Ref: Hex element master nodes index: [0,1,2,3,4,5,6,7]
        v1 = coords[:, 1] - coords[:, 0]  # x-direction
        v2 = coords[:, 3] - coords[:, 0]  # y-direction
        v3 = coords[:, 4] - coords[:, 0]  # z-direction
        jac_det = np.dot(np.cross(v1, v2), v3)

        if jac_det < 0:
            # Swap two nodes to flip orientation (commonly swap node 1 and 3)
            t_fixed[1, e], t_fixed[3, e] = t_fixed[3, e], t_fixed[1, e]

    return t_fixed


def fix_tetrahedron_orientation(t, p):
    """
    Returns a corrected version of `t` where all tetrahedral elements
    follow a right-handed (positive volume) orientation.

    Parameters
    ----------
    t : (4, n_elem) int
        Tetrahedral element connectivity array (e.g., mesh.t in scikit-fem),
        where each column contains indices of 4 nodes forming one element.
    p : (3, n_nodes) float
        Coordinates of the mesh nodes (e.g., mesh.p in scikit-fem), where
        each column represents a node in 3D space.

    Returns
    -------
    t_fixed : (4, n_elem) int
        A corrected connectivity array where the node ordering of each
        tetrahedron is adjusted (if needed) to ensure positive volume.
    """

    t_fixed = np.array(t, copy=True)
    n_elem = t.shape[1]

    for e in range(n_elem):
        n0, n1, n2, n3 = t_fixed[:, e]
        v1 = p[:, n1] - p[:, n0]
        v2 = p[:, n2] - p[:, n0]
        v3 = p[:, n3] - p[:, n0]
        vol = np.dot(np.cross(v1, v2), v3) / 6.0

        if vol < 0:
            t_fixed[1, e], t_fixed[2, e] = t_fixed[2, e], t_fixed[1, e]

    return t_fixed


@njit(parallel=True)
def fix_tetrahedron_orientation_numba(t, p):
    """
    Returns a corrected version of `t` where all tetrahedral elements
    follow a right-handed (positive volume) orientation.

    Parameters
    ----------
    t : (4, n_elem) int
        Tetrahedral element connectivity array (e.g., mesh.t in scikit-fem),
        where each column contains indices of 4 nodes forming one element.
    p : (3, n_nodes) float
        Coordinates of the mesh nodes (e.g., mesh.p in scikit-fem), where
        each column represents a node in 3D space.

    Returns
    -------
    t_fixed : (4, n_elem) int
        A corrected connectivity array where the node ordering of each
        tetrahedron is adjusted (if needed) to ensure positive volume.
    """

    t_fixed = np.array(t, copy=True)
    n_elem = t.shape[1]
    for e in prange(n_elem):
        n0, n1, n2, n3 = t_fixed[:, e]
        v1 = p[:, n1] - p[:, n0]
        v2 = p[:, n2] - p[:, n0]
        v3 = p[:, n3] - p[:, n0]
        vol = np.dot(np.cross(v1, v2), v3) / 6.0
        if vol < 0:
            t_fixed[1, e], t_fixed[2, e] = t_fixed[2, e], t_fixed[1, e]

    return t_fixed


def fix_elements_orientation(mesh):
    if isinstance(mesh, skfem.MeshTet):
        return fix_tetrahedron_orientation(mesh.t, mesh.p)
    elif isinstance(mesh, skfem.MeshHex):
        return fix_hexahedron_orientation(mesh.t, mesh.p)
    else:
        raise ValueError("")


def build_element_adjacency_matrix(mesh):
    """
    Returns sparse adjacency matrix A such that A[i, j] = 1
    if element i and j share at least one node.
    """
    num_elements = mesh.nelements
    rows, cols = [], []

    for i, elem_nodes in enumerate(mesh.t.T):
        for node in elem_nodes:
            connected_elements = np.where((mesh.t == node).any(axis=0))[0]
            for j in connected_elements:
                rows.append(i)
                cols.append(j)

    data = np.ones(len(rows), dtype=np.uint8)
    adjacency = csr_matrix(
        (data, (rows, cols)), shape=(num_elements, num_elements)
    )

    return adjacency


def get_adjacent_elements(mesh, element_indices):
    """
    Given a list of element indices, return the set of elements
    that are adjacent (share at least one node) with any of them.
    """
    adjacency = build_element_adjacency_matrix(mesh)
    neighbors = set()

    for idx in element_indices:
        adjacent = adjacency[idx].nonzero()[1]
        neighbors.update(adjacent)

    # exlude original elements
    neighbors.difference_update(element_indices)

    return sorted(neighbors)


def get_elements_by_nodes(
    mesh: skfem.Mesh, target_nodes: np.ndarray | list[np.ndarray]
) -> np.ndarray:
    """
    Fast retrieval of element indices that
     contain any of the given node indices.

    Parameters
    ----------
    mesh : skfem.Mesh
        The mesh object from scikit-fem.
    target_nodes : np.ndarray | list[np.ndarray]
        Array or list of arrays of global node indices.

    Returns
    -------
    elems : np.ndarray
        Sorted, unique array of element indices that
        include any of the target nodes.
    """

    # Initialize cache once
    if not hasattr(get_elements_by_nodes, "_cache"):
        get_elements_by_nodes._cache = {}

    mesh_id = id(mesh)
    if mesh_id not in get_elements_by_nodes._cache:
        # Build: node index → set of element indices
        node_to_elements = defaultdict(set)
        # mesh.t.T: shape = (n_elements, nodes_per_element)
        for e, nodes in enumerate(mesh.t.T):
            for n in nodes:
                node_to_elements[n].add(e)
        get_elements_by_nodes._cache[mesh_id] = node_to_elements

    node_to_elements = get_elements_by_nodes._cache[mesh_id]

    # Normalize target_nodes
    if isinstance(target_nodes, np.ndarray):
        all_target_nodes = np.unique(target_nodes)
    else:
        all_target_nodes = np.unique(np.concatenate(target_nodes))

    # Accumulate matching element indices
    elems = set()
    for node in all_target_nodes:
        elems.update(node_to_elements.get(node, []))

    return np.ascontiguousarray(np.array(sorted(elems), dtype=np.int32))


def build_element_adjacency_matrix_fast(mesh):
    node_to_elements = defaultdict(set)
    for e, nodes in enumerate(mesh.t.T):
        for n in nodes:
            node_to_elements[n].add(e)

    neighbors = defaultdict(set)
    for n, elems in node_to_elements.items():
        for e1 in elems:
            for e2 in elems:
                if e1 != e2:
                    neighbors[e1].add(e2)

    # tocsr
    row, col = [], []
    for e, adj in neighbors.items():
        for a in adj:
            row.append(e)
            col.append(a)

    data = np.ones(len(row), dtype=np.uint8)
    return csr_matrix(
        (data, (row, col)), shape=(mesh.nelements, mesh.nelements)
    )


def get_adjacent_elements_fast(adjacency, element_indices):
    neighbors = set()
    for idx in element_indices:
        adjacent = adjacency[idx].nonzero()[1]
        neighbors.update(adjacent)
    neighbors.difference_update(element_indices)
    return np.array(sorted(neighbors), dtype=np.int32)
