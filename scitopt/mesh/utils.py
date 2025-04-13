from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
import skfem


import numpy as np

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



def get_elements_with_points(mesh: skfem.mesh, target_nodes_list: list[np.ndarray]) -> np.ndarray:
    """
    """
    all_target_nodes = np.unique(np.concatenate(target_nodes_list))
    mask = np.any(np.isin(mesh.t, all_target_nodes), axis=0)
    return np.where(mask)[0]


def get_elements_without_points(mesh: skfem.mesh, excluded_nodes_list: list[np.ndarray]):
    """
    """
    all_excluded_nodes = np.unique(np.concatenate(excluded_nodes_list))
    mask = ~np.any(np.isin(mesh.t, all_excluded_nodes), axis=0)
    return np.where(mask)[0]


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
    adjacency = csr_matrix((data, (rows, cols)), shape=(num_elements, num_elements))

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


def get_elements_with_points_fast(mesh, target_nodes):
    if not hasattr(get_elements_with_points_fast, "_cache"):
        get_elements_with_points_fast._cache = {}

    mesh_id = id(mesh)
    if mesh_id not in get_elements_with_points_fast._cache:
        # 構築: node → set of element indices
        node_to_elements = defaultdict(set)
        for e, nodes in enumerate(mesh.t.T):
            for n in nodes:
                node_to_elements[n].add(e)
        get_elements_with_points_fast._cache[mesh_id] = node_to_elements

    node_to_elements = get_elements_with_points_fast._cache[mesh_id]

    all_target_nodes = np.unique(np.concatenate(target_nodes))
    elems = set()
    for node in all_target_nodes:
        elems.update(node_to_elements.get(node, []))

    # return np.array(sorted(elems), dtype=np.int32)
    return np.ascontiguousarray(np.array(sorted(elems), dtype=np.int32))


def get_elements_with_nodes_fast(
    mesh: skfem.Mesh, target_nodes: np.ndarray | list[np.ndarray]
) -> np.ndarray:
    """
    Fast retrieval of element indices that contain any of the given node indices.

    Parameters
    ----------
    mesh : skfem.Mesh
        The mesh object from scikit-fem.
    target_nodes : np.ndarray | list[np.ndarray]
        Array or list of arrays of global node indices.

    Returns
    -------
    elems : np.ndarray
        Sorted, unique array of element indices that include any of the target nodes.
    """

    # Initialize cache once
    if not hasattr(get_elements_with_nodes_fast, "_cache"):
        get_elements_with_nodes_fast._cache = {}

    mesh_id = id(mesh)
    if mesh_id not in get_elements_with_nodes_fast._cache:
        # Build: node index → set of element indices
        node_to_elements = defaultdict(set)
        for e, nodes in enumerate(mesh.t.T):  # mesh.t.T: shape = (n_elements, nodes_per_element)
            for n in nodes:
                node_to_elements[n].add(e)
        get_elements_with_nodes_fast._cache[mesh_id] = node_to_elements

    node_to_elements = get_elements_with_nodes_fast._cache[mesh_id]

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
    return csr_matrix((data, (row, col)), shape=(mesh.nelements, mesh.nelements))


def get_adjacent_elements_fast(adjacency, element_indices):
    neighbors = set()
    for idx in element_indices:
        adjacent = adjacency[idx].nonzero()[1]
        neighbors.update(adjacent)
    neighbors.difference_update(element_indices)
    return np.array(sorted(neighbors), dtype=np.int32)


def get_boundary_nodes_from_elements(elements: np.ndarray, mesh: skfem.Mesh, dirichlet_nodes: np.ndarray) -> np.ndarray:
    """
    Given a set of element indices, return the indices of boundary (Dirichlet) nodes contained in those elements.

    Parameters
    ----------
    elements : np.ndarray
        Array of element indices (e.g., bc_force_elements).
    mesh : skfem.Mesh
        The skfem mesh object.
    dirichlet_nodes : np.ndarray
        Global indices of nodes where Dirichlet boundary conditions are applied.

    Returns
    -------
    boundary_nodes : np.ndarray
        Unique indices of boundary nodes contained in the specified elements.
    """
    element_nodes = mesh.t[:, elements].flatten()
    boundary_nodes = np.intersect1d(element_nodes, dirichlet_nodes)
    return boundary_nodes


def in_box(coords, x_range, y_range, z_range):
    return (
        (x_range[0] <= coords[0]) & (coords[0] <= x_range[1]) &
        (y_range[0] <= coords[1]) & (coords[1] <= y_range[1]) &
        (z_range[0] <= coords[2]) & (coords[2] <= z_range[1])
    )

def get_point_indices_in_range(
    basis: skfem.Basis, x_range: tuple, y_range: tuple, z_range: tuple
) -> np.ndarray:
    mask = in_box(basis.mesh.p, x_range, y_range, z_range)
    return np.flatnonzero(mask)


def get_dofs_in_range(
    basis: skfem.Basis, x_range: tuple, y_range: tuple, z_range: tuple
) -> np.ndarray:
    return basis.get_dofs(
        lambda x: in_box(x, x_range, y_range, z_range)
    )

def get_elements_in_box(
    mesh: skfem.Mesh,
    x_range: tuple,
    y_range: tuple,
    z_range: tuple
) -> np.ndarray:
    element_centers = mesh.p[:, mesh.t].mean(axis=1)
    mask = in_box(element_centers, x_range, y_range, z_range)
    return np.flatnonzero(mask)


# def get_point_indices_in_range(
#     basis: skfem.Basis, x_range: tuple, y_range: tuple, z_range: tuple
# ):
#     x = basis.mesh.p  # (3, n_points)
#     mask = (
#         (x_range[0] <= x[0]) & (x[0] <= x_range[1]) &
#         (y_range[0] <= x[1]) & (x[1] <= y_range[1]) &
#         (z_range[0] <= x[2]) & (x[2] <= z_range[1])
#     )

#     return np.where(mask)[0]


# def get_dofs_in_range(
#     basis: skfem.Basis, x_range: tuple, y_range: tuple, z_range: tuple
# ):
#     return basis.get_dofs(
#         lambda x: (x_range[0] <= x[0]) & (x[0] <= x_range[1]) &
#                   (y_range[0] <= x[1]) & (x[1] <= y_range[1]) &
#                   (z_range[0] <= x[2]) & (x[2] <= z_range[1])
#     )

# def get_elements_in_box(
#     mesh: skfem.Mesh,
#     x_range: tuple,
#     y_range: tuple,
#     z_range: tuple
# ) -> np.ndarray:
#     """
#     Returns the indices of elements whose centers lie within the specified rectangular box.

#     Parameters:
#         mesh (skfem.Mesh): The mesh object.
#         x_range (tuple): Range in the x-direction (xmin, xmax).
#         y_range (tuple): Range in the y-direction (ymin, ymax).
#         z_range (tuple): Range in the z-direction (zmin, zmax).

#     Returns:
#         np.ndarray: Array of indices of elements that satisfy the given conditions.

#     """
#     # element_centers = mesh.p[:, mesh.t].mean(axis=0)
#     element_centers = np.array([np.mean(mesh.p[:, elem], axis=1) for elem in mesh.t.T]).T

#     mask = (
#         (x_range[0] <= element_centers[0]) & (element_centers[0] <= x_range[1]) &
#         (y_range[0] <= element_centers[1]) & (element_centers[1] <= y_range[1]) &
#         (z_range[0] <= element_centers[2]) & (element_centers[2] <= z_range[1])
#     )

#     return np.where(mask)[0]


