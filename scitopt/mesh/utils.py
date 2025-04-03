from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
import skfem


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



def in_box(coords, x_range, y_range, z_range):
    """汎用的な座標範囲フィルタ"""
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


