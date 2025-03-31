import numpy as np
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


def get_point_indices_in_range(
    basis: skfem.Basis, x_range: tuple, y_range: tuple, z_range: tuple
):
    x = basis.mesh.p  # (3, n_points)
    mask = (
        (x_range[0] <= x[0]) & (x[0] <= x_range[1]) &
        (y_range[0] <= x[1]) & (x[1] <= y_range[1]) &
        (z_range[0] <= x[2]) & (x[2] <= z_range[1])
    )

    return np.where(mask)[0]


def get_dofs_in_range(
    basis: skfem.Basis, x_range: tuple, y_range: tuple, z_range: tuple
):
    return basis.get_dofs(
        lambda x: (x_range[0] <= x[0]) & (x[0] <= x_range[1]) &
                  (y_range[0] <= x[1]) & (x[1] <= y_range[1]) &
                  (z_range[0] <= x[2]) & (x[2] <= z_range[1])
    )

def get_elements_in_box(
    mesh: skfem.Mesh,
    x_range: tuple,
    y_range: tuple,
    z_range: tuple
) -> np.ndarray:
    """
    Returns the indices of elements whose centers lie within the specified rectangular box.

    Parameters:
        mesh (skfem.Mesh): The mesh object.
        x_range (tuple): Range in the x-direction (xmin, xmax).
        y_range (tuple): Range in the y-direction (ymin, ymax).
        z_range (tuple): Range in the z-direction (zmin, zmax).

    Returns:
        np.ndarray: Array of indices of elements that satisfy the given conditions.

    """
    # element_centers = mesh.p[:, mesh.t].mean(axis=0)
    element_centers = np.array([np.mean(mesh.p[:, elem], axis=1) for elem in mesh.t.T]).T

    mask = (
        (x_range[0] <= element_centers[0]) & (element_centers[0] <= x_range[1]) &
        (y_range[0] <= element_centers[1]) & (element_centers[1] <= y_range[1]) &
        (z_range[0] <= element_centers[2]) & (element_centers[2] <= z_range[1])
    )

    return np.where(mask)[0]
