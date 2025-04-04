import numpy as np
import meshio
import matplotlib.pyplot as plt
import scitopt
from scitopt.mesh import utils


def save_info_on_mesh(
    tsk,
    rho: np.ndarray,
    rho_prev: np.ndarray,
    file_path='levelset.vtk'
):
    
    mesh = tsk.mesh
    dirichlet_ele = utils.get_elements_with_points(mesh, [tsk.dirichlet_points])
    F_ele = utils.get_elements_with_points(mesh, [tsk.force_points])
    element_colors_df1 = np.zeros(mesh.nelements, dtype=int)
    element_colors_df2 = np.zeros(mesh.nelements, dtype=int)
    element_colors_df1[tsk.design_elements] = 1
    element_colors_df1[tsk.fixed_elements_in_rho] = 2
    element_colors_df2[dirichlet_ele] = 1
    element_colors_df2[F_ele] = 2
    
    # rho_projected = techniques.heaviside_projection(
    #     rho, beta=beta, eta=eta
    # )
    cell_outputs = dict()
    cell_outputs["rho"] = [rho]
    cell_outputs["rho-diff"] = [rho - rho_prev]
    # cell_outputs["rho_projected"] = [rho_projected]
    cell_outputs["desing-fixed"] = [element_colors_df1]
    cell_outputs["condition"] = [element_colors_df2]
    # if sigma_v is not None:
    #     cell_outputs["sigma_v"] = [sigma_v]
    
    meshio_mesh = meshio.Mesh(
        points=mesh.p.T,
        cells=[("tetra", mesh.t.T)],
        cell_data=cell_outputs
    )
    meshio.write(file_path, meshio_mesh)


def export_submesh(
    tsk: scitopt.mesh.TaskConfig,
    rho_projected: np.ndarray,
    threshold: float,
    dst_path: str
):
    mesh = tsk.mesh
    remove_elements = tsk.design_elements[rho_projected[tsk.design_elements] <= threshold]
    kept_elements = scitopt.mesh.task.setdiff1d(tsk.all_elements, remove_elements)
    kept_t = mesh.t[:, kept_elements]
    unique_vertex_indices = np.unique(kept_t)
    new_points = np.ascontiguousarray(mesh.p[:, unique_vertex_indices])
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
    new_elements = np.vectorize(index_map.get)(kept_t)
    new_elements = np.ascontiguousarray(new_elements)
    meshtype = type(mesh)
    submesh = meshtype(new_points, new_elements)
    submesh.save(dst_path)


def rho_histo_plot(
    rho: np.ndarray,
    dst_path: str
):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(rho.flatten(), bins=50)
    ax.set_xlabel("Density (rho)")
    ax.set_ylabel("Number of Elements")
    ax.set_title("Density Distribution")
    ax.grid(True)
    fig.savefig(dst_path)
    plt.close("all")
