import os
import glob

from typing import Optional
import imageio.v2 as imageio
    
import numpy as np
import meshio
import pyvista as pv
import matplotlib.pyplot as plt
import scitopt
from scitopt.mesh import utils


def rank_scale_0_1(dc: np.ndarray) -> np.ndarray:
    """
    Scale dC to range [0.0, 1.0] based on rank (by absolute value).

    Parameters
    ----------
    dc : np.ndarray
        Sensitivity array (1D)

    Returns
    -------
    scaled : np.ndarray
        Ranked and scaled array with values in [0.0, 1.0]
    """
    abs_dc = np.abs(dc)
    ranks = np.argsort(np.argsort(abs_dc))  # rank: 0 (min) → n-1 (max)
    
    n = len(dc)
    scaled = ranks / (n - 1 + 1e-8)  # avoid divide-by-zero

    return scaled



def save_info_on_mesh(
    tsk, 
    rho: np.ndarray,
    rho_prev: np.ndarray,
    dC: Optional[np.ndarray]=None,
    mesh_path: str='mesh.vtu',
    rho_image_path: Optional[str]=None,
    rho_image_title: Optional[str]=None,
    dC_image_path: Optional[str]=None,
    dC_image_title: Optional[str]=None
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
    if dC is not None:
        dC[tsk.fixed_elements_in_rho] = 0.0
        dC_ranked = rank_scale_0_1(dC)
        cell_outputs["strain_energy_ranked"] = [dC_ranked]
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
    meshio.write(mesh_path, meshio_mesh)
    
    # 
    if isinstance(rho_image_path, str):
        pv.start_xvfb()
        mesh = pv.read(mesh_path)
        # scalar_names = list(mesh.cell_data.keys())
        scalar_name = "rho"
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            cmap="turbo",
            clim=(0, 1),
            opacity=0.3,
            show_edges=False,
            scalar_bar_args={"title": scalar_name}
        )
        plotter.add_text(rho_image_title, position="upper_left", font_size=12, color="black")
        plotter.screenshot(rho_image_path)
        plotter.close()

    if isinstance(dC_image_path, str):
        pv.start_xvfb()
        mesh = pv.read(mesh_path)
        # scalar_names = list(mesh.cell_data.keys())
        scalar_name = "strain_energy_ranked"
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            cmap="turbo",
            clim=(0, 1),
            opacity=0.3,
            show_edges=False,
            scalar_bar_args={"title": scalar_name}
        )
        plotter.add_text(dC_image_title, position="upper_left", font_size=12, color="black")
        plotter.screenshot(dC_image_path)
        plotter.close()

    
# def save_info_on_mesh_as_image(
#     tsk, 
#     info: list[np.ndarray],
#     info_name: str,
#     mesh_path: str,
#     image_path :str,
#     image_title: str
# ):
    
#     mesh = tsk.mesh
#     element_colors_df1 = np.zeros_like(info[0])
#     element_colors_df1[tsk.design_elements] = 1
#     element_colors_df1[tsk.fixed_elements_in_rho] = 0.0
    
#     cell_outputs = dict()
#     cell_outputs[info_name] = info
#     meshio_mesh = meshio.Mesh(
#         points=mesh.p.T,
#         cells=[("tetra", mesh.t.T)],
#         cell_data=cell_outputs
#     )
#     meshio.write(mesh_path, meshio_mesh)
    
#     pv.start_xvfb()
#     mesh = pv.read(mesh_path)
#     scalar_name = "rho"
#     plotter = pv.Plotter(off_screen=True)
#     plotter.add_mesh(
#         mesh,
#         scalars=scalar_name,
#         cmap="turbo",
#         clim=(0, 1),
#         opacity=0.3,
#         show_edges=False,
#         scalar_bar_args={"title": scalar_name}
#     )
#     plotter.add_text(image_title, position="upper_left", font_size=12, color="black")
#     plotter.screenshot(image_path)


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


def images2gif(
    dir_path: str,
    prefix: str="rho"
):
    file_pattern = f"{dir_path}/mesh_rho/info_{prefix}-*.jpg"
    image_files = sorted(glob.glob(file_pattern))
    output_gif = os.path.join(dir_path, f"animation-{prefix}.gif")
    
    if len(image_files) > 0:
        with imageio.get_writer(output_gif, mode='I', duration=0.2) as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                writer.append_data(image)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument(
        '--images_path', '-IP', type=str, default="./result/test1_oc2", help=''
    )
    args = parser.parse_args()
    images2gif(f"{args.images_path}", "rho")
    images2gif(f"{args.images_path}", "dC")
    