import os
import glob
from typing import Optional

import numpy as np
import imageio.v2 as imageio
import skfem
import meshio
import pyvista as pv
import matplotlib.pyplot as plt


def export_mesh_with_info(
    mesh: skfem.Mesh,
    point_data_values: Optional[list[np.ndarray]] = None,
    point_data_names: Optional[list[str]] = None,
    cell_data_values: Optional[list[np.ndarray]] = None,
    cell_data_names: Optional[list[str]] = None,
    filepath: str = "output.vtu"
):
    """
    Export a skfem.Mesh object and its data to a VTU file via meshio.

    Parameters
    ----------
    mesh : skfem.Mesh
        The finite element mesh object (MeshTet, MeshTri, MeshHex, etc.).
    
    point_data_values : list of np.ndarray, optional
        List of arrays of point-wise data (length = n_nodes).
    
    point_data_names : list of str, optional
        Names for each point-wise data array.
    
    cell_data_values : list of np.ndarray, optional
        List of arrays of cell-wise data (length = n_elements).
    
    cell_data_names : list of str, optional
        Names for each cell-wise data array.
    
    filepath : str
        Output filename (e.g., "result.vtu").
    """
    # Determine element type
    if isinstance(mesh, skfem.MeshTet):
        cell_type = "tetra"
    elif isinstance(mesh, skfem.MeshHex):
        cell_type = "hexahedron"
    else:
        raise ValueError(f"Unsupported mesh type: {type(mesh)}")

    # Convert skfem's (dim, n) shape to meshio format
    points = mesh.p.T
    # cells = [(cell_type, mesh.t.T)]

    # Point data
    point_data = {}
    if point_data_values and point_data_names:
        for name, val in zip(point_data_names, point_data_values):
            point_data[name] = val

    # Cell data (wrap in a list for each cell block)
    cell_data = {}
    if cell_data_values and cell_data_names:
        for name, val in zip(cell_data_names, cell_data_values):
            cell_data[name] = [val]

    # Build and write meshio.Mesh
    meshio_mesh = meshio.Mesh(
        points=points,
        cells=[meshio.CellBlock(cell_type, mesh.t.T)],
        point_data=point_data,
        cell_data=cell_data
    )
    meshio_mesh.write(filepath)


def write_mesh_with_info_as_image(
    mesh_path: str,
    mesh_scalar_name: str,
    clim: tuple,
    image_path: str,
    image_title: str
):
    """
    Render a scalar field on a VTU mesh and save it as an image using PyVista.

    Parameters
    ----------
    mesh_path : str
        Path to the VTU mesh file to be read.

    mesh_scalar_name : str
        The name of the scalar field stored in the mesh to be visualized (must match one of the scalar field names in the file).

    clim : tuple of float
        Color limits (min, max) for the scalar colormap.

    image_path : str
        File path where the rendered image (e.g., PNG) will be saved.

    image_title : str
        Title text to display in the upper left corner of the image.

    Notes
    -----
    - Requires the `pyvista` package.
    - Works in headless environments by starting an off-screen xvfb session.
    - Assumes the scalar field is stored in cell data or point data with the specified name.
    """
    if os.path.exists(mesh_path):
        pv.start_xvfb()
        mesh = pv.read(mesh_path)
        plotter = pv.Plotter(off_screen=True)
        add_mesh_params = dict(
            scalars=mesh_scalar_name,
            cmap="cividis",
            clim=clim,
            show_edges=True,
            scalar_bar_args={"title": mesh_scalar_name}
        )
        # if opaque is True:
        #     add_mesh_params["opacity"] = (rho > 1e-1).astype(float)
        plotter.add_mesh(
            mesh, **add_mesh_params
        )
        plotter.add_text(
            image_title, position="upper_left", font_size=12, color="black"
        )
        plotter.screenshot(image_path)
        plotter.close()
    else:
        raise ValueError(f"mesh: {mesh_path} does not exist.")


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
    prefix: str = "rho",
    scale: float = 0.7,
    skip_frame: int = 0
):
    from scipy.ndimage import zoom

    file_pattern = f"{dir_path}/mesh_rho/info_{prefix}-*.jpg"
    image_files = sorted(glob.glob(file_pattern))
    if skip_frame > 0:
        image_files = image_files[::skip_frame+1]

    output_gif = os.path.join(dir_path, f"animation-{prefix}.gif")
    if len(image_files) > 0:
        with imageio.get_writer(
            output_gif, mode='I', duration=0.2, loop=0
        ) as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                image_small = zoom(image, (scale, scale, 1))  # (H, W, C)
                image_small = image_small.astype("uint8")
                writer.append_data(image_small)
                # writer.append_data(image)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument(
        '--images_path', '-IP', type=str, default="./result/test1_oc2", help=''
    )
    parser.add_argument(
        '--scale', '-SL', type=float, default=0.50, help=''
    )
    parser.add_argument(
        '--skip_frame', '-SF', type=int, default=0, help=''
    )
    args = parser.parse_args()
    images2gif(
        f"{args.images_path}", "rho", scale=args.scale,
        skip_frame=args.skip_frame
    )
    images2gif(
        f"{args.images_path}", "dC", scale=args.scale,
        skip_frame=args.skip_frame
    )
