import pathlib
import numpy as np
import skfem
from skfem import MeshTet
import meshio
from sktopt.mesh import task_elastic
from sktopt.mesh import utils


def create_box_hex(x_len, y_len, z_len, mesh_size):
    """
    Create a hexahedral mesh box with given dimensions and element size.

    Parameters
    ----------
    x_len, y_len, z_len : float
        Dimensions of the box in x, y, z directions.
    mesh_size : float
        Desired approximate size of each hex element.

    Returns
    -------
    mesh : MeshHex
        A scaled MeshHex object with the specified dimensions.
    """
    nx = int(np.ceil(x_len / mesh_size))
    ny = int(np.ceil(y_len / mesh_size))
    nz = int(np.ceil(z_len / mesh_size))

    x = np.linspace(0, x_len, nx + 1)
    y = np.linspace(0, y_len, ny + 1)
    z = np.linspace(0, z_len, nz + 1)

    mesh = skfem.MeshHex.init_tensor(x, y, z)
    t_fixed = utils.fix_hexahedron_orientation(mesh.t, mesh.p)
    mesh_fixed = skfem.MeshHex(mesh.p, t_fixed)
    return mesh_fixed


def create_box_tet(x_len, y_len, z_len, mesh_size):
    max_len = max(x_len, y_len, z_len)
    n_refine = int(np.ceil(np.log2(max_len / mesh_size)))
    mesh = MeshTet().refined(n_refine)
    scale = np.array([x_len, y_len, z_len])
    mesh = mesh.scaled(scale)
    t_fixed = utils.fix_tetrahedron_orientation(mesh.t, mesh.p)
    mesh_fixed = MeshTet(mesh.p, t_fixed)
    return mesh_fixed


def toy_base(
    mesh_size: float,
    intorder: int = 2
):
    x_len = 8.0
    y_len = 6.0
    z_len = 4.0
    eps = 1.2
    if False:
        mesh = create_box_tet(x_len, y_len, z_len, mesh_size)
        e = skfem.ElementVector(skfem.ElementTetP1())
    else:
        mesh = create_box_hex(x_len, y_len, z_len, mesh_size)
        e = skfem.ElementVector(skfem.ElementHex1())
    dirichlet_in_range = utils.get_points_in_range(
        (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_facets = mesh.facets_satisfying(dirichlet_in_range)
    force_in_range = utils.get_points_in_range(
        (x_len - eps, x_len+0.1), (y_len*2/5, y_len*3/5), (z_len-eps, z_len)
    )
    force_facets = mesh.facets_satisfying(force_in_range)
    desing_in_range = utils.get_points_in_range(
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )
    design_elements = mesh.elements_satisfying(desing_in_range)
    E0 = 210e3
    F = -100.0
    e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=2)
    return task_elastic.LinearElasticity.from_facets(
        basis,
        dirichlet_facets,
        "all",
        force_facets,
        "u^3",
        F,
        design_elements,
        E0,
        0.30,
    )


def toy_test():
    return toy_base(1.0)


def toy1():
    return toy_base(0.3)


def toy1_fine():
    return toy_base(0.2)


def toy2():
    x_len = 8.0
    y_len = 8.0
    z_len = 1.0
    mesh_size = 0.3
    # mesh = create_box_tet(x_len, y_len, z_len, mesh_size)
    # e = skfem.ElementVector(skfem.ElementTetP1())
    mesh = create_box_hex(x_len, y_len, z_len, mesh_size)
    e = skfem.ElementVector(skfem.ElementHex1())
    dirichlet_in_range = utils.get_points_in_range(
        (0.0, 0.05), (0.0, y_len), (0.0, z_len)
    )
    eps = mesh_size
    neumann_in_range_0 = utils.get_points_in_range(
        (x_len, x_len), (y_len-eps, y_len), (0, z_len)
    )
    neumann_in_range_1 = utils.get_points_in_range(
        (x_len, x_len), (0, eps), (0, z_len)
    )
    neumann_dir_type = ["u^2", "u^2"]
    neumann_value = [-1.0, 1.0]
    boundaries = {
        "dirichlet": dirichlet_in_range,
        "neumann_0": neumann_in_range_0,
        "neumann_1": neumann_in_range_1
    }
    mesh = mesh.with_boundaries(boundaries)
    subdomains = {"design": np.array(range(mesh.nelements))}
    mesh = mesh.with_subdomains(subdomains)
    basis = skfem.Basis(mesh, e, intorder=2)
    E0 = 210e3
    return task_elastic.LinearElasticity.from_mesh_tags(
        basis,
        "all",
        neumann_dir_type,
        neumann_value,
        E0,
        0.30,
    )


def load_mesh_auto(msh_path: str):
    msh = meshio.read(msh_path)
    cell_types = [cell.type for cell in msh.cells]
    if "tetra" in cell_types:
        return skfem.MeshTet.load(pathlib.Path(msh_path))
    elif "hexahedron" in cell_types:
        return skfem.MeshHex.load(pathlib.Path(msh_path))
    else:
        raise ValueError("")


# from memory_profiler import profile
# @profile
def toy_msh(
    task_elastic_name: str = "down",
    msh_path: str = 'plate.msh',
):
    if task_elastic_name == "down":
        x_len = 4.0
        y_len = 0.16
        # z_len = 1.0
        z_len = 2.0
    elif task_elastic_name == "down_box":
        x_len = 4.0
        y_len = 3.0
        z_len = 2.0
    elif task_elastic_name == "pull":
        x_len = 8.0
        # x_len = 4.0
        y_len = 3.0
        z_len = 0.5
    elif task_elastic_name == "pull_2":
        # x_len = 8.0
        x_len = 4.0
        # x_len = 6.0
        y_len = 2.0
        z_len = 0.5
    # eps = 0.10
    # eps = 0.20
    eps = 0.03
    # eps = 0.5
    mesh = load_mesh_auto(msh_path)
    coords = mesh.p.T  # (n_nodes, dim)
    a_point = mesh.p.T[0]
    dists = np.linalg.norm(coords[1::] - a_point, axis=1)
    eps = np.min(dists) * 0.8
    # eps = np.min(dists) * 1.2
    # eps = np.min(dists) * 5.0
    print(f"eps: {eps}")
    # mesh = skfem.MeshTet.from_mesh(meshio.read(msh_path))
    if isinstance(mesh, skfem.MeshTet):
        e = skfem.ElementVector(skfem.ElementTetP1())
    elif isinstance(mesh, skfem.MeshHex):
        e = skfem.ElementVector(skfem.ElementHex1())
    else:
        raise ValueError("")
    print("basis")
    # basis = skfem.Basis(mesh, e, intorder=2)
    basis = skfem.Basis(mesh, e, intorder=3)
    dirichlet_in_range = utils.get_points_in_range(
        (0.0, 0.05), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_facets = basis.mesh.facets_satisfying(dirichlet_in_range)
    if task_elastic_name == "down":
        x_range = (x_len-eps, x_len+0.05)
        y_range = (y_len/2-eps, y_len/2+eps)
        z_range = (0.0, eps)
        force_dir_type = "u^3"
        force_value = -800
    if task_elastic_name == "down_box":
        x_range = (x_len-eps, x_len+0.05)
        y_range = (0, y_len)
        z_range = (0.0, eps)
        force_dir_type = "u^3"
        force_value = -800

    elif task_elastic_name == "pull":
        x_range = (x_len-eps, x_len+0.05),
        y_range = (y_len*2/5, y_len*3/5)
        z_range = (z_len*2/5, z_len*3/5)
        force_dir_type = "u^1"
        force_value = 1200.0
    elif task_elastic_name == "pull_2":
        x_range = (x_len-eps, x_len+0.05),
        y_range = (y_len/2.0-eps, y_len/2+eps),
        z_range = (z_len*2/5, z_len*3/5)
        force_dir_type = "u^1"
        force_value = 200.0

    force_in_range = utils.get_points_in_range(x_range, y_range, z_range)
    force_facets = basis.mesh.facets_satisfying(force_in_range)
    desing_in_range = utils.get_points_in_range(
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )
    design_elements = basis.mesh.elements_satisfying(desing_in_range)
    print("generate config")
    E0 = 210e3
    return task_elastic.LinearElasticity.from_facets(
        basis,
        dirichlet_facets,
        "all",
        force_facets,
        force_dir_type,
        force_value,
        design_elements,
        E0,
        0.30,
    )


if __name__ == '__main__':

    from sktopt.fea import solver
    tsk = toy1()
    rho = np.ones_like(tsk.design_elements)
    p = 3.0
    compliance, u = solver.compute_compliance_basis_numba(
        tsk.basis,
        tsk.free_dofs, tsk.dirichlet_dofs, tsk.force,
        tsk.E0, tsk.Emin, p, tsk.nu0, rho
    )
    print("compliance: ", compliance)
