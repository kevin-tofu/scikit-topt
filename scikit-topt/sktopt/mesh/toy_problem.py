import pathlib
import numpy as np
import skfem
from skfem import MeshTet
import meshio
from sktopt.mesh import task
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


def toy_base0(
    mesh_size: float,
    intorder: int = 2
):
    x_len = 8.0
    y_len = 6.0
    z_len = 4.0
    eps = 1.2

    # if True:
    if False:
        mesh = create_box_tet(x_len, y_len, z_len, mesh_size)
        e = skfem.ElementVector(skfem.ElementTetP1())
    else:
        mesh = create_box_hex(x_len, y_len, z_len, mesh_size)
        e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=intorder)
    dirichlet_nodes = utils.get_nodes_indices_in_range(
        basis.mesh, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_dofs = basis.get_dofs(dirichlet_nodes).all()
    F_nodes = utils.get_nodes_indices_in_range(
        basis.mesh,
        (x_len - eps, x_len+0.1), (y_len*2/5, y_len*3/5), (z_len-eps, z_len)
    )
    F_dofs = basis.get_dofs(nodes=F_nodes).nodal['u^3']
    design_elements = utils.get_elements_in_box(
        mesh,
        # (0.3, 0.7), (0.0, 1.0), (0.0, 1.0)
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )

    print("generate config")
    E0 = 210e9
    F = -100.0
    return task.TaskConfig.from_defaults(
        E0,
        0.30,
        basis,
        dirichlet_nodes,
        dirichlet_dofs,
        F_nodes,
        F_dofs,
        F,
        design_elements
    )


def toy_base(
    mesh_size: float,
    intorder: int = 2
):
    x_len = 8.0
    y_len = 6.0
    z_len = 4.0
    eps = 1.2

    mesh = create_box_hex(x_len, y_len, z_len, mesh_size)
    e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=intorder)
    dirichlet_nodes = utils.get_nodes_indices_in_range(
        basis.mesh, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    )
    F_nodes = utils.get_nodes_indices_in_range(
        basis.mesh,
        (x_len - eps, x_len+0.1), (y_len*2/5, y_len*3/5), (z_len-eps, z_len)
    )
    
    design_elements = utils.get_elements_in_box(
        mesh,
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )
    E0 = 210e9
    F = -100.0

    return task.TaskConfig.from_nodes(
        E0,
        0.30,
        basis,
        dirichlet_nodes,
        "all",
        F_nodes,
        "u^3",
        F,
        design_elements
    )


def toy_test():
    return toy_base(1.0)


def toy1():
    return toy_base(0.3)


def toy1_fine():
    return toy_base(0.2)


def load_mesh_auto(msh_path: str):
    msh = meshio.read(msh_path)
    cell_types = [cell.type for cell in msh.cells]
    if "tetra" in cell_types:
        return skfem.MeshTet.load(pathlib.Path(msh_path))
    elif "hexahedron" in cell_types:
        return skfem.MeshHex.load(pathlib.Path(msh_path))
    else:
        raise ValueError("")


def toy2():
    x_len = 8.0
    y_len = 8.0
    z_len = 1.0
    # mesh_size = 0.5
    # mesh_size = 0.3
    mesh_size = 0.2
    mesh = create_box_hex(x_len, y_len, z_len, mesh_size)
    e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=2)
    dirichlet_nodes_0 = utils.get_nodes_indices_in_range(
        basis.mesh, (0.0, 0.05), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_nodes = dirichlet_nodes_0
    dirichlet_dofs = basis.get_dofs(nodes=dirichlet_nodes).all()
    F_nodes_0 = utils.get_nodes_indices_in_range(
        basis.mesh, (x_len, x_len), (y_len, y_len), (0, z_len)
    )
    F_dofs_0 = basis.get_dofs(nodes=F_nodes_0).nodal["u^2"]
    F_nodes_1 = utils.get_nodes_indices_in_range(
        basis.mesh, (x_len, x_len), (0, 0), (0, z_len)
    )
    F_dofs_1 = basis.get_dofs(nodes=F_nodes_1).nodal["u^2"]
    design_elements = utils.get_elements_in_box(
        mesh,
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )

    print("generate config")
    E0 = 210e9
    F = [-300, 300]
    print("F:", F)
    return task.TaskConfig.from_defaults(
        E0,
        0.30,
        basis,
        dirichlet_nodes,
        dirichlet_dofs,
        [F_nodes_0, F_nodes_1],
        [F_dofs_0, F_dofs_1],
        F,
        design_elements
    )


# from memory_profiler import profile
# @profile
def toy_msh(
    task_name: str = "down",
    msh_path: str = 'plate.msh',
):
    if task_name == "down":
        x_len = 4.0
        y_len = 0.16
        # z_len = 1.0
        z_len = 2.0
    elif task_name == "down_box":
        x_len = 4.0
        y_len = 3.0
        z_len = 2.0
    elif task_name == "pull":
        x_len = 8.0
        # x_len = 4.0
        y_len = 3.0
        z_len = 0.5
    elif task_name == "pull_2":
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
    # basis = skfem.Basis(mesh, e, intorder=4)
    # basis = skfem.Basis(mesh, e, intorder=5)
    print("dirichlet_nodes")
    dirichlet_nodes = utils.get_nodes_indices_in_range(
        basis.mesh, (0.0, 0.05), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_dofs = basis.get_dofs(nodes=dirichlet_nodes).all()
    if task_name == "down":
        F_nodes = utils.get_nodes_indices_in_range(
            basis.mesh,
            (x_len-eps, x_len+0.05),
            (y_len/2-eps, y_len/2+eps),
            (0.0, eps)
            # (y_len*2/5, y_len*3/5),
            # (z_len*2/5, z_len*3/5)
        )
        F_dofs = basis.get_dofs(nodes=F_nodes).nodal["u^3"]
        F = -800
    if task_name == "down_box":
        F_nodes = utils.get_nodes_indices_in_range(
            basis.mesh,
            (x_len-eps, x_len+0.05),
            (0, y_len),
            (0.0, eps)
            # (y_len*2/5, y_len*3/5),
            # (z_len*2/5, z_len*3/5)
        )
        F_dofs = basis.get_dofs(nodes=F_nodes).nodal["u^3"]
        F = -800
    elif task_name == "pull":
        F_nodes = utils.get_nodes_indices_in_range(
            basis.mesh,
            (x_len-eps, x_len+0.05),
            (y_len*2/5, y_len*3/5),
            (z_len*2/5, z_len*3/5)
        )
        F_dofs = basis.get_dofs(nodes=F_nodes).nodal["u^1"]
        F = 1200.0
    elif task_name == "pull_2":
        F_nodes = utils.get_nodes_indices_in_range(
            basis.mesh,
            (x_len-eps, x_len+0.05),
            (y_len/2.0-eps, y_len/2+eps),
            (z_len*2/5, z_len*3/5)
        )
        F_dofs = basis.get_dofs(nodes=F_nodes).nodal["u^1"]
        F = 200.0
    design_elements = utils.get_elements_in_box(
        mesh,
        # (0.3, 0.7), (0.0, 1.0), (0.0, 1.0)
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )
    if task_name == "down":
        # removed_elements = utils.get_elements_in_box(
        #     mesh,
        #     (0.0, x_len), (0.0, y_len), (0.0, z_len)
        # )
        # design_elements = np.setdiff1d(design_elements, removed_elements)
        pass

    print("generate config")
    E0 = 210e9
    print("F:", F)
    print("F_nodes:", F_nodes.shape)
    print("F_dofs:", F_dofs.shape)
    return task.TaskConfig.from_defaults(
        E0,
        0.30,
        basis,
        dirichlet_nodes,
        dirichlet_dofs,
        F_nodes,
        F_dofs,
        F,
        design_elements
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
