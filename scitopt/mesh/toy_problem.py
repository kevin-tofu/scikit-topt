import pathlib
import numpy as np
import skfem
from skfem import MeshTet
import meshio
from scitopt.mesh import task
from scitopt.mesh import utils
from scitopt.fea import composer


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

    print("Before mesh.t fix:", mesh.t[:, 0])
    t_fixed = utils.fix_hexahedron_orientation(mesh.t, mesh.p)
    
    print("After fix        :", t_fixed[:, 0])
    mesh_fixed = skfem.MeshHex(mesh.p, t_fixed)
    print("Mesh fixed .t    :", mesh_fixed.t[:, 0])

    return mesh


def create_box_tet(x_len, y_len, z_len, mesh_size):
    max_len = max(x_len, y_len, z_len)
    n_refine = int(np.ceil(np.log2(max_len / mesh_size)))
    mesh = MeshTet().refined(n_refine)
    scale = np.array([x_len, y_len, z_len])
    mesh = mesh.scaled(scale)

    
    print("Before mesh.t fix:", mesh.t[:, 0])
    t_fixed = utils.fix_tetrahedron_orientation(mesh.t, mesh.p)
    # t_fixed = utils.fix_tetrahedron_orientation_numba(mesh.t, mesh.p)
    
    print("After fix        :", t_fixed[:, 0])
    mesh_fixed = MeshTet(mesh.p, t_fixed)
    print("Mesh fixed .t    :", mesh_fixed.t[:, 0])
    # composer._get_elements_volume(mesh_fixed.t, mesh_fixed.p)
    return mesh_fixed


def toy_base(
    mesh_size: float
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
    basis = skfem.Basis(mesh, e, intorder=3)
    dirichlet_points = utils.get_point_indices_in_range(
        basis, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_nodes = utils.get_dofs_in_range(
        basis, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    ).all()
    F_points = utils.get_point_indices_in_range(
        basis, (x_len - eps, x_len+0.1), (y_len*2/5, y_len*3/5), (z_len*2/5, z_len*3/5)
    )
    F_nodes = utils.get_dofs_in_range(
        basis, (x_len - eps, x_len+0.1), (y_len*2/5, y_len*3/5), (z_len*2/5, z_len*3/5)
    ).nodal['u^1']
    design_elements = utils.get_elements_in_box(
        mesh,
        # (0.3, 0.7), (0.0, 1.0), (0.0, 1.0)
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )

    print("generate config")
    E0 = 1.0
    F = 0.3
    return task.TaskConfig.from_defaults(
        E0,
        0.30,
        1e-3 * E0,
        mesh,
        basis,
        dirichlet_points,
        dirichlet_nodes,
        F_points,
        F_nodes,
        F,
        design_elements
    )

def toy_test():
    return toy_base(1.0)

def toy1():
    return toy_base(0.3)


def toy1_fine():
    return toy_base(0.2)


def toy2():
    x_len = 16.0
    y_len = 9.0
    z_len = 2.0
    # mesh_size = 0.5
    # mesh_size = 0.3
    mesh_size = 0.1
    mesh = create_box_hex(x_len, y_len, z_len, mesh_size)
    
    # 
    e = skfem.ElementVector(skfem.ElementTetP1())
    basis = skfem.Basis(mesh, e, intorder=3)
    dirichlet_points = utils.get_point_indices_in_range(
        basis, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_nodes = utils.get_dofs_in_range(
        basis, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    ).all()
    F_points = utils.get_point_indices_in_range(
        basis, (x_len, x_len), (y_len*2/5, y_len*3/5), (z_len*2/5, z_len*3/5)
    )
    F_nodes = utils.get_dofs_in_range(
        basis, (x_len, x_len), (y_len*2/5, y_len*3/5), (z_len*2/5, z_len*3/5)
    ).nodal['u^1']
    design_elements = utils.get_elements_in_box(
        mesh,
        # (0.3, 0.7), (0.0, 1.0), (0.0, 1.0)
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )

    print("generate config")
    E0 = 1.0
    F = [0.3, -0.3]
    print("F:", F)
    return task.TaskConfig.from_defaults(
        E0,
        0.30,
        1e-3 * E0,
        mesh,
        basis,
        dirichlet_points,
        dirichlet_nodes,
        F_points,
        F_nodes,
        F,
        design_elements
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


def toy_msh(
    msh_path: str = 'plate.msh'
):
    from scitopt.fea import composer
    x_len = 4.0
    y_len = 3.0
    # z_len = 1.0
    z_len = 2.0
    # eps = 0.10
    # eps = 0.20
    eps = 0.1
    # eps = 0.5
    mesh = load_mesh_auto(msh_path)

    # Check Index Order.
    # print("Before mesh.t fix:", mesh.t[:, 0])
    # t_fixed = utils.fix_tetrahedron_orientation(mesh.t, mesh.p)
    # print("After fix        :", t_fixed[:, 0])
    # mesh_fixed = MeshTet(mesh.p, t_fixed)
    # print("Mesh fixed .t    :", mesh_fixed.t[:, 0])
    # composer._get_elements_volume(mesh_fixed.t, mesh_fixed.p)
    
    # mesh = skfem.MeshTet.from_mesh(meshio.read(msh_path))
    if isinstance(mesh, skfem.MeshTet):
        e = skfem.ElementVector(skfem.ElementTetP1())
    elif isinstance(mesh, skfem.MeshHex):
        e = skfem.ElementVector(skfem.ElementHex1())
    else:
        raise ValueError("")
    basis = skfem.Basis(mesh, e, intorder=3)
    # basis = skfem.Basis(mesh, e, intorder=4)
    # basis = skfem.Basis(mesh, e, intorder=5)
    
    # 
    
    dirichlet_points = utils.get_point_indices_in_range(
        basis, (0.0, 0.05), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_nodes = utils.get_dofs_in_range(
        basis, (0.0, 0.05), (0.0, y_len), (0.0, z_len)
    ).all()
    # print("dirichlet_nodes:", dirichlet_nodes.shape, dirichlet_nodes.dtype)
    # .nodal['u^1']
    # .all()
    F_points = utils.get_point_indices_in_range(
        basis, (x_len-eps, x_len+0.05),
        (0, y_len),
        (0.0, eps)
        # (y_len*2/5, y_len*3/5),
        # (z_len*2/5, z_len*3/5)
    )
    F_nodes = utils.get_dofs_in_range(
        basis, (x_len-eps, x_len+0.05),
        (0, y_len),
        (0.0, eps)
        # (y_len*2/5, y_len*3/5),
        # (z_len*2/5, z_len*3/5)
    ).nodal['u^1']
    design_elements = utils.get_elements_in_box(
        mesh,
        # (0.3, 0.7), (0.0, 1.0), (0.0, 1.0)
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )
    
    p = basis.mesh.p
    xmax_mask = np.isclose(p[0], 4.0)
    print("generate config")
    E0 = 1.0
    # F = [0.3, -0.3]
    # F = -0.2
    F = 0.002
    # F = 0.3
    
    # F = 0.3
    # F = 1.2
    # F = 0.4
    # F = 1.0
    # F = 150.0
    print("F:", F)
    print("F_points:", F_points.shape)
    print("F_nodes:", F_nodes.shape)
    
    return task.TaskConfig.from_defaults(
        E0,
        0.30,
        1e-3 * E0,
        mesh,
        basis,
        dirichlet_points,
        dirichlet_nodes,
        F_points,
        F_nodes,
        F,
        design_elements
    )
    
    
if __name__ == '__main__':

    import scitopt
    from scitopt.fea import solver
    tsk = toy()
    
    rho = np.ones_like(tsk.design_elements)
    p = 3.0
    compliance, u = solver.compute_compliance_basis_numba(
        tsk.basis,
        tsk.free_nodes, tsk.dirichlet_nodes, tsk.force,
        tsk.E0, tsk.Emin, p, tsk.nu0, rho
    )
    print("compliance: ", compliance)