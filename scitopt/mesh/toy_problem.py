import pathlib
import numpy as np
import skfem
import gmsh
from scitopt.mesh import task
from scitopt.mesh import utils


def create_box(x_len, y_len, z_len, mesh_size):
    print("generate mesh")
    gmsh.initialize()
    gmsh.model.add('plate')
    gmsh.model.occ.addBox(0, 0, 0, x_len, y_len, z_len)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.generate(3)
    # gmsh.write("plate.msh")
    # gmsh.finalize()

    print("load mesh")
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, node_tags_for_elements = gmsh.model.mesh.getElements(3)
    tet_type = None
    for etype in element_types:
        if etype in [4, 11]:  # 4: linear tetrahedra, 11: quadratic
            tet_type = etype
            break
    if tet_type is None:
        raise ValueError("Tetrahedral elements not found.")
    # indices correponds to tet_type
    idx = np.where(element_types == tet_type)[0][0]
    # convert node indicers with 0-based indexing
    t = np.array(node_tags_for_elements[idx], dtype=np.int32).reshape(-1, 4).T - 1
    p = node_coords.reshape(-1, 3).T
    t = np.ascontiguousarray(t)
    p = np.ascontiguousarray(p)
    mesh = skfem.MeshTet(p, t)
    # mesh = skfem.MeshTet.load(pathlib.Path('plate.msh'))
    gmsh.finalize()
    return mesh


def toy_base(mesh_size: float):
    x_len = 16.0
    y_len = 9.0
    z_len = 4.0
    mesh = create_box(x_len, y_len, z_len, mesh_size)

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
    ).nodal['u^2']
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
    mesh = create_box(x_len, y_len, z_len, mesh_size)
    
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
    ).nodal['u^2']
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