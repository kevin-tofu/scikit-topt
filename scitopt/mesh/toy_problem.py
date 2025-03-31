import pathlib
import skfem
from scitopt.mesh import task
from scitopt.mesh import utils


def toy():
    import gmsh

    gmsh.initialize()
    x_len = 16.0
    y_len = 9.0
    z_len = 5.0
    mesh_size = 0.5
    # mesh_size = 0.3

    gmsh.model.add('plate')
    gmsh.model.occ.addBox(0, 0, 0, x_len, y_len, z_len)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.generate(3)
    gmsh.write("plate.msh")
    gmsh.finalize()

    # mesh = skfem.MeshTet().refined(4).with_defaults()
    mesh = skfem.MeshTet.load(pathlib.Path('plate.msh'))
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