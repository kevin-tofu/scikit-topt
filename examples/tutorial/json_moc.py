import numpy as np
import skfem
from skfem.io.json import to_file, from_file
# import meshio

import sktopt


def msh_export():

    x_len = 8.0
    y_len = 6.0
    z_len = 4.0
    mesh_size = 0.2
    eps = mesh_size / 2.0

    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, mesh_size
    )
    dirichlet_in_range = sktopt.mesh.utils.get_points_in_range(
        (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    )
    neumann_in_range = sktopt.mesh.utils.get_points_in_range(
        (x_len - eps, x_len+0.1), (y_len*2/5, y_len*3/5), (z_len-eps, z_len)
    )
    boundaries = {
        "dirichlet": dirichlet_in_range,
        "neumann": neumann_in_range,
    }
    subdomains = {"design": np.arange(mesh.nelements)}

    mesh = mesh.with_boundaries(boundaries)
    mesh = mesh.with_subdomains(subdomains)
    to_file(mesh, "./mesh.json")


def get_task():

    # mesh_path = "./tagged_mesh.msh"
    # basis = sktopt.mesh.loader.basis_from_file(
    #   mesh_path, intorder=1
    # )
    mesh = from_file("./mesh.json")
    e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=1)
    dirichlet_dir = "all"
    neumann_dir_type = "u^3"
    neumann_value = 100
    # Define it as a task
    tsk = sktopt.mesh.LinearElastisicity.from_mesh_tags(
        210e9,
        0.30,
        basis,
        dirichlet_dir,
        neumann_dir_type,
        neumann_value
    )
    return tsk


def get_cfg():

    cfg = sktopt.core.optimizers.LogMOC_Config(
        dst_path="./result/tutorial_json_moc",
        export_img=True,
        p=sktopt.tools.SchedulerConfig.constant(
            target_value=3.0
        ),
        vol_frac=sktopt.tools.SchedulerConfig.constant(
            target_value=0.6
        ),
    )
    return cfg


def main():

    msh_export()
    tsk = get_task()
    cfg = get_cfg()
    optimizer = sktopt.core.LogMOC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()


if __name__ == '__main__':
    main()
