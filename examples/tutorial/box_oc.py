import numpy as np
import skfem
import sktopt


def get_task():

    x_len = 8.0
    y_len = 8.0
    z_len = 1.0
    mesh_size = 0.2

    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, mesh_size
    )
    e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=2)
    dirichlet_in_range = sktopt.mesh.utils.get_points_in_range(
        (0.0, 0.05), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_dir = "all"

    eps = mesh_size
    neumann_in_range_0 = sktopt.mesh.utils.get_points_in_range(
        (x_len, x_len), (y_len-eps, y_len), (0, z_len)
    )
    neumann_in_range_1 = sktopt.mesh.utils.get_points_in_range(
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

    e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=2)
    E0 = 210e3
    mytask = sktopt.mesh.LinearElastisicity.from_mesh_tags(
        basis,
        dirichlet_dir,
        neumann_dir_type,
        neumann_value,
        E0,
        0.30,
    )
    return mytask


def get_cfg():

    cfg = sktopt.core.optimizers.OC_Config(
        dst_path="./result/tutorial_box_oc",
        export_img=True,
        p=sktopt.tools.SchedulerConfig.constant(
            target_value=3.0
        ),
        vol_frac=sktopt.tools.SchedulerConfig.constant(
            target_value=0.4
        ),
        max_iters=10,
        record_times=10,
        # filter_type="spacial"
        filter_type="helmholtz"
    )
    return cfg


def main():

    tsk = get_task()
    cfg = get_cfg()
    optimizer = sktopt.core.OC_Optimizer(cfg, tsk)

    optimizer.parameterize()
    optimizer.optimize()


if __name__ == '__main__':
    main()
