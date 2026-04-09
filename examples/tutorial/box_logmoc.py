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
    dirichlet_in_range = sktopt.mesh.utils.get_points_in_range(
        (0.0, 0.05), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_dir = "all"

    eps = mesh_size
    neumann_in_range_0 = sktopt.mesh.utils.get_points_in_range(
        (x_len, x_len), (y_len - eps, y_len), (0, z_len)
    )
    neumann_in_range_1 = sktopt.mesh.utils.get_points_in_range(
        (x_len, x_len), (0, eps), (0, z_len)
    )
    neumann_dir_type = ["u^2", "u^2"]
    neumann_value = [-1.0, 1.0]

    boundaries = {
        "dirichlet": dirichlet_in_range,
        "neumann_0": neumann_in_range_0,
        "neumann_1": neumann_in_range_1,
    }
    mesh = mesh.with_boundaries(boundaries)
    mesh = mesh.with_subdomains({"design": np.arange(mesh.nelements)})

    e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=2)
    return sktopt.mesh.LinearElasticity.from_mesh_tags(
        basis,
        dirichlet_dir,
        neumann_dir_type,
        neumann_value,
        210e3,
        0.30,
    )


def get_cfg():
    return sktopt.core.LogMOC_Config(
        dst_path="./result/tutorial_box_logmoc",
        export_img=True,
        p=sktopt.tools.SchedulerConfig.constant(target_value=3.0),
        vol_frac=sktopt.tools.SchedulerConfig.constant(target_value=0.6),
        beta=sktopt.tools.SchedulerConfig.step_accelerating(
            init_value=0.6,
            target_value=4.0,
            num_steps=4,
            curvature=2.0,
        ),
        filter_radius=sktopt.tools.SchedulerConfig.constant(target_value=0.6),
        move_limit=sktopt.tools.SchedulerConfig.constant(target_value=0.02),
        eta=sktopt.tools.SchedulerConfig.constant(target_value=0.5),
        lambda_lower=-1e7,
        lambda_upper=1e7,
        mu_p=5.0,
        lambda_v=0.1,
        lambda_decay=0.97,
        filter_type="spacial",
        max_iters=100,
        record_times=20,
        check_convergence=False,
    )


def main():
    tsk = get_task()
    cfg = get_cfg()
    optimizer = sktopt.core.LogMOC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()


if __name__ == '__main__':
    main()
