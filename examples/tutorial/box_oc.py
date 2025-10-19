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
    force_in_range_0 = sktopt.mesh.utils.get_points_in_range(
        (x_len, x_len), (y_len-eps, y_len), (0, z_len)
    )
    force_in_range_1 = sktopt.mesh.utils.get_points_in_range(
        (x_len, x_len), (0, eps), (0, z_len)
    )
    force_dir_type = ["u^2", "u^2"]
    force_value = [-100, 100]

    boundaries = {
        "dirichlet": dirichlet_in_range,
        "force_0": force_in_range_0,
        "force_1": force_in_range_1
    }
    mesh = mesh.with_boundaries(boundaries)
    subdomains = {"design": np.array(range(mesh.nelements))}
    mesh = mesh.with_subdomains(subdomains)

    e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=2)
    E0 = 210e9
    mytask = sktopt.mesh.task.TaskConfig.from_mesh_tags(
        E0,
        0.30,
        basis,
        dirichlet_dir,
        force_dir_type,
        force_value
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
        max_iters=40,
        record_times=40
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
