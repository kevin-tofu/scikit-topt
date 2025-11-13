import numpy as np
import skfem
import sktopt


def get_task_0(
    multiple_robin: bool = True,
    design_robin_boundary: bool = True
):

    x_len = 8.0
    y_len = 8.0
    z_len = 1.0
    # z_len = 8.0
    mesh_size = 0.2

    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, mesh_size
    )

    if multiple_robin:
        # robin_in_range_0 = sktopt.mesh.utils.get_points_in_range(
        #     (0.0, 0.0), (0.0, y_len/9), (0.0, z_len)
        # )
        # robin_in_range_1 = sktopt.mesh.utils.get_points_in_range(
        #     (0.0, 0.0), (4*y_len/9, 5*y_len/9), (0.0, z_len)
        # )
        # robin_in_range_2 = sktopt.mesh.utils.get_points_in_range(
        #     (0.0, 0.0), (8*y_len/9, 9*y_len/9), (0.0, z_len)
        # )
        robin_in_range_0 = sktopt.mesh.utils.get_points_in_range(
            (0.0, 0.0), (0.0, y_len), (0.0, z_len)
        )
        robin_in_range_1 = sktopt.mesh.utils.get_points_in_range(
            (0.0, x_len), (y_len, y_len), (0.0, z_len)
        )

        robin_coefficient = 4.0e-5
        robin_bc_value = 300.0
    else:
        robin_in_range = sktopt.mesh.utils.get_points_in_range(
            (0.0, 0.0), (0.0, y_len), (0.0, z_len)
        )
        robin_coefficient = 4.0e-3
        robin_bc_value = 300.0

    eps = mesh_size
    dirichlet_in_range = sktopt.mesh.utils.get_points_in_range(
        (x_len-1.0*x_len/20, x_len), (0.0, 1.0*y_len/20), (0.0, z_len)
    )
    # dirichlet_in_range = sktopt.mesh.utils.get_points_in_range(
    #     (x_len, x_len), (4*y_len/9, 5*y_len/9), (4*z_len/8, 5*z_len/9)
    # )
    dirichlet_value = 600.0

    # boundaries = {
    #     "robin": robin_in_range,
    #     "dirichlet_0": dirichlet_in_range
    # }
    if multiple_robin:
        boundaries = {
            "robin_0": robin_in_range_0,
            "robin_1": robin_in_range_1,
            # "robin_2": robin_in_range_2,
            "dirichlet_0": dirichlet_in_range
        }
    else:
        boundaries = {
            "robin": robin_in_range,
            "dirichlet_0": dirichlet_in_range
        }

    mesh = mesh.with_boundaries(boundaries)
    subdomains = {"design": np.array(range(mesh.nelements))}
    mesh = mesh.with_subdomains(subdomains)

    e = skfem.ElementHex1()
    basis = skfem.Basis(mesh, e, intorder=1)
    k = 10.0
    # objective = "averaged_temp"
    objective = "compliance"
    mytask = sktopt.mesh.LinearHeatConduction.from_mesh_tags(
        basis,
        dirichlet_value,
        robin_coefficient,
        robin_bc_value,
        design_robin_boundary,
        k, objective
    )
    return mytask


def get_task_1():

    x_len = 8.0
    y_len = 8.0
    z_len = 1.0
    mesh_size = 0.2

    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, mesh_size
    )
    eps = mesh_size
    dirichlet_in_range_0 = sktopt.mesh.utils.get_points_in_range(
        (x_len, x_len), (y_len-eps, y_len), (0, z_len)
    )
    dirichlet_in_range_1 = sktopt.mesh.utils.get_points_in_range(
        (x_len, x_len), (0, eps), (0, z_len)
    )
    dirichlet_value = [600.0, 600.0]

    boundaries = {
        "dirichlet_0": dirichlet_in_range_0,
        "dirichlet_1": dirichlet_in_range_1
    }
    mesh = mesh.with_boundaries(boundaries)
    subdomains = {"design": np.array(range(mesh.nelements))}
    mesh = mesh.with_subdomains(subdomains)

    e = skfem.ElementHex1()
    basis = skfem.Basis(mesh, e, intorder=1)
    k = 10.0
    mytask = sktopt.mesh.LinearHeatConduction.from_mesh_tags(
        basis,
        dirichlet_value,
        None,
        None,
        k,
    )
    return mytask


def get_cfg():

    cfg = sktopt.core.optimizers.OC_Config(
        dst_path="./result/tutorial_box_oc_heat",
        export_img=True,
        p=sktopt.tools.SchedulerConfig.constant(
            target_value=3.0
        ),
        vol_frac=sktopt.tools.SchedulerConfig.constant(
            target_value=0.7
        ),
        max_iters=40,
        record_times=40,
        # filter_type="spacial"
        filter_type="helmholtz"
    )
    return cfg


def main():

    tsk = get_task_0(multiple_robin=True, design_robin_boundary=True)
    # tsk = get_task_0(multiple_robin=True, design_robin_boundary=False)
    # tsk = get_task_0(multiple_robin=False, design_robin_boundary=False)
    # tsk = get_task_1()
    cfg = get_cfg()
    optimizer = sktopt.core.OC_Optimizer(cfg, tsk)

    optimizer.parameterize()
    optimizer.optimize()


if __name__ == '__main__':
    main()
