import sktopt
# import pytest


def define_task() -> sktopt.mesh.task.TaskConfig:

    import skfem
    x_len, y_len, z_len = 1.0, 1.0, 1.0
    element_size = 0.1
    e = skfem.ElementVector(skfem.ElementHex1())
    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, element_size
    )
    basis = skfem.Basis(mesh, e, intorder=3)

    # Specify Dirichlet Boundary Conditions
    dirichlet_nodes = sktopt.mesh.utils.get_nodes_indices_in_range(
        basis.mesh, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_dofs = basis.get_dofs(nodes=dirichlet_nodes).all()

    # Specify Force Vector
    F_points = sktopt.mesh.utils.get_nodes_indices_in_range(
        basis.mesh,
        (x_len, x_len),
        (y_len*2/5, y_len*3/5),
        (z_len*2/5, z_len*3/5)
    )
    F_nodes = basis.get_dofs(nodes=F_points).nodal["u^1"]
    F = 100

    # Specify Design Field
    design_elements = sktopt.mesh.utils.get_elements_in_box(
        mesh,
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )
    return sktopt.mesh.task.TaskConfig.from_defaults(
        210e9,
        0.30,
        basis,
        dirichlet_nodes,
        dirichlet_dofs,
        F_points,
        F_nodes,
        F,
        design_elements
    )


def oc_ramp_optimize(tsk):
    cfg = sktopt.core.optimizers.OC_Config()
    cfg.max_iters = 1
    cfg.record_times = 1
    optimizer = sktopt.core.OC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()


def logmoc_optimize(tsk):
    cfg = sktopt.core.optimizers.LogMOC_Config()
    cfg.max_iters = 1
    cfg.record_times = 1
    optimizer = sktopt.core.LogMOC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()


# def linearmoc_optimize(tsk):
#     cfg = sktopt.core.optimizers.LinearMOC_Config()
#     cfg.max_iters = 1
#     cfg.record_times = 1
#     optimizer = sktopt.core.LinearMOC_Optimizer(cfg, tsk)
#     optimizer.parameterize()
#     optimizer.optimize()


def loglagrangian_optimize(tsk):
    cfg = sktopt.core.optimizers.LogLagrangian_Config(
        dst_path="./result/pytests"
    )
    cfg.max_iters = 1
    cfg.record_times = 1

    optimizer = sktopt.core.LogLagrangian_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()

    cfg2 = sktopt.core.optimizers.LogLagrangian_Config(
        dst_path="./result/pytests",
        restart=True,
        restart_from=1
    )
    cfg2.max_iters = 1
    cfg2.record_times = 1
    optimizer2 = sktopt.core.LogLagrangian_Optimizer(cfg2, tsk)
    optimizer2.parameterize()
    optimizer2.optimize()


def test_optimizers():
    tsk1 = sktopt.mesh.toy_problem.toy_test()
    oc_ramp_optimize(tsk1)
    logmoc_optimize(tsk1)
    loglagrangian_optimize(tsk1)


if __name__ == "__main__":

    tsk = define_task()
    loglagrangian_optimize(tsk)
