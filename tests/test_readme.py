import scitopt
# import pytest


def test_define_task():

    import skfem
    x_len, y_len, z_len = 1.0, 1.0, 1.0
    element_size = 0.1
    e = skfem.ElementVector(skfem.ElementHex1())
    mesh = scitopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, element_size
    )
    basis = skfem.Basis(mesh, e, intorder=3)

    # Specify Dirichlet Boundary Conditions
    dirichlet_points = scitopt.mesh.utils.get_point_indices_in_range(
        basis, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_nodes = basis.get_dofs(nodes=dirichlet_points).all()

    # Specify Force Vector
    F_points = scitopt.mesh.utils.get_point_indices_in_range(
        basis,
        (x_len, x_len),
        (y_len*2/5, y_len*3/5),
        (z_len*2/5, z_len*3/5)
    )
    F_nodes = basis.get_dofs(nodes=F_points).nodal["u^1"]
    F = 100

    # Specify Design Field
    design_elements = scitopt.mesh.utils.get_elements_in_box(
        mesh,
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )
    _ = scitopt.mesh.task.TaskConfig.from_defaults(
        210e9,
        0.30,
        basis,
        dirichlet_points,
        dirichlet_nodes,
        F_points,
        F_nodes,
        F,
        design_elements
    )


def oc_ramp_optimize(tsk):
    cfg = scitopt.core.optimizers.OC_Config()
    cfg.max_iters = 1
    cfg.record_times = 1
    optimizer = scitopt.core.OC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()


def ldmoc_optimize(tsk):
    cfg = scitopt.core.optimizers.LDMOC_Config()
    cfg.max_iters = 1
    cfg.record_times = 1
    optimizer = scitopt.core.LDMOC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()


def eumoc_optimize(tsk):
    cfg = scitopt.core.optimizers.EUMOC_Config(
        dst_path="./result/pytests"
    )
    cfg.max_iters = 1
    cfg.record_times = 1

    optimizer = scitopt.core.EUMOC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()

    cfg2 = scitopt.core.optimizers.EUMOC_Config(
        dst_path="./result/pytests",
        restart=True,
        restart_from=1
    )
    cfg2.max_iters = 1
    cfg2.record_times = 1
    optimizer2 = scitopt.core.EUMOC_Optimizer(cfg2, tsk)
    optimizer2.parameterize()
    optimizer2.optimize()


def test_optimizers():
    tsk = scitopt.mesh.toy_problem.toy_test()
    oc_ramp_optimize(tsk)
    ldmoc_optimize(tsk)
    eumoc_optimize(tsk)

    test_define_task()


if __name__ == "__main__":
    tsk = scitopt.mesh.toy_problem.toy_test()
    eumoc_optimize(tsk)
