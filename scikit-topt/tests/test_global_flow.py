import sktopt
# import pytest


def define_task() -> sktopt.mesh.task.LinearElastisicity:

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
    return sktopt.mesh.task.LinearElastisicity.from_defaults(
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


def oc_optimize(tsk) -> sktopt.core.OC_Optimizer:
    cfg = sktopt.core.optimizers.OC_Config()
    cfg.max_iters = 1
    cfg.record_times = 1
    optimizer = sktopt.core.OC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()
    return optimizer


def logmoc_optimize(tsk) -> sktopt.core.LogMOC_Optimizer:
    cfg = sktopt.core.optimizers.LogMOC_Config(
        p=sktopt.tools.SchedulerConfig(
            init_value=1.0, target_value=3.0,
            num_steps=3,
            scheduler_type="Step"
        ),
        vol_frac=sktopt.tools.SchedulerConfig(
            target_value=0.6, scheduler_type="Step"
        ),
    )
    cfg.max_iters = 1
    cfg.record_times = 1
    optimizer = sktopt.core.LogMOC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()
    return optimizer


def test_optimizers():
    import numpy as np

    tsk1 = sktopt.mesh.toy_problem.toy_test()
    oc_optimizer = oc_optimize(tsk1)
    result1 = oc_optimizer.recorder.as_object_latest()
    result1 = oc_optimizer.recorder.as_object()
    assert oc_optimizer is not None, "OC optimizer returned None"
    assert np.isfinite(result1.compliance), "OC compliance must be finite"

    tsk2 = sktopt.mesh.toy_problem.toy2()
    moc_optimizer = logmoc_optimize(tsk2)
    result2 = moc_optimizer.recorder.as_object()
    assert moc_optimizer is not None, "LogMOC optimizer returned None"
    assert np.isfinite(result2.compliance), "LogMOC compliance must be finite"


if __name__ == "__main__":

    # tsk = define_task()
    # logmoc_optimize(tsk)
    test_optimizers()
