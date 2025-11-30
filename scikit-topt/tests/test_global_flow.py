import os
import tempfile

import sktopt
# import pytest


def define_task_elasticity() -> sktopt.mesh.LinearElasticity:

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
    return sktopt.mesh.LinearElasticity.from_defaults(
        basis,
        dirichlet_nodes,
        dirichlet_dofs,
        F_points,
        F_nodes,
        F,
        design_elements,
        210e3,
        0.30,
    )


def define_task_heatconduction() -> sktopt.mesh.LinearHeatConduction:
    import numpy as np
    import skfem
    x_len = 8.0
    y_len = 8.0
    z_len = 1.0
    # z_len = 8.0
    mesh_size = 0.2

    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, mesh_size
    )
    robin_in_range_0 = sktopt.mesh.utils.get_points_in_range(
        (0.0, 0.0), (0.0, y_len), (0.0, z_len)
    )
    robin_in_range_1 = sktopt.mesh.utils.get_points_in_range(
        (0.0, x_len), (y_len, y_len), (0.0, z_len)
    )

    robin_coefficient = 4.0e-5
    robin_bc_value = 300.0

    eps = mesh_size
    dirichlet_in_range = sktopt.mesh.utils.get_points_in_range(
        (x_len-1.0*x_len/20, x_len), (0.0, 1.0*y_len/20), (0.0, z_len)
    )
    dirichlet_value = 600.0
    boundaries = {
        "robin_0": robin_in_range_0,
        "robin_1": robin_in_range_1,
        "dirichlet_0": dirichlet_in_range
    }

    mesh = mesh.with_boundaries(boundaries)
    subdomains = {"design": np.array(range(mesh.nelements))}
    mesh = mesh.with_subdomains(subdomains)

    e = skfem.ElementHex1()
    basis = skfem.Basis(mesh, e, intorder=1)
    k = 10.0
    objective = "compliance"
    design_robin_boundary = True
    mytask = sktopt.mesh.LinearHeatConduction.from_mesh_tags(
        basis,
        dirichlet_value,
        robin_coefficient,
        robin_bc_value,
        design_robin_boundary,
        k, objective
    )
    return mytask


def oc_optimize(tsk, dst_path: str) -> sktopt.core.OC_Optimizer:
    cfg = sktopt.core.optimizers.OC_Config()
    cfg.max_iters = 1
    cfg.record_times = cfg.max_iters
    cfg.dst_path = dst_path
    optimizer = sktopt.core.OC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()
    return optimizer


def logmoc_optimize(tsk, dst_path: str) -> sktopt.core.LogMOC_Optimizer:
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
    cfg.record_times = cfg.max_iters
    cfg.dst_path = dst_path
    optimizer = sktopt.core.LogMOC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()
    return optimizer


def test_optimizers():
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        tsk1 = sktopt.mesh.toy_problem.toy_test()
        oc_optimizer = oc_optimize(tsk1, os.path.join(tmpdir, "oc_elasticity"))
        result1 = oc_optimizer.recorder.as_object_latest()
        assert oc_optimizer is not None, "OC optimizer returned None"
        assert np.isfinite(result1.compliance), "OC compliance must be finite"

        tsk2 = sktopt.mesh.toy_problem.toy2()
        moc_optimizer = logmoc_optimize(tsk2, os.path.join(tmpdir, "logmoc"))
        result2 = moc_optimizer.recorder.as_object_latest()
        assert moc_optimizer is not None, "LogMOC optimizer returned None"
        assert np.isfinite(result2.compliance), "LogMOC compliance must be finite"
        
        tsk3 = define_task_heatconduction()
        oc_optimizer_hc = oc_optimize(tsk3, os.path.join(tmpdir, "oc_heat"))
        result3 = oc_optimizer_hc.recorder.as_object_latest()
        assert oc_optimizer_hc is not None, "OC optimizer returned None"
        assert np.isfinite(result3.compliance), "OC compliance must be finite"


if __name__ == "__main__":

    # tsk = define_task()
    # logmoc_optimize(tsk)
    test_optimizers()
