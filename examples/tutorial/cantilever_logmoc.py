import numpy as np
import skfem

import sktopt
from sktopt.mesh import toy_problem


def get_task():
    x_len = 8.0
    y_len = 2.0
    z_len = 2.0
    mesh_size = 0.12
    eps = mesh_size

    mesh = toy_problem.create_box_hex(
        x_len, y_len, z_len, mesh_size
    )

    def box_selector(x_rng, y_rng, z_rng):
        def in_range(x):
            return (
                (x[0] >= x_rng[0]) & (x[0] <= x_rng[1]) &
                (x[1] >= y_rng[0]) & (x[1] <= y_rng[1]) &
                (x[2] >= z_rng[0]) & (x[2] <= z_rng[1])
            )
        return in_range

    dirichlet_in_range = box_selector(
        (0.0, eps),
        (0.0, y_len),
        (0.0, z_len)
    )
    dirichlet_facets = mesh.facets_satisfying(dirichlet_in_range)
    dirichlet_dir = "all"

    neumann_in_range = box_selector(
        (x_len - eps, x_len),
        (y_len * 0.3, y_len * 0.7),
        (z_len * 0.3, z_len * 0.7),
    )
    neumann_facets = mesh.facets_satisfying(neumann_in_range)
    neumann_dir_type = "u^2"
    neumann_value = -5.0

    boundaries = {
        "dirichlet": dirichlet_facets,
        "neumann": neumann_facets,
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
        dst_path="./result/tutorial_cantilever_logmoc",
        export_img=True,
        filter_type="spacial",
        vol_frac=sktopt.tools.SchedulerConfig.step_accelerating(
            init_value=0.8,
            target_value=0.40,
            num_steps=30,
            curvature=2.0
        ),
        p=sktopt.tools.SchedulerConfig.step(
            init_value=1.0,
            target_value=3.0,
            num_steps=30
        ),
        beta=sktopt.tools.SchedulerConfig.step_accelerating(
            init_value=0.1,
            target_value=8.0,
            num_steps=30.0,
            curvature=8.0
        ),
        filter_radius=sktopt.tools.SchedulerConfig.step(
            init_value=1.4,
            target_value=0.40,
            num_steps=20
        ),
        move_limit=sktopt.tools.SchedulerConfig.step(
            init_value=0.015,
            target_value=0.03,
            num_steps=25
        ),
        eta=sktopt.tools.SchedulerConfig.constant(
            target_value=0.5
        ),
        neumann_scale=sktopt.tools.SchedulerConfig.step_to_one(
            num_steps=50
        ),
        solver_option="petsc_spdirect",
        mu_p=0.5,
        lambda_v=1e-2,
        lambda_decay=0.99,
        lambda_lower=-1e7,
        lambda_upper=1e7,
        max_iters=600,
        record_times=50,
        check_convergence=False,
        tol_rho_change=1e-3,
        tol_kkt_residual=1e-5,
    )


def main():
    tsk = get_task()
    cfg = get_cfg()
    optimizer = sktopt.core.LogMOC_Optimizer(cfg, tsk)
    optimizer.parameterize()
    optimizer.optimize()


if __name__ == "__main__":
    main()
