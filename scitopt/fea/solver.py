from typing import Callable
import scipy
import skfem
from scipy.sparse.linalg import cg

from scitopt.fea import composer


def compute_compliance_simp_basis(
    basis, free_nodes, dirichlet_nodes, force,
    E0, Emin, p, nu0,
    rho,
) -> tuple:
    K = composer.assemble_stiffness_matrix(
        basis, rho, E0,
        Emin, p, nu0
    )
    K_e, F_e = skfem.enforce(K, force, D=dirichlet_nodes)
    # u = scipy.sparse.linalg.spsolve(K_e, F_e)
    u = skfem.solve(K_e, F_e)
    f_free = force[free_nodes]
    compliance = f_free @ u[free_nodes]
    return (compliance, u)


def compute_compliance_basis_numba(
    basis, free_nodes, dirichlet_nodes, force,
    E0, Emin, p, nu0,
    rho,
    elem_func: Callable=composer.ramp_interpolation_numba
) -> tuple:
    K = composer.assemble_stiffness_matrix_numba(
        basis, rho, E0,
        Emin, p, nu0, elem_func
    )
    K_e, F_e = skfem.enforce(K, force, D=dirichlet_nodes)
    # u = scipy.sparse.linalg.spsolve(K_e, F_e)
    # u = skfem.solve(K_e, F_e)
    u, info = skfem.solve(K_e, F_e, solver=cg)
    
    print("compute_compliance_simp_basis_numba-info", info)
    f_free = force[free_nodes]
    compliance = f_free @ u[free_nodes]
    return (compliance, u)


def compute_compliance_simp(
    prb,
    rho,
    p
) -> tuple:
    return compute_compliance_simp_basis(
        prb.basis, prb.free_nodes, prb.dirichlet_nodes, prb.force,
        prb.E0, prb.Emin, p, prb.nu0, rho
    )
