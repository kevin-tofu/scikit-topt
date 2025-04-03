from typing import Callable
import scipy
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spilu, LinearOperator
import skfem
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
    elem_func: Callable=composer.ramp_interpolation_numba,
    rtol=1e-6,
    maxiter=1000
) -> tuple:
    K = composer.assemble_stiffness_matrix_numba(
        basis, rho, E0,
        Emin, p, nu0, elem_func
    )
    K_e, F_e = skfem.enforce(K, force, D=dirichlet_nodes)
    
    # if True:
    #     M_diag = K.diagonal()
    #     M_inv = 1.0 / M_diag
    #     M = scipy.linalg.LinearOperator(K.shape, matvec=lambda x: M_inv * x)

    # else:
    #     ilu = spilu(K.tocsc())
    #     M = LinearOperator(K.shape, matvec=ilu.solve)

    u, info = skfem.solve(K_e, F_e, solver=cg)
    # u, info = cg(K_e, F_e, M, rtol=rtol, maxiter=maxiter)
    
    # u = scipy.sparse.linalg.spsolve(K_e, F_e)
    # u = skfem.solve(K_e, F_e)
    
    
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


if __name__ == '__main__':
    pass