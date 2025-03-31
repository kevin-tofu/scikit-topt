import scipy
import skfem
from scitopt.fea import composer



def computer_compliance_simp_basis(
    basis, free_nodes, dirichlet_nodes, force,
    E0, Emin, p, nu0,
    rho,
) -> tuple:
    K = composer.assemble_stiffness_matrix(
        basis, rho, E0,
        Emin, p, nu0
    )
    K_e, F_e = skfem.enforce(K, force, D=dirichlet_nodes)
    u = scipy.sparse.linalg.spsolve(K_e, F_e)
    f_free = force[free_nodes]
    compliance = f_free @ u[free_nodes]
    return (compliance, u)


def computer_compliance_simp(
    prb,
    rho,
    p
) -> tuple:
    return computer_compliance_simp_basis(
        prb.basis, prb.free_nodes, prb.dirichlet_nodes, prb.force,
        prb.E0, prb.Emin, p, prb.nu0, rho
    )
