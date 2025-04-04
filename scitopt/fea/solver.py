from typing import Callable, Literal
import numpy as np
import scipy
from scipy.sparse.linalg import cg, spilu, LinearOperator
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
    elem_func: Callable = composer.ramp_interpolation_numba,
    rtol: float = 1e-6,
    solver: Literal['auto', 'cg', 'spsolve', 'pyamg'] = 'auto',
    maxiter: int = None,
) -> tuple:
    K = composer.assemble_stiffness_matrix_numba(
        basis, rho, E0, Emin, p, nu0, elem_func
    )
    K_e, F_e = skfem.enforce(K, force, D=dirichlet_nodes)
    n_dof = K.shape[0]

    # Solver auto-selection
    if solver == 'auto':
        if n_dof < 5000:
            chosen_solver = 'spsolve'
        elif n_dof < 30000:
            chosen_solver = 'cg'
        else:
            # chosen_solver = 'pyamg'
            chosen_solver = 'cg'
            
    else:
        chosen_solver = solver

    _maxiter = min(1000, max(300, n_dof // 5)) if maxiter is None else maxiter

    try:
        if chosen_solver == 'cg':
            M_diag = K.diagonal()
            M_inv = 1.0 / M_diag
            M = LinearOperator(K.shape, matvec=lambda x: M_inv * x)
            u, info = cg(A=K_e, b=F_e, M=M, rtol=rtol, maxiter=_maxiter)
            print("CG (diag preconditioner) solver info:", info)

        elif chosen_solver == 'pyamg':
            import pyamg
            ml = pyamg.ruge_stuben_solver(K)
            u = ml.solve(F_e, tol=1e-8)
            info = 0

        elif chosen_solver == 'spsolve':
            u = scipy.sparse.linalg.spsolve(K_e, F_e)
            info = 0
            print("Direct solver used: spsolve")

        # elif chosen_solver == 'cg_spilu':
        #     ilu = spilu(K.tocsc())
        #     M = LinearOperator(K.shape, matvec=ilu.solve)
        #     u, info = cg(A=K_e, b=F_e, M=M, rtol=rtol, maxiter=maxiter)
        #     print("CG (spilu preconditioner) solver info:", info)

        else:
            raise ValueError(f"Unknown solver: {chosen_solver}")

    except Exception as e:
        print(f"Solver exception - {e}, falling back to spsolve.")
        u = scipy.sparse.linalg.spsolve(K_e, F_e)

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
    
    from scitopt.mesh import toy_problem
    tsk = toy_problem.toy()
    
    rho = np.ones(tsk.all_elements.shape)
    p = 1.0
    compliacne, u = compute_compliance_basis_numba(
        tsk.basis, tsk.free_nodes, tsk.dirichlet_nodes, tsk.force,
        tsk.E0, tsk.Emin, p, tsk.nu0,
        rho,
    )
    print(f"compliacne: {compliacne}")