from typing import Callable, Literal
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
import skfem
import pyamg
from scitopt.fea import composer
from scitopt.tools.logconf import mylogger
logger = mylogger(__name__)


def compute_compliance_simp_basis(
    basis, free_dofs, dirichlet_dofs, force,
    E0, Emin, p, nu0,
    rho,
) -> tuple:
    K = composer.assemble_stiffness_matrix(
        basis, rho, E0,
        Emin, p, nu0
    )
    K_e, F_e = skfem.enforce(K, force, D=dirichlet_dofs)
    # u = scipy.sparse.linalg.spsolve(K_e, F_e)
    u = skfem.solve(K_e, F_e)
    f_free = force[free_dofs]
    compliance = f_free @ u[free_dofs]
    return (compliance, u)


def solve_u(
    K_cond: scipy.sparse.csc_matrix,
    F_cond: np.ndarray,
    chosen_solver: Literal['cg', 'spsolve', 'pyamg'] = 'auto',
    rtol: float = 1e-8,
    maxiter: int = None,
) -> np.ndarray:
    try:
        if chosen_solver == 'cg':
            M_diag = K_cond.diagonal()
            M_inv = 1.0 / M_diag
            M = LinearOperator(K_cond.shape, matvec=lambda x: M_inv * x)

            u_c, info = scipy.sparse.linalg.cg(
                A=K_cond, b=F_cond, M=M, rtol=rtol, maxiter=maxiter
            )
            logger.info(f"CG (diag preconditioner) solver info: {info}")

        elif chosen_solver == 'pyamg':
            ml = pyamg.smoothed_aggregation_solver(K_cond)
            M = ml.aspreconditioner()

            # u_c, info = scipy.sparse.linalg.cg(
            #     A=K_cond, b=F_cond, M=M, tol=rtol, maxiter=maxiter
            # )
            u_c, info = scipy.sparse.linalg.cg(
                A=K_cond, b=F_cond, M=M, rtol=rtol, maxiter=maxiter
            )
            logger.info(f"CG (AMG preconditioner) solver info: {info}")

        elif chosen_solver == 'spsolve':
            u_c = scipy.sparse.linalg.spsolve(K_cond, F_cond)
            info = 0
            logger.info("Direct solver used: spsolve")

        else:
            raise ValueError(f"Unknown solver: {chosen_solver}")

    except Exception as e:
        print(f"Solver exception - {e}, falling back to spsolve.")
        u_c = scipy.sparse.linalg.spsolve(K_cond, F_cond)

    return u_c


def compute_compliance_basis(
    basis, free_dofs, dirichlet_dofs, force,
    E0, Emin, p, nu0,
    rho,
    elem_func: Callable = composer.simp_interpolation,
    solver: Literal['auto', 'cg', 'spsolve', 'pyamg'] = 'auto',
    rtol: float = 1e-5,
    maxiter: int = None,
) -> tuple:
    K = composer.assemble_stiffness_matrix(
        basis, rho, E0, Emin, p, nu0, elem_func
    )
    n_dof = K.shape[0]
    # Solver auto-selection
    if solver == 'auto':
        if n_dof < 1000:
            chosen_solver = 'spsolve'
        elif n_dof < 30000:
            # chosen_solver = 'cg'
            chosen_solver = 'pyamg'
        else:
            chosen_solver = 'pyamg'
            # chosen_solver = 'cg'
    else:
        chosen_solver = solver

    _maxiter = min(1000, max(300, n_dof // 5)) if maxiter is None else maxiter
    K_csr = K.tocsr()
    # condense
    # K_c, F_c, U_c, I = skfem.condense(K_csr, force, D=dirichlet_dofs)
    # U_c[I] = solve_u(
    #     K_c, F_c, chosen_solver=chosen_solver,
    #     rtol=rtol, maxiter=_maxiter
    # )
    # U_c[dirichlet_dofs] = 0.0
    # u = U_c
    # enforce
    # K_e, F_e = skfem.enforce(K_csr, force, D=dirichlet_dofs)
    # u = solve_u(
    #     K_e, F_e , chosen_solver=chosen_solver,
    #     rtol=rtol, maxiter=_maxiter
    # )
    all_dofs = np.arange(K_csr.shape[0])
    free_dofs = np.setdiff1d(all_dofs, dirichlet_dofs, assume_unique=True)

    K_c = K_csr[free_dofs, :][:, free_dofs]
    F_c = force[free_dofs]
    u_free = solve_u(
        K_c, F_c, chosen_solver=chosen_solver, rtol=rtol, maxiter=_maxiter
    )
    u = np.zeros_like(force)
    u[free_dofs] = u_free
    f_free = force[free_dofs]
    compliance = f_free @ u[free_dofs]
    return (float(compliance), u)


def compute_compliance_basis_numba(
    basis, free_dofs, dirichlet_dofs, force,
    E0, Emin, p, nu0,
    rho,
    elem_func: Callable = composer.simp_interpolation_numba,
    solver: Literal['auto', 'cg', 'spsolve', 'pyamg'] = 'auto',
    rtol: float = 1e-8,
    maxiter: int = None,
) -> tuple:
    K = composer.assemble_stiffness_matrix_numba(
        basis, rho, E0, Emin, p, nu0, elem_func
    )
    K_e, F_e = skfem.enforce(K, force, D=dirichlet_dofs)
    n_dof = K.shape[0]

    # Solver auto-selection
    if solver == 'auto':
        if n_dof < 5000:
            chosen_solver = 'spsolve'
        elif n_dof < 30000:
            chosen_solver = 'cg'
        else:
            chosen_solver = 'pyamg'
            # chosen_solver = 'cg'
    else:
        chosen_solver = solver

    _maxiter = min(1000, max(300, n_dof // 5)) if maxiter is None else maxiter
    u = solve_u(
        K_e, F_e, chosen_solver=chosen_solver,
        rtol=rtol, maxiter=_maxiter
    )

    f_free = force[free_dofs]
    compliance = f_free @ u[free_dofs]
    return (compliance, u)


def compute_compliance_batch(
    basis, free_dofs, dirichlet_dofs, force_batch,
    E0, Emin, p, nu0,
    rho,
    elem_func: Callable = composer.ramp_interpolation_numba,
    solver: Literal['spsolve'] = 'spsolve',
    rtol: float = 1e-6,
    maxiter: int = None,
) -> tuple:
    """
    Parameters:
        force_batch: np.ndarray of shape (n_dof, n_load_cases)
    Returns:
        compliance_all: np.ndarray of shape (n_load_cases,)
        u_all: np.ndarray of shape (n_dof, n_load_cases)
    """
    K = composer.assemble_stiffness_matrix_numba(
        basis, rho, E0, Emin, p, nu0, elem_func
    )

    n_cases = force_batch.shape[1]
    n_dof = K.shape[0]
    # Apply boundary condition once (assuming same dirichlet_dofs for all)
    K_e, _ = skfem.enforce(K, np.zeros(n_dof), D=dirichlet_dofs)
    # Apply boundary condition to all force vectors
    F_batch = np.zeros((n_dof, n_cases))
    for i in range(n_cases):
        _, F_batch[:, i] = skfem.enforce(
            K, force_batch[:, i], D=dirichlet_dofs
        )

    if solver != "spsolve":
        raise NotImplementedError(
            "Currently only 'spsolve' is supported for batch."
        )

    # Solve Ku = f for all cases
    # shape: (n_dof, n_cases)
    u_all = scipy.sparse.linalg.spsolve(K_e, F_batch)

    f_free = force_batch[free_dofs, :]
    u_free = u_all[free_dofs, :]
    compliance_all = np.sum(f_free * u_free, axis=0)  # f^T u for each case

    return compliance_all, u_all


if __name__ == '__main__':
    from scitopt.mesh import toy_problem
    tsk = toy_problem.toy_msh("plate-0.2.msh")
    rho = np.ones(tsk.all_elements.shape)
    p = 1.0
    compliacne, u = compute_compliance_basis_numba(
        tsk.basis, tsk.free_dofs, tsk.dirichlet_dofs, tsk.force,
        tsk.E0, tsk.Emin, p, tsk.nu0,
        rho,
    )
    print(f"compliacne: {compliacne}")
