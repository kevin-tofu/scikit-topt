from typing import Callable, Literal
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
import skfem
import pyamg
from sktopt.fea import composer
from sktopt.tools.logconf import mylogger
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
    chosen_solver: Literal['cg_jacobi', 'spsolve', 'cg_pyamg'] = 'auto',
    rtol: float = 1e-8,
    maxiter: int = None,
) -> np.ndarray:
    try:
        if chosen_solver == 'cg_jacobi':
            M_diag = K_cond.diagonal()
            M_inv = 1.0 / M_diag
            M = LinearOperator(K_cond.shape, matvec=lambda x: M_inv * x)

            u_c, info = scipy.sparse.linalg.cg(
                A=K_cond, b=F_cond, M=M, rtol=rtol, maxiter=maxiter
            )
            logger.info(f"CG (diag preconditioner) solver info: {info}")

        elif chosen_solver == 'cg_pyamg':
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
    solver: Literal['auto', 'cg_jacobi', 'spsolve', 'cg_pyamg'] = 'auto',
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
            # chosen_solver = 'cg_jacobi'
            chosen_solver = 'cg_pyamg'
        else:
            chosen_solver = 'cg_pyamg'
            # chosen_solver = 'cg_jacobi'
    else:
        chosen_solver = solver

    _maxiter = min(1000, max(300, n_dof // 5)) if maxiter is None else maxiter
    K_csr = K.tocsr()
    all_dofs = np.arange(K_csr.shape[0])
    free_dofs = np.setdiff1d(all_dofs, dirichlet_dofs, assume_unique=True)

    # enforce
    K_e, F_e = skfem.enforce(K_csr, force, D=dirichlet_dofs)
    u = solve_u(
        K_e, F_e, chosen_solver=chosen_solver,
        rtol=rtol, maxiter=_maxiter
    )

    # condense
    # K_c, F_c, U_c, I = skfem.condense(K, F, D=fixed_dofs)
    # K_c = K_csr[free_dofs, :][:, free_dofs]
    # F_c = force[free_dofs]
    # u_free = solve_u(
    #     K_c, F_c, chosen_solver=chosen_solver, rtol=rtol, maxiter=_maxiter
    # )
    # u = np.zeros_like(force)
    # u[free_dofs] = u_free
    # f_free = force[free_dofs]
    # compliance = f_free @ u[free_dofs]
    compliance = F_e[free_dofs] @ u[free_dofs]
    return (float(compliance), u)


def solve_multi_load(
    basis: skfem.CellBasis,
    free_dofs: np.ndarray,
    dirichlet_dofs: np.ndarray,
    force_list: list[np.ndarray],
    E0: float, Emin: float, p: float, nu0: float,
    rho: np.ndarray,
    u_all: np.ndarray,
    solver: Literal['auto', 'cg_jacobi', 'spsolve', 'cg_pyamg'] = 'auto',
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int = None,
    n_joblib: int = 1
) -> float:
    solver = 'spsolve' if solver == 'auto' else solver
    n_dof = basis.N
    assert u_all.shape == (n_dof, len(force_list))

    K = composer.assemble_stiffness_matrix(
        basis, rho, E0, Emin, p, nu0, elem_func
    )
    _maxiter = min(1000, max(300, n_dof // 5)) if maxiter is None else maxiter
    K_csr = K.tocsr()

    # all_dofs = np.arange(n_dof)
    # free_dofs = np.setdiff1d(all_dofs, dirichlet_dofs, assume_unique=True)
    # K_e = K_csr[free_dofs, :][:, free_dofs]
    # K_e, _ = skfem.enforce(
    #     K_csr[free_dofs, :][:, free_dofs], force_list[0],
    #     D=dirichlet_dofs
    # )
    # K_c, _, _, _ = skfem.condense(K_csr, force_list[0], D=dirichlet_dofs)
    K_e, _ = skfem.enforce(K_csr, force_list[0], D=dirichlet_dofs)
    F_stack = np.column_stack([
        skfem.enforce(K_csr, f, D=dirichlet_dofs)[1] for f in force_list
    ])
    compliance_total = 0.0
    u_all[:, :] = 0.0
    if solver == 'spsolve':
        if n_joblib > 1:
            from joblib import Parallel, delayed, parallel_backend
            lu = scipy.sparse.linalg.splu(K_e.tocsc())

            def solve_system(F_stack):
                return lu.solve(F_stack)

            with parallel_backend("threading"):
                u_all[:, :] = np.column_stack(
                    Parallel(n_jobs=n_joblib)(
                        delayed(solve_system)(F_stack[:, i]) for i in range(
                            F_stack.shape[1]
                        )
                    )
                )

        else:
            lu = scipy.sparse.linalg.splu(K_e.tocsc())
            u_all[:, :] = np.column_stack(
                [lu.solve(F_stack[:, i]) for i in range(F_stack.shape[1])]
            )

    else:
        # choose preconditioner if needed
        if solver == 'cg_jacobi':
            M_diag = K_e.diagonal()
            M_inv = 1.0 / M_diag
            M = LinearOperator(K_e.shape, matvec=lambda x: M_inv * x)
        elif solver == 'cg_pyamg':
            ml = pyamg.smoothed_aggregation_solver(K_e)
            M = ml.aspreconditioner()
        else:
            raise ValueError(f"Unknown solver: {solver}")

        for i, _ in enumerate(force_list):
            F_e = F_stack[:, i]
            # _, F_e = skfem.enforce(K_csr, force, D=dirichlet_dofs)
            u_e, info = scipy.sparse.linalg.cg(
                K_e, F_e, M=M, rtol=rtol, maxiter=_maxiter
            )
            if info != 0:
                logger.info(
                    f"[warning] \
                        CG did not converge for load case {i}: info = {info}"
                )
            u_all[:, i] = u_e
            # compliance_total += F_e[free_dofs] @ u_e[free_dofs]

    # compliance_total = np.sum(np.einsum('ij,ij->j', F_stack, u_all))
    return F_stack


def compute_compliance_basis_multi_load(
    basis: skfem.CellBasis,
    free_dofs: np.ndarray,
    dirichlet_dofs: np.ndarray,
    force_list: list[np.ndarray],
    E0: float, Emin: float, p: float, nu0: float,
    rho: np.ndarray,
    u_all: np.ndarray,
    solver: Literal['auto', 'cg_jacobi', 'spsolve', 'cg_pyamg'] = 'auto',
    elem_func: Callable = composer.simp_interpolation,
    rtol: float = 1e-5,
    maxiter: int = None,
    n_joblib: int = 1
) -> np.ndarray:
    F_stack = solve_multi_load(
        basis, free_dofs, dirichlet_dofs, force_list,
        E0, Emin, p, nu0,
        rho,
        u_all,
        solver=solver, elem_func=elem_func, rtol=rtol,
        maxiter=maxiter, n_joblib=n_joblib
    )

    # compliance_total = np.sum(np.einsum('ij,ij->j', F_stack, u_all))
    compliance_each = np.einsum('ij,ij->j', F_stack, u_all)
    return compliance_each
