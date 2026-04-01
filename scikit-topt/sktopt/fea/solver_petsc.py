from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy

PETScSolverMode = str


@dataclass(frozen=True)
class PETScOptions:
    ksp_type: str = "cg"
    pc_type: str = "gamg"
    pc_factor_mat_solver_type: str | None = None
    options_prefix: str | None = None
    use_set_from_options: bool = True


def direct_petsc_options(
    factor_solver_type: str | None = None,
    options_prefix: str | None = None,
) -> PETScOptions:
    return PETScOptions(
        ksp_type="preonly",
        pc_type="lu",
        pc_factor_mat_solver_type=factor_solver_type,
        options_prefix=options_prefix,
    )


def merge_petsc_options(
    base: PETScOptions,
    override: PETScOptions | dict[str, object] | None = None,
) -> PETScOptions:
    if override is None:
        return base
    if isinstance(override, dict):
        return PETScOptions(
            ksp_type=str(override.get("ksp_type", base.ksp_type)),
            pc_type=str(override.get("pc_type", base.pc_type)),
            pc_factor_mat_solver_type=(
                base.pc_factor_mat_solver_type
                if override.get("pc_factor_mat_solver_type") is None
                else str(override["pc_factor_mat_solver_type"])
            ),
            options_prefix=(
                base.options_prefix
                if override.get("options_prefix") is None
                else str(override["options_prefix"])
            ),
            use_set_from_options=bool(
                override.get("use_set_from_options", base.use_set_from_options)
            ),
        )
    override_opts = normalize_petsc_options(override)
    return PETScOptions(
        ksp_type=override_opts.ksp_type,
        pc_type=override_opts.pc_type,
        pc_factor_mat_solver_type=(
            override_opts.pc_factor_mat_solver_type
            if override_opts.pc_factor_mat_solver_type is not None
            else base.pc_factor_mat_solver_type
        ),
        options_prefix=(
            override_opts.options_prefix
            if override_opts.options_prefix is not None
            else base.options_prefix
        ),
        use_set_from_options=override_opts.use_set_from_options,
    )


def normalize_petsc_options(
    petsc_options: PETScOptions | dict[str, object] | None,
    direct: bool = False,
) -> PETScOptions:
    if isinstance(petsc_options, PETScOptions):
        return petsc_options
    if isinstance(petsc_options, dict):
        base = direct_petsc_options() if direct else PETScOptions()
        return PETScOptions(
            ksp_type=str(petsc_options.get("ksp_type", base.ksp_type)),
            pc_type=str(petsc_options.get("pc_type", base.pc_type)),
            pc_factor_mat_solver_type=(
                None if petsc_options.get("pc_factor_mat_solver_type") is None
                else str(petsc_options["pc_factor_mat_solver_type"])
            ),
            options_prefix=(
                None if petsc_options.get("options_prefix") is None
                else str(petsc_options["options_prefix"])
            ),
            use_set_from_options=bool(
                petsc_options.get(
                    "use_set_from_options", base.use_set_from_options
                )
            ),
        )
    return direct_petsc_options() if direct else PETScOptions()


def petsc_options_for_solver(
    chosen_solver: PETScSolverMode,
    petsc_options: PETScOptions | dict[str, object] | None,
) -> PETScOptions | None:
    if chosen_solver != "petsc_spdirect":
        return (
            None if petsc_options is None
            else normalize_petsc_options(petsc_options)
        )

    direct_options = direct_petsc_options()
    if petsc_options is None:
        return direct_options
    petsc_options = merge_petsc_options(direct_options, petsc_options)
    if petsc_options.ksp_type != direct_options.ksp_type:
        raise ValueError(
            "chosen_solver='petsc_spdirect' requires ksp_type='preonly'"
        )
    if petsc_options.pc_type != direct_options.pc_type:
        raise ValueError(
            "chosen_solver='petsc_spdirect' requires pc_type='lu'"
        )
    return merge_petsc_options(direct_options, petsc_options)


def _import_petsc():
    try:
        from petsc4py import PETSc
    except Exception as e:
        raise RuntimeError(
            "chosen_solver='petsc' requires a working petsc4py installation "
            "with the PETSc shared libraries available at runtime"
        ) from e
    return PETSc


def _create_mat(PETSc, K_cond: scipy.sparse.spmatrix):
    K_csr = K_cond.tocsr()
    A = PETSc.Mat().createAIJ(
        size=K_csr.shape,
        csr=(K_csr.indptr, K_csr.indices, K_csr.data),
        comm=PETSc.COMM_SELF,
    )
    A.assemblyBegin()
    A.assemblyEnd()
    return A


def _create_ksp(
    PETSc,
    A,
    rtol: float,
    maxiter: int | None,
    petsc_options: PETScOptions,
):
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(A)
    if petsc_options.options_prefix is not None and hasattr(ksp, "setOptionsPrefix"):
        ksp.setOptionsPrefix(petsc_options.options_prefix)
    ksp.setType(petsc_options.ksp_type)
    pc = ksp.getPC()
    pc.setType(petsc_options.pc_type)
    if (
        petsc_options.pc_factor_mat_solver_type is not None
        and hasattr(pc, "setFactorSolverType")
    ):
        pc.setFactorSolverType(petsc_options.pc_factor_mat_solver_type)
    if maxiter is None:
        ksp.setTolerances(rtol=rtol)
    else:
        ksp.setTolerances(rtol=rtol, max_it=maxiter)
    if petsc_options.use_set_from_options:
        ksp.setFromOptions()
    return ksp


def _solve_with_ksp(PETSc, ksp, rhs: np.ndarray, size: int) -> tuple[np.ndarray, int]:
    b = PETSc.Vec().createWithArray(
        np.ascontiguousarray(rhs, dtype=float),
        comm=PETSc.COMM_SELF,
    )
    x = PETSc.Vec().createSeq(size, comm=PETSc.COMM_SELF)
    ksp.solve(b, x)
    return x.getArray().copy(), ksp.getConvergedReason()


def solve_u_petsc(
    K_cond: scipy.sparse.spmatrix,
    F_cond: np.ndarray,
    rtol: float = 1e-8,
    maxiter: int | None = None,
    petsc_options: PETScOptions | dict[str, object] | None = None,
) -> tuple[np.ndarray, int]:
    PETSc = _import_petsc()
    A = _create_mat(PETSc, K_cond)
    options = normalize_petsc_options(petsc_options)
    ksp = _create_ksp(
        PETSc, A, rtol=rtol, maxiter=maxiter, petsc_options=options
    )
    return _solve_with_ksp(PETSc, ksp, F_cond, K_cond.shape[0])


def solve_u_petsc_multi(
    K_cond: scipy.sparse.spmatrix,
    F_cond: np.ndarray,
    rtol: float = 1e-8,
    maxiter: int | None = None,
    petsc_options: PETScOptions | dict[str, object] | None = None,
) -> tuple[np.ndarray, list[int]]:
    PETSc = _import_petsc()
    rhs = np.asarray(F_cond, dtype=float)
    options = normalize_petsc_options(petsc_options)
    if rhs.ndim == 1:
        u, info = solve_u_petsc(
            K_cond,
            rhs,
            rtol=rtol,
            maxiter=maxiter,
            petsc_options=options,
        )
        return u[:, None], [info]

    A = _create_mat(PETSc, K_cond)
    ksp = _create_ksp(
        PETSc, A, rtol=rtol, maxiter=maxiter, petsc_options=options
    )

    solutions = np.empty_like(rhs, dtype=float)
    infos: list[int] = []
    for idx in range(rhs.shape[1]):
        solutions[:, idx], info = _solve_with_ksp(
            PETSc, ksp, rhs[:, idx], K_cond.shape[0]
        )
        infos.append(info)

    return solutions, infos
