from __future__ import annotations

from dataclasses import dataclass


def _missing_petsc(*_args, **_kwargs):
    raise RuntimeError(
        "PETSc support is unavailable. Install the optional PETSc support "
        "and ensure `sktopt.fea.solver_petsc` is importable."
    )


try:
    from sktopt.fea.solver_petsc import (  # type: ignore[attr-defined]
        PETScOptions,
        normalize_petsc_options,
        petsc_options_for_solver,
        solve_u_petsc,
        solve_u_petsc_multi,
    )
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class PETScOptions:
        ksp_type: str = "cg"
        pc_type: str = "gamg"
        pc_factor_mat_solver_type: str | None = None
        options_prefix: str | None = None
        use_set_from_options: bool = True

    def normalize_petsc_options(
        petsc_options: PETScOptions | dict[str, object] | None,
        direct: bool = False,
    ) -> PETScOptions:
        base = PETScOptions(
            ksp_type="preonly" if direct else "cg",
            pc_type="lu" if direct else "gamg",
        )
        if isinstance(petsc_options, PETScOptions):
            return petsc_options
        if isinstance(petsc_options, dict):
            return PETScOptions(
                ksp_type=str(petsc_options.get("ksp_type", base.ksp_type)),
                pc_type=str(petsc_options.get("pc_type", base.pc_type)),
                pc_factor_mat_solver_type=(
                    None
                    if petsc_options.get("pc_factor_mat_solver_type") is None
                    else str(petsc_options["pc_factor_mat_solver_type"])
                ),
                options_prefix=(
                    None
                    if petsc_options.get("options_prefix") is None
                    else str(petsc_options["options_prefix"])
                ),
                use_set_from_options=bool(
                    petsc_options.get(
                        "use_set_from_options",
                        base.use_set_from_options,
                    )
                ),
            )
        return base

    def petsc_options_for_solver(
        chosen_solver: str,
        petsc_options: PETScOptions | dict[str, object] | None,
    ) -> PETScOptions | None:
        if chosen_solver not in {"petsc", "petsc_spdirect"}:
            return (
                None if petsc_options is None
                else normalize_petsc_options(petsc_options)
            )
        return _missing_petsc()

    solve_u_petsc = _missing_petsc
    solve_u_petsc_multi = _missing_petsc
