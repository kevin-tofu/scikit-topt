import os
import sys
import types

import numpy as np
import pytest
import scipy.sparse as sp

from sktopt.fea.solver_elastic import (
    FEM_SimpLinearElasticity,
    LinearSolverConfig,
    solve_u,
    solve_multi_load,
)
from sktopt.fea.solver_heat import (
    FEM_SimpLinearHeatConduction,
    solve_multi_load as solve_heat_multi_load,
)
from sktopt.core.optimizers.common_density import DensityMethodConfig
from sktopt.fea.solver_petsc import PETScOptions, solve_u_petsc, solve_u_petsc_multi


def _spd_diagonal_system(n: int = 8):
    diag = np.arange(1, n + 1, dtype=float)
    K = sp.diags(diag, format="csr")
    F = np.ones(n, dtype=float)
    return K, F, diag


class _FakeVec:
    def __init__(self):
        self._array = None

    def createWithArray(self, array, comm=None):
        self._array = np.array(array, dtype=float, copy=True)
        return self

    def createSeq(self, size, comm=None):
        self._array = np.zeros(size, dtype=float)
        return self

    def getArray(self):
        return self._array


class _FakeMat:
    def __init__(self):
        self.matrix = None

    def createAIJ(self, size, csr, comm=None):
        indptr, indices, data = csr
        self.matrix = sp.csr_matrix((data, indices, indptr), shape=size)
        return self

    def assemblyBegin(self):
        return None

    def assemblyEnd(self):
        return None


class _FakePC:
    last_instance = None

    def __init__(self):
        self.pc_type = None
        self.factor_solver_type = None
        _FakePC.last_instance = self

    def setType(self, pc_type):
        self.pc_type = pc_type

    def setFactorSolverType(self, solver_type):
        self.factor_solver_type = solver_type


class _FakeKSP:
    solve_calls = 0
    last_instance = None

    def __init__(self):
        self.matrix = None
        self.pc = _FakePC()
        self.ksp_type = None
        self.tolerances = {}
        self.options_prefix = None
        _FakeKSP.last_instance = self

    def create(self, comm=None):
        return self

    def setOperators(self, A):
        self.matrix = A.matrix

    def setType(self, ksp_type):
        self.ksp_type = ksp_type

    def getPC(self):
        return self.pc

    def setTolerances(self, **kwargs):
        self.tolerances.update(kwargs)

    def setOptionsPrefix(self, prefix):
        self.options_prefix = prefix

    def setFromOptions(self):
        return None

    def solve(self, b, x):
        _FakeKSP.solve_calls += 1
        x._array[:] = sp.linalg.spsolve(self.matrix, b.getArray())

    def getConvergedReason(self):
        return 4


def _install_fake_petsc(monkeypatch):
    _FakeKSP.solve_calls = 0
    _FakeKSP.last_instance = None
    _FakePC.last_instance = None
    fake_petsc = types.SimpleNamespace(
        PETSc=types.SimpleNamespace(
            COMM_SELF=object(),
            Mat=_FakeMat,
            Vec=_FakeVec,
            KSP=_FakeKSP,
        )
    )
    monkeypatch.setitem(sys.modules, "petsc4py", fake_petsc)


def _remove_petsc4py(monkeypatch):
    monkeypatch.setitem(sys.modules, "petsc4py", None)


def test_solve_u_petsc_matches_direct_with_fake_backend(monkeypatch):
    _install_fake_petsc(monkeypatch)
    K, F, diag = _spd_diagonal_system()

    u_petsc, info = solve_u_petsc(K, F, rtol=1e-10, maxiter=200)
    u_direct = F / diag

    assert info > 0
    np.testing.assert_allclose(u_petsc, u_direct, rtol=1e-8, atol=1e-10)


def test_solve_u_dispatches_to_petsc(monkeypatch):
    _install_fake_petsc(monkeypatch)
    K, F, diag = _spd_diagonal_system()

    u_petsc = solve_u(K, F, chosen_solver="petsc", rtol=1e-10, maxiter=200)
    u_direct = F / diag

    assert _FakeKSP.solve_calls == 1
    np.testing.assert_allclose(u_petsc, u_direct, rtol=1e-8, atol=1e-10)


def test_solve_u_accepts_linear_solver_config(monkeypatch):
    _install_fake_petsc(monkeypatch)
    K, F, diag = _spd_diagonal_system()

    u_petsc = solve_u(
        K,
        F,
        chosen_solver=LinearSolverConfig(
            solver="petsc",
            rtol=1e-10,
            maxiter=200,
        ),
    )
    u_direct = F / diag

    assert _FakeKSP.solve_calls == 1
    np.testing.assert_allclose(u_petsc, u_direct, rtol=1e-8, atol=1e-10)


def test_density_method_config_builds_solver_config():
    cfg = DensityMethodConfig(
        solver_option="petsc_spdirect",
        petsc_options={"pc_factor_mat_solver_type": "mumps"},
    )

    assert cfg.solver_config.solver == "petsc_spdirect"
    assert cfg.solver_config.petsc_options is not None
    assert cfg.solver_config.petsc_options.pc_factor_mat_solver_type == "mumps"


def test_density_method_config_normalizes_petsc_spdirect_defaults():
    cfg = DensityMethodConfig(solver_option="petsc_spdirect")

    assert cfg.solver_config.solver == "petsc_spdirect"
    assert cfg.solver_config.petsc_options is not None
    assert cfg.solver_config.petsc_options.ksp_type == "preonly"
    assert cfg.solver_config.petsc_options.pc_type == "lu"


def test_solve_u_petsc_explicit_failure_does_not_fallback(monkeypatch):
    _remove_petsc4py(monkeypatch)
    K, F, _diag = _spd_diagonal_system()

    with pytest.raises(RuntimeError, match="working petsc4py installation"):
        solve_u(K, F, chosen_solver="petsc")


def test_solve_u_petsc_multi_reuses_ksp(monkeypatch):
    _install_fake_petsc(monkeypatch)
    K, F, diag = _spd_diagonal_system()
    rhs = np.column_stack([F, 2.0 * F])

    u_petsc, infos = solve_u_petsc_multi(K, rhs, rtol=1e-10, maxiter=200)
    u_direct = np.column_stack([F / diag, 2.0 * F / diag])

    assert infos == [4, 4]
    assert _FakeKSP.solve_calls == 2
    np.testing.assert_allclose(u_petsc, u_direct, rtol=1e-8, atol=1e-10)


def test_solve_u_dispatches_to_petsc_spdirect(monkeypatch):
    _install_fake_petsc(monkeypatch)
    K, F, diag = _spd_diagonal_system()

    u_petsc = solve_u(
        K,
        F,
        chosen_solver="petsc_spdirect",
        petsc_options={
            "pc_factor_mat_solver_type": "mumps",
            "options_prefix": "elas_",
        },
    )
    u_direct = F / diag

    assert _FakeKSP.last_instance.ksp_type == "preonly"
    assert _FakeKSP.last_instance.options_prefix == "elas_"
    assert _FakePC.last_instance.pc_type == "lu"
    assert _FakePC.last_instance.factor_solver_type == "mumps"
    np.testing.assert_allclose(u_petsc, u_direct, rtol=1e-8, atol=1e-10)


def test_solve_u_rejects_inconsistent_petsc_spdirect_options(monkeypatch):
    _install_fake_petsc(monkeypatch)
    K, F, _diag = _spd_diagonal_system()

    with pytest.raises(ValueError, match="requires ksp_type='preonly'"):
        solve_u(
            K,
            F,
            chosen_solver="petsc_spdirect",
            petsc_options=PETScOptions(ksp_type="cg", pc_type="lu"),
        )


def test_solve_multi_load_dispatches_to_petsc(monkeypatch):
    import skfem

    from sktopt.mesh.toy_problem import create_box_hex

    _install_fake_petsc(monkeypatch)

    mesh = create_box_hex(1.0, 1.0, 1.0, 1.0)
    basis = skfem.Basis(mesh, skfem.ElementVector(skfem.ElementHex1()))
    n_dof = basis.N
    rho = np.ones(mesh.nelements, dtype=float)
    dirichlet_dofs = basis.get_dofs(
        nodes=np.where(mesh.p[0] == mesh.p[0].min())[0]
    ).all()
    free_dofs = np.setdiff1d(np.arange(n_dof), dirichlet_dofs)

    force_0 = np.zeros(n_dof, dtype=float)
    force_1 = np.zeros(n_dof, dtype=float)
    force_0[free_dofs[0]] = 1.0
    force_1[free_dofs[min(1, len(free_dofs) - 1)]] = 2.0
    u_all = np.zeros((n_dof, 2), dtype=float)

    F_stack = solve_multi_load(
        basis=basis,
        free_dofs=free_dofs,
        dirichlet_dofs=dirichlet_dofs,
        force_list=[force_0, force_1],
        E0=1.0,
        Emin=1e-3,
        p=3.0,
        nu0=0.3,
        rho=rho,
        u_all=u_all,
        solver="petsc",
    )

    assert F_stack.shape == (n_dof, 2)
    assert u_all.shape == (n_dof, 2)
    assert np.isfinite(u_all).all()
    assert _FakeKSP.solve_calls == 2


def test_solve_multi_load_accepts_linear_solver_config(monkeypatch):
    import skfem

    from sktopt.mesh.toy_problem import create_box_hex

    _install_fake_petsc(monkeypatch)

    mesh = create_box_hex(1.0, 1.0, 1.0, 1.0)
    basis = skfem.Basis(mesh, skfem.ElementVector(skfem.ElementHex1()))
    n_dof = basis.N
    rho = np.ones(mesh.nelements, dtype=float)
    dirichlet_dofs = basis.get_dofs(
        nodes=np.where(mesh.p[0] == mesh.p[0].min())[0]
    ).all()
    free_dofs = np.setdiff1d(np.arange(n_dof), dirichlet_dofs)

    force_0 = np.zeros(n_dof, dtype=float)
    force_1 = np.zeros(n_dof, dtype=float)
    force_0[free_dofs[0]] = 1.0
    force_1[free_dofs[min(1, len(free_dofs) - 1)]] = 2.0
    u_all = np.zeros((n_dof, 2), dtype=float)

    solve_multi_load(
        basis=basis,
        free_dofs=free_dofs,
        dirichlet_dofs=dirichlet_dofs,
        force_list=[force_0, force_1],
        E0=1.0,
        Emin=1e-3,
        p=3.0,
        nu0=0.3,
        rho=rho,
        u_all=u_all,
        solver_config=LinearSolverConfig(solver="petsc", rtol=1e-10, maxiter=200),
    )

    assert np.isfinite(u_all).all()
    assert _FakeKSP.solve_calls == 2


def test_fem_elasticity_prefers_solver_config_over_legacy_options():
    task = types.SimpleNamespace(E=1.0)
    fem = FEM_SimpLinearElasticity(
        task=task,
        E_min_coeff=1e-3,
        solver_config=LinearSolverConfig(
            solver="petsc_spdirect",
            petsc_options=PETScOptions(pc_factor_mat_solver_type="mumps"),
        ),
        solver_option="spsolve",
        petsc_options=PETScOptions(),
    )

    assert fem.solver_config.solver == "petsc_spdirect"
    assert fem.solver_option == "petsc_spdirect"
    assert fem.petsc_options is not None
    assert fem.petsc_options.pc_factor_mat_solver_type == "mumps"


def test_fem_heat_prefers_solver_config_over_legacy_options():
    task = types.SimpleNamespace(k=1.0)
    fem = FEM_SimpLinearHeatConduction(
        task=task,
        E_min_coeff=1e-3,
        solver_config=LinearSolverConfig(solver="petsc", rtol=1e-7, maxiter=50),
        solver_option="spsolve",
    )

    assert fem.solver_config.solver == "petsc"
    assert fem.solver_option == "petsc"
    assert fem.solver_config.rtol == 1e-7
    assert fem.solver_config.maxiter == 50


def test_heat_solve_multi_load_dispatches_to_petsc(monkeypatch):
    import skfem

    from sktopt.mesh.toy_problem import create_box_hex

    _install_fake_petsc(monkeypatch)

    mesh = create_box_hex(1.0, 1.0, 1.0, 1.0)
    basis = skfem.Basis(mesh, skfem.ElementHex1())
    n_dof = basis.N
    rho = np.ones(mesh.nelements, dtype=float)
    u_all = np.zeros((n_dof, 2), dtype=float)
    dirichlet_nodes = np.where(mesh.p[0] == mesh.p[0].min())[0]

    K_csr, emit, dirichlet_dofs_list = solve_heat_multi_load(
        basis=basis,
        free_dofs=np.array([], dtype=int),
        dirichlet_nodes_list=[dirichlet_nodes, dirichlet_nodes],
        dirichlet_values_list=[0.0, 1.0],
        robin_bilinear=sp.csr_matrix((n_dof, n_dof)),
        robin_linear=np.zeros(n_dof, dtype=float),
        k0=1.0,
        kmin=1e-3,
        p=3.0,
        rho=rho,
        u_all=u_all,
        solver="petsc",
    )

    assert K_csr.shape == (n_dof, n_dof)
    assert emit.shape == (n_dof,)
    assert len(dirichlet_dofs_list) == 2
    assert _FakeKSP.solve_calls == 2
    assert np.isfinite(u_all).all()


def test_heat_solve_multi_load_dispatches_to_petsc_spdirect(monkeypatch):
    import skfem

    from sktopt.mesh.toy_problem import create_box_hex

    _install_fake_petsc(monkeypatch)

    mesh = create_box_hex(1.0, 1.0, 1.0, 1.0)
    basis = skfem.Basis(mesh, skfem.ElementHex1())
    n_dof = basis.N
    rho = np.ones(mesh.nelements, dtype=float)
    u_all = np.zeros((n_dof, 2), dtype=float)
    dirichlet_nodes = np.where(mesh.p[0] == mesh.p[0].min())[0]

    solve_heat_multi_load(
        basis=basis,
        free_dofs=np.array([], dtype=int),
        dirichlet_nodes_list=[dirichlet_nodes, dirichlet_nodes],
        dirichlet_values_list=[0.0, 1.0],
        robin_bilinear=sp.csr_matrix((n_dof, n_dof)),
        robin_linear=np.zeros(n_dof, dtype=float),
        k0=1.0,
        kmin=1e-3,
        p=3.0,
        rho=rho,
        u_all=u_all,
        solver_config=LinearSolverConfig(
            solver="petsc_spdirect",
            petsc_options=PETScOptions(
                ksp_type="preonly",
                pc_type="lu",
                pc_factor_mat_solver_type="mumps",
            ),
        ),
    )

    assert _FakeKSP.last_instance.ksp_type == "preonly"
    assert _FakePC.last_instance.pc_type == "lu"
    assert _FakePC.last_instance.factor_solver_type == "mumps"
    assert np.isfinite(u_all).all()


def test_fem_heat_averaged_temp_uses_petsc_for_state_and_adjoint(monkeypatch):
    import skfem

    from sktopt.mesh.toy_problem import create_box_hex

    _install_fake_petsc(monkeypatch)

    mesh = create_box_hex(1.0, 1.0, 1.0, 1.0)
    basis = skfem.Basis(mesh, skfem.ElementHex1())
    n_dof = basis.N
    dirichlet_nodes = np.where(mesh.p[0] == mesh.p[0].min())[0]
    task = types.SimpleNamespace(
        k=1.0,
        basis=basis,
        free_dofs=np.array([], dtype=int),
        dirichlet_nodes=[dirichlet_nodes, dirichlet_nodes],
        dirichlet_values=[0.0, 1.0],
        robin_bilinear=[],
        robin_linear=[],
        robin_coefficient=0.0,
        robin_bc_value=0.0,
        design_robin_boundary=False,
        objective="averaged_temp",
    )

    fem = FEM_SimpLinearHeatConduction(
        task=task,
        E_min_coeff=1e-3,
        solver_config=LinearSolverConfig(solver="petsc", rtol=1e-10, maxiter=200),
    )

    rho = np.ones(mesh.nelements, dtype=float)
    u_all = np.zeros((n_dof, 2), dtype=float)
    objectives = fem.objectives_multi_load(rho=rho, p=3.0, u_dofs=u_all)

    assert objectives.shape == (2,)
    assert fem.λ_all is not None
    assert fem.λ_all.shape == u_all.shape
    assert np.isfinite(u_all).all()
    assert np.isfinite(fem.λ_all).all()
    assert _FakeKSP.solve_calls == 4


def test_solve_multi_load_dispatches_to_petsc_spdirect(monkeypatch):
    import skfem

    from sktopt.mesh.toy_problem import create_box_hex

    _install_fake_petsc(monkeypatch)

    mesh = create_box_hex(1.0, 1.0, 1.0, 1.0)
    basis = skfem.Basis(mesh, skfem.ElementVector(skfem.ElementHex1()))
    n_dof = basis.N
    rho = np.ones(mesh.nelements, dtype=float)
    dirichlet_dofs = basis.get_dofs(
        nodes=np.where(mesh.p[0] == mesh.p[0].min())[0]
    ).all()
    free_dofs = np.setdiff1d(np.arange(n_dof), dirichlet_dofs)

    force_0 = np.zeros(n_dof, dtype=float)
    force_1 = np.zeros(n_dof, dtype=float)
    force_0[free_dofs[0]] = 1.0
    force_1[free_dofs[min(1, len(free_dofs) - 1)]] = 2.0
    u_all = np.zeros((n_dof, 2), dtype=float)

    solve_multi_load(
        basis=basis,
        free_dofs=free_dofs,
        dirichlet_dofs=dirichlet_dofs,
        force_list=[force_0, force_1],
        E0=1.0,
        Emin=1e-3,
        p=3.0,
        nu0=0.3,
        rho=rho,
        u_all=u_all,
        solver="petsc_spdirect",
        petsc_options={"pc_factor_mat_solver_type": "superlu_dist"},
    )

    assert _FakeKSP.last_instance.ksp_type == "preonly"
    assert _FakePC.last_instance.pc_type == "lu"
    assert _FakePC.last_instance.factor_solver_type == "superlu_dist"
    assert np.isfinite(u_all).all()


def test_solve_u_petsc_matches_direct_when_real_petsc_available():
    if os.environ.get("SKTOPT_RUN_REAL_PETSC") != "1":
        pytest.skip("real PETSc runtime test is opt-in; set SKTOPT_RUN_REAL_PETSC=1")

    pytest.importorskip("petsc4py")
    try:
        from petsc4py import PETSc  # noqa: F401
    except Exception as exc:
        pytest.skip(f"PETSc runtime is unavailable: {exc}")

    K, F, diag = _spd_diagonal_system()

    u_petsc, _info = solve_u_petsc(K, F, rtol=1e-10, maxiter=200)
    u_direct = F / diag

    np.testing.assert_allclose(u_petsc, u_direct, rtol=1e-8, atol=1e-10)
