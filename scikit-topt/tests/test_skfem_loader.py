"""Regression tests for backend-specific compatibility helpers."""

import importlib.util
import os
import subprocess
import sys

import pytest


def test_native_backend_enforce_accepts_native_dofs_view():
    if importlib.util.find_spec("skfemntv") is None:
        pytest.skip("scikit-fem-native is not installed")

    environment = os.environ.copy()
    environment["SKTOPT_FEM_BACKEND"] = "skfemntv"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import numpy as np
from scipy.sparse import eye
from sktopt.skfem_loader import skfem

mesh = skfem.MeshTri.init_tensor([0.0, 1.0], [0.0, 1.0])
basis = skfem.Basis(mesh, skfem.ElementTriP1())
dofs = basis.get_dofs(facets=mesh.boundary_facets())
matrix, rhs = skfem.enforce(eye(basis.N, format="csr"), np.zeros(basis.N), D=dofs)
assert matrix.shape == (basis.N, basis.N)
assert rhs.shape == (basis.N,)
assert skfem.enforce.__module__ == "skfemntv.utils"
""",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
