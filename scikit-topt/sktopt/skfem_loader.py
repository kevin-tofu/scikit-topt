"""Load the configured finite-element assembly backend.

Set ``SKTOPT_FEM_BACKEND`` to ``skfem`` (the default) or ``skfemntv`` before
importing :mod:`sktopt` to choose the implementation.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any


_BACKENDS = {
    "skfem": "skfem",
    "skfemntv": "skfemntv",
    "scikit-fem-native": "skfemntv",
}


def load_skfem(backend: str | None = None) -> ModuleType:
    """Return the selected scikit-fem-compatible module."""
    requested = backend or os.environ.get("SKTOPT_FEM_BACKEND", "skfem")
    module_name = _BACKENDS.get(requested.lower())
    if module_name is None:
        choices = ", ".join(("skfem", "skfemntv"))
        raise ValueError(f"Unknown FEM backend {requested!r}; choose {choices}.")
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package_name = "scikit-fem-native" if module_name == "skfemntv" else "scikit-fem"
        raise ImportError(
            f"FEM backend {requested!r} requires the {package_name} package."
        ) from exc


backend = os.environ.get("SKTOPT_FEM_BACKEND", "skfem").lower()
skfem = load_skfem(backend)
_compat = None

# scikit-fem-native intentionally focuses on assembly and does not implement
# scikit-fem's linear solver/constraint utilities. Keep those utilities
# available from scikit-fem when both packages are installed, while all core
# backend types and assembly operations continue to come from the selection.
if skfem.__name__ == "skfemntv":
    try:
        _compat = importlib.import_module("skfem")
    except ImportError:
        _compat = None
    if _compat is not None:
        for _name in dir(_compat):
            if not _name.startswith("_") and not hasattr(skfem, _name):
                setattr(skfem, _name, getattr(_compat, _name))


def __getattr__(name: str) -> Any:
    """Expose backend symbols and its common submodule APIs from this loader."""
    try:
        return getattr(skfem, name)
    except AttributeError:
        pass

    submodules = (
        "helpers",
        "assembly",
        "models.elasticity",
        "utils",
        "quadrature",
        "mapping",
        "io.json",
    )
    module_names = [f"{skfem.__name__}.{submodule}" for submodule in submodules]
    if _compat is not None:
        module_names.extend(f"{_compat.__name__}.{submodule}" for submodule in submodules)

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"FEM backend {backend!r} does not provide {name!r}.")


__all__ = ["backend", "load_skfem", "skfem"]
