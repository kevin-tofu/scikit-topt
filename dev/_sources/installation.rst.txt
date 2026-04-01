Installation
============

Scikit-Topt is a Python library distributed on PyPI as the package
``scikit-topt`` (imported as ``sktopt``).
It supports **Python 3.10–3.13**:

- **Python 3.10–3.12**: fully supported and tested.
- **Python 3.13**: core topology optimization works as normal,
  but VTK-based features (VTU export and image rendering) require
  PyVista/VTK, which do not currently provide wheels for Python 3.13.

Standard Installation
---------------------

Open your terminal or command prompt and run one of the following commands:

- Using pip:

  ::

      pip install scikit-topt

- Using poetry:

  ::

      poetry add scikit-topt

This installs the standard solver stack and does **not** require PETSc.
Typical workflows using ``solver_option="spsolve"`` or ``"cg_pyamg"``
work without ``petsc4py``.

PETSc Optional Installation
---------------------------

If you want to use the PETSc-backed solver paths
(``solver_option="petsc"`` or ``"petsc_spdirect"``), install the optional
extra:

- Using pip:

  ::

      pip install "scikit-topt[petsc4py]"

- Legacy alias with pip:

  ::

      pip install "scikit-topt[petsc]"

- Using poetry:

  ::

      poetry add scikit-topt -E petsc4py

The ``petsc4py`` extra installs the Python bindings only. A working PETSc
installation must also be available at runtime.

PETSc Runtime Setup
-------------------

Common PETSc runtime environment variables are:

- ``PETSC_DIR``: PETSc installation root
- ``PETSC_ARCH``: PETSc build architecture name
- ``LD_LIBRARY_PATH``: shared-library search path on Linux

Typical Linux example:

::

    export PETSC_DIR=/path/to/petsc
    export PETSC_ARCH=arch-linux-c-opt
    export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH"

If PETSc was installed by a package manager or an HPC module, these may
already be configured for you.

You can validate the PETSc runtime with:

::

    python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"

Verify the Installation
-----------------------

After installation is complete, verify that ``sktopt`` imports correctly:

::

    python -c "import sktopt; print(sktopt.__version__)"

If you installed the PETSc extra, also verify that ``petsc4py`` imports
correctly:

::

    python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
