#!/usr/bin/env bash

set -euo pipefail

# PETSc runtime discovery is environment-specific. Export these before running
# when PETSc is installed outside a standard library path.
#
# Example:
#   export PETSC_DIR=/path/to/petsc
#   export PETSC_ARCH=arch-linux-c-opt
#   export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH"
#
# This benchmark is pinned to the PETSc sparse-direct backend.
SOLVER_OPTION="petsc_spdirect"
RESULT_DIR="${RESULT_DIR:-./result/oc_heat_double_1_${SOLVER_OPTION}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_DIR="${PYTHONPATH_DIR:-./scikit-topt}"
OMP_THREADS="${OMP_THREADS:-3}"

echo "Running multi-load heat OC benchmark"
echo "  solver: ${SOLVER_OPTION}"
echo "  result: ${RESULT_DIR}"
if [[ -n "${PETSC_DIR:-}" ]]; then
  echo "  PETSC_DIR: ${PETSC_DIR}"
fi
if [[ -n "${PETSC_ARCH:-}" ]]; then
  echo "  PETSC_ARCH: ${PETSC_ARCH}"
fi

/usr/bin/time -p env \
  OMP_NUM_THREADS="${OMP_THREADS}" \
  OPENBLAS_NUM_THREADS="${OMP_THREADS}" \
  MKL_NUM_THREADS="${OMP_THREADS}" \
  PYTHONPATH="${PYTHONPATH_DIR}" \
  RESULT_DIR="${RESULT_DIR}" \
  SOLVER_OPTION="${SOLVER_OPTION}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import sktopt
from examples.tutorial import box_oc_heat

tsk = box_oc_heat.get_task_1()
cfg = box_oc_heat.get_cfg()
cfg.dst_path = os.environ["RESULT_DIR"]
cfg.solver_option = os.environ["SOLVER_OPTION"]

optimizer = sktopt.core.OC_Optimizer(cfg, tsk)
optimizer.parameterize()
optimizer.optimize()
PY
