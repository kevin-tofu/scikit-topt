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
RESULT_DIR="${RESULT_DIR:-./result/logmoc_heat_double_1_${SOLVER_OPTION}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_DIR="${PYTHONPATH_DIR:-./scikit-topt}"
OMP_THREADS="${OMP_THREADS:-3}"

echo "Running multi-load heat LogMOC benchmark"
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

const = sktopt.tools.SchedulerConfig.constant

tsk = box_oc_heat.get_task_0(
    multiple_robin=True,
    design_robin_boundary=True,
)
cfg = sktopt.core.LogMOC_Config(
    dst_path=os.environ["RESULT_DIR"],
    export_img=True,
    p=const(target_value=3.0),
    vol_frac=const(target_value=0.7),
    filter_radius=const(target_value=0.2),
    move_limit=const(target_value=0.08),
    beta=const(target_value=1.0),
    eta=const(target_value=0.5),
    filter_type="helmholtz",
    solver_option=os.environ["SOLVER_OPTION"],
    max_iters=40,
    record_times=40,
    mu_p=0.5,
    augmented_lagrangian_mu=0.01,
    lambda_v=1e-2,
    lambda_decay=0.97,
)

optimizer = sktopt.core.LogMOC_Optimizer(cfg, tsk)
optimizer.parameterize()
optimizer.optimize()
PY
