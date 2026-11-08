# MMA backend comparison

This benchmark compares the same density-based MMA optimization using
`skfem` and `skfemntv`. Each backend runs in a separate Python process because
`SKTOPT_FEM_BACKEND` is read when `sktopt` is imported.

From the repository root, install the project dependencies and run:

```bash
poetry install
poetry run python benchmarks/mma_backend/compare.py \
  --mesh-size 0.5 --iterations 10 --record-times 1 --repeats 3
```

Use the same mesh, iteration count, volume fraction, move limit, and filter
radius for both backends. The summary reports median optimization time,
whole-process time, time spent inside scikit-topt's MMA density-update
adapter, compliance, and volume error. Each per-run record also includes
named optimizer timing sections, MMA dual iteration counts, and
volume-correction evaluation counts. `--record-times` defaults to 1 to reduce
mesh-export overhead; set it higher to measure a different output frequency.
Setup timing includes filter matrix assembly and factorization. The full
report is written to `results/latest.json` and includes the native submodule
version and commit. Failed runs include their exception and traceback so
backend compatibility issues are visible.

The default linear solver is `spsolve`. To compare with the iterative AMG
solver on a larger system, pass `--solver-option cg_pyamg`. The selected solver
is recorded in the JSON report. Compare compliance, volume error, and solver
time as well as total runtime when changing solver options.

For a faster smoke run, use `--mesh-size 1.0 --iterations 2 --repeats 1`.
Do not treat a smoke run as a performance result; use multiple repeats on a
mesh large enough that optimization takes several seconds. Native
compilation inside the assembly path is included in optimization time.
Compare medians and keep the same software versions and hardware when
reporting results.
