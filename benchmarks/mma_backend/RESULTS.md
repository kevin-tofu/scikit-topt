# MMA backend scaling and filter reuse

## What changed

Profiling showed that MMA's nonlinear volume correction called the nodal
Helmholtz filter up to 40 times per iteration. Each call rebuilt the same
radius-dependent Helmholtz operator and refactorized it, even though the mesh
and filter radius were unchanged. `HelmholtzFilterNodal` now caches its basis,
operator, mass matrix, and sparse factorization, rebuilding them when the
filter radius changes. The volume correction also stops once it reaches a
feasible point within one tenth of the configured volume tolerance.

The benchmark now defaults to recording one snapshot (`--record-times 1`) to
limit mesh export overhead. `run_case.py` captures named optimizer timing
sections, MMA dual iterations, and the number and time of volume evaluations.

## Environment and method

- Python 3.12.15, Linux x86_64
- `scikit-fem` 11.0.0, `mmahelpy` 0.1.0
- Native source: `scikit-fem-native` 0.4.3, commit
  `aa2218266baf5c4f019328e8862290a9455d2f72`
- Same MMA settings, volume fraction 0.8, move limit 0.2, filter radius 0.1
- Separate process per run; medians from 3 runs per backend
- The scaling sweep used 2 iterations. The baseline size was also run for
  10 iterations.

## Results after the filter and volume-correction changes

| Mesh size | Elements | DOFs | Iterations | `skfem` optimize | `skfemntv` optimize | Speedup | `skfem` / `skfemntv` whole process |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 1,536 | 5,967 | 2 | 6.547 s | 5.096 s | 1.285x | 12.439 / 11.024 s |
| 0.4 | 3,000 | 11,088 | 2 | 10.006 s | 7.465 s | 1.340x | 16.226 / 13.848 s |
| 0.3 | 7,560 | 26,460 | 2 | 30.021 s | 21.236 s | 1.414x | 37.875 / 29.012 s |
| 0.5 | 1,536 | 5,967 | 10 | 18.087 s | 10.355 s | 1.747x | 24.015 / 16.250 s |

All runs succeeded. Compliance matched to the displayed precision for each
mesh. Absolute volume error stayed below `1e-6`; the 10-iteration case ended at
`3.74e-7`. In the 26,460-DOF case, the median state solve time was about
14.83 s for `skfem` and 14.47 s for `skfemntv`, while stiffness assembly took
3.46 s and 0.72 s respectively. This shows why the total speedup remains
modest: the selected native backend accelerates assembly, while the PyAMG
or direct state solve is shared by both backends and dominates at this size.

At these measured sizes, the 2-iteration optimization speedup rose gradually
from 1.285x to 1.414x as DOFs increased. This is evidence of a widening gap on
this workload, not a guarantee for larger meshes. The solver and export costs
become a larger share of total runtime, which limits the benefit of faster
assembly.

## Linear solver option at 26,460 DOFs

The same 7,560-element case was also run with the optional `cg_pyamg` solver
(2 iterations, 3 runs per backend). Median timings were:

| Solver | Backend | Optimization | Stiffness assembly | State solve | Compliance | Absolute volume error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `spsolve` | `skfem` | 30.021 s | 3.460 s | 14.829 s | 0.406797463609344 | 8.87e-7 |
| `spsolve` | `skfemntv` | 21.236 s | 0.723 s | 14.468 s | 0.406797463609387 | 8.87e-7 |
| `cg_pyamg` | `skfem` | 20.682 s | 3.119 s | 6.148 s | 0.406797463610499 | 8.87e-7 |
| `cg_pyamg` | `skfemntv` | 12.726 s | 0.732 s | 5.949 s | 0.406797463610532 | 8.87e-7 |

At this size, `cg_pyamg` reduced optimization time by 1.45x with `skfem` and
1.67x with `skfemntv`; the native backend speedup for the same solver was
1.625x. Both solver choices gave matching compliance and KKT residuals to
numerical precision in this case. The project default remains `spsolve`; the
benchmark accepts `--solver-option cg_pyamg` for workloads where its
iterative-solver tradeoffs are appropriate.

The detailed per-run JSON is generated under `results/` and ignored by Git.
Reproduce the 10-iteration baseline with:

```bash
poetry run python benchmarks/mma_backend/compare.py \
  --mesh-size 0.5 --iterations 10 --record-times 1 --repeats 3 \
  --output benchmarks/mma_backend/results/latest.json
```

These measurements describe one machine and one problem. Repeat them on the
target hardware and workload before treating the speedup as representative.
