# Backend comparison result

The `skfemntv` backend was compared with `skfem` using the same MMA optimizer,
mesh, constraints, and stopping conditions. The measured native source is
`scikit-fem-native` 0.4.3 (commit
`aa2218266baf5c4f019328e8862290a9455d2f72`), which includes the symbolic
`trace(sym_grad(...))` fix.

## Configuration

- Python 3.12.15 on Linux x86_64
- `scikit-fem` 11.0.0 and `mmahelpy` 0.1.0
- Hex mesh size 0.5: 1,536 elements and 5,967 DOFs
- 10 MMA iterations, volume fraction 0.8, move limit 0.2, filter radius 0.1
- 3 separate-process runs per backend

## Results

| Backend | Median optimization time | Range | Median MMA adapter time | Compliance | Absolute volume error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `skfem` | 95.339 s | 94.796–96.664 s | 70.904 s | 0.4080233177213601 | 1.0e-6 |
| `skfemntv` | 41.372 s | 41.320–42.867 s | 21.480 s | 0.4080233177213412 | 1.0e-6 |

For this case, `skfemntv` completed the optimization loop in 2.304x less time.
Median whole-process time was 101.096 s for `skfem` and 47.265 s for
`skfemntv`. The time inside scikit-topt's MMA adapter was 3.30x lower. Both
backends produced compliance values differing by about 2e-14, the same volume
error (about 1e-6), and matching KKT residuals to numerical precision. The
adapter timing includes filtered/projected volume constraint work as well as
the `mmahelpy` solver call; it is not an isolated timing of the MMA algorithm.

The benchmark command writes the reproducible per-run data to
`results/latest.json`. It records the native version and commit, Python and
platform, timings, and optimization result metrics.

These measurements describe one machine and one problem size. Use the
benchmark script to repeat them on target hardware before relying on the
speedup for another workload.
