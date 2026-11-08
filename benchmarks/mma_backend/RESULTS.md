# Backend comparison result

The `skfemntv` backend was compared with `skfem` using the same MMA optimizer,
mesh, constraints, and stopping conditions. The measured native code includes
the symbolic `trace(sym_grad(...))` fix later released as
`scikit-fem-native` 0.4.3 (commit `aa221826`).

## Configuration

- Python 3.12.15 on Linux x86_64
- `scikit-fem` 11.0.0 and `mmahelpy` 0.1.0
- Hex mesh size 0.5: 1,536 elements and 5,967 DOFs
- 10 MMA iterations, volume fraction 0.8, move limit 0.2, filter radius 0.1
- 3 separate-process runs per backend

## Results

| Backend | Median optimization time | Range | Median MMA adapter time | Compliance | Absolute volume error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `skfem` | 96.944 s | 96.728–98.738 s | 71.975 s | 0.4080233177213601 | 1.0e-6 |
| `skfemntv` | 42.114 s | 41.875–52.174 s | 21.795 s | 0.4080233177213412 | 1.0e-6 |

For this case, `skfemntv` completed the optimization loop in 2.30x less time.
The time inside scikit-topt's MMA adapter was 3.30x lower. Both backends
produced the same compliance to about 2e-14 and the same volume error. The
adapter timing includes filtered/projected volume constraint work as well as
the `mmahelpy` solver call; it is not an isolated timing of the MMA algorithm.

These measurements describe one machine and one problem size. Use the
benchmark script to repeat them on target hardware before relying on the
speedup for another workload.
