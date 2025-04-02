# Scikit Topology Optimization (Scikit-Topt)
## Abstract
 

### Install Package
```bash
pip install scitopt
poetry add scitopt
```

### Optimize Toy Problem with command line.
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  MKL_NUM_THREADS=1 PYTHONPATH=./ python ./scitopt/core/optimizer/oc.py \
 --dst_path ./result/test1_oc \
 --p 3.0 \
 --p_rate 12.0 \
 --filter_radius 0.7 \
 --move_limit 0.2 \
 --move_limit_rate 10.0 \
 --vol_frac 0.4 \
 --vol_frac_rate 5.0 \
 --beta 5.0 \
 --beta_rate 1.0 \
 --eta 1.0 \
 --record_times 80 \
 --max_iters 200
```

### Analyze from Python Script

```Python
import scitopt

toy = scitopt.mesh.toy_problem.toy()
cfg = scitopt.core.OC_RAMP_Config

optimizer = scitopt.core.OC_Optimizer(cfg, toy)

optimizer.parameterize(preprocess=True)
optimizer.optimize()
```


## Acknowledgements
 My software and research does not exist in a vacuum.
Scikit-Topt is standing on the shoulders of proverbial giants. In particular, I want to thank the following projects for constituting the technical backbone of the project:
 - Scipy
 - Scikit-fem
 - Numba
 - MeshIO
 - Gmsh
 - Topology Optimization Community

