![CI](https://github.com/kevin-tofu/scikit-topt/actions/workflows/python-tests.yml/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/scitopt.svg)](https://pypi.org/project/scitopt/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Scikit Topology Optimization (Scikit-Topt)
## Features
 There are few topology optimization codes built on mesh-based frameworks available on GitHub (It may be just that I do not know this field so well though). Moreover, many of them are hard-coded, making them difficult to understand. As far as I know, there doesn’t seem to be a project that serves as a de facto standard. To contribute to the open-source community and education—which I’ve always benefited from—I decided to start this project. 
 
  The currently supported features are as follows:
- Coding with Python  
- Tetrahedral 1st order elements  
- Topology optimization using the density method and the OC (Optimality Criteria) method  
- Multiple objective functions (forces)  
- High-performance computation using sparse matrices and Numba  
- easy installation with pip/poetry



## ToDo
- density interpolation
- density visualization
- coarse to fine optimization
- stabilize
- set break point from the optimization loop
- Add LevelSet
- Add Optimization ALgorithms such as MMA

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

tsk = scitopt.mesh.toy_problem.toy()
cfg = scitopt.core.OC_RAMP_Config()

optimizer = scitopt.core.OC_Optimizer(cfg, tsk)

optimizer.parameterize(preprocess=True)
optimizer.optimize()
```


## Acknowledgements
### Standing on the shoulders of proverbial giants
 This software does not exist in a vacuum.
Scikit-Topt is standing on the shoulders of proverbial giants. In particular, I want to thank the following projects for constituting the technical backbone of the project:
 - Scipy
 - PyAMG
 - Scikit-fem
 - Numba
 - MeshIO
 - Gmsh
 - Matplotlib
 - Topology Optimization Community


## Optiization Algorithm
### OC (Optimality Criteria) method
