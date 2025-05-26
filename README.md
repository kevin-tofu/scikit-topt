[![PyPI version](https://img.shields.io/pypi/v/scitopt.svg?cacheSeconds=60)](https://pypi.org/project/scitopt/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/957674835.svg)](https://doi.org/10.5281/zenodo.15441499)
[![Python Version](https://img.shields.io/pypi/pyversions/scitopt.svg)](https://pypi.org/project/scitopt/)
[![PyPI Downloads](https://static.pepy.tech/badge/scitopt)](https://pepy.tech/projects/scitopt)
![CI](https://github.com/kevin-tofu/scikit-topt/actions/workflows/python-tests.yml/badge.svg)
![CI](https://github.com/kevin-tofu/scikit-topt/actions/workflows/sphinx.yml/badge.svg)


# 🧠 Scikit Topt
**A lightweight, flexible Python library for topology optimization built on top of Scikit Libraries**
- [scipy](https://scipy.org/)
- [scikit-fem](https://github.com/kinnala/scikit-fem)


## Examples and Fieatures
### Example 1 : Single Load Condition
<p align="center">
  <img src="assets/ex-pull-down-0.gif" alt="Optimization Process Pull-Down-0" width="400" style="margin-right: 20px;">
  <img src="assets/ex-pull-down-1.jpg" alt="Optimization Process Pull-Down-1" width="400">
</p>

### Example 2 : Multiple Load Condition
<p align="center">
  <img src="assets/ex-multi-load-condition.png" alt="multi-load-condition" width="400" style="margin-right: 20px;">
  <img src="assets/ex-multi-load-v-50.jpg" alt="multi-load-condition-distribution" width="400">
</p>

### Progress Report
<p align="center">
  <img src="assets/ex-progress-report.jpg" alt="multi-load-condition-progress" width="600">
</p>


## Features
 To contribute to the open-source community and education—which I’ve always benefited from—I decided to start this project. 
 
The currently supported features are as follows:
- Coding with Python  
- easy installation with pip/poetry
- Implement FEA on unstructured mesh using scikit-fem
- Topology optimization using the density method and its optimization algorithm
  - Optimality Criteria (OC) Method  
  - (Log-Space) Modified OC Method 
  - Lagrangian Method
- able to handle multiple objectives / constraints
- High-performance computation using sparse matrices with Scipy and PyAMG  
- has a function to monitor the transition of parameters.


## 📖 Citation

If you use Scikit Topt in your research or software, please cite it as:

```bibtex
@misc{scikit-topt2025,
author = {Kohei Watanabe},
title = {{Scikit Topt}: A Python library for topology optimization with {Scipy Ecosystem}},
publisher = {Zenodo},
year = {2025},
doi = {10.5281/zenodo.15441499},
}
```


## Usage
### Install Package
```bash
pip install scitopt
poetry add scitopt
```


### How to define a task from mesh structure (or .msh file)

```Python
import pathlib
import skfem
import scitopt

# Load or Creat a mesh model
x_len, y_len, z_len = 1.0, 1.0, 1.0
element_size = 0.1
e = skfem.ElementVector(skfem.ElementHex1())
# msh_path = "model.msh"
# mesh = skfem.MeshHex.load(pathlib.Path(msh_path))

# define basis
mesh = scitopt.mesh.toy_problem.create_box_hex(x_len, y_len, z_len, element_size)
basis = skfem.Basis(mesh, e, intorder=3)

# Specify Dirichlet Boundary Conditions
dirichlet_points = scitopt.mesh.utils.get_point_indices_in_range(
    basis, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
)
dirichlet_nodes = basis.get_dofs(nodes=dirichlet_points).all()

# Define Force Vector
F_points = scitopt.mesh.utils.get_point_indices_in_range(
    basis,
    (x_len, x_len),
    (y_len*2/5, y_len*3/5),
    (z_len*2/5, z_len*3/5)
)
F_nodes = basis.get_dofs(nodes=F_points).nodal["u^1"]
F = 100

# Specify Design Field
design_elements = scitopt.mesh.utils.get_elements_in_box(
    mesh,
    (0.0, x_len), (0.0, y_len), (0.0, z_len)
)

# Define it as a task
tsk = scitopt.mesh.task.TaskConfig.from_defaults(
    210e9,
    0.30,
    basis,
    dirichlet_points,
    dirichlet_nodes,
    F_points,
    F_nodes,
    F,
    design_elements
)
```


### Optimize Toy Problem with Python Script

```Python
import scitopt

tsk = scitopt.mesh.toy_problem.toy1()
cfg = scitopt.core.LogMOC_Config()

optimizer = scitopt.core.LogMOC_Optimizer(cfg, tsk)

optimizer.parameterize()
optimizer.optimize()
```


### Optimize Toy Problem with command line.
```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONPATH=./ python ./scitopt/core/optimizer/logmoc.py \
 --dst_path ./result/base_moc_down \
 --interpolation SIMP \
 --p_init 1.0 \
 --p 3.0 \
 --p_step -4 \
 --filter_radius_init 0.2 \
 --filter_radius 0.20 \
 --filter_radius_step -2 \
 --move_limit_init 0.20 \
 --move_limit 0.02 \
 --move_limit_step -2 \
 --vol_frac_init 0.60 \
 --vol_frac 0.40 \
 --vol_frac_step -2 \
 --beta_init 1.0 \
 --beta 2.0 \
 --beta_step 2 \
 --beta_curvature 2.0 \
 --percentile_init 70 \
 --percentile -90 \
 --percentile_step -4 \
 --eta 0.8 \
 --record_times 100 \
 --max_iters 600 \
 --lambda_v 0.01 \
 --lambda_decay  0.80 \
 --mu_p 2000 \
 --lambda_lower 1e-9 \
 --lambda_upper 1e+9 \
 --export_img true \
 --sensitivity_filter false \
 --task_name down_box \
 --solver_option pyamg \
 --rho_min 1e-2 \
 --E0 210e9 \
 --E_min 210e5 \
 --design_dirichlet true
```


## Optiization Algorithm
Optimization Algorithms and Techniques are briefly summarized here.  
[Optimization Algorithms and Techniques](https://kevin-tofu.github.io/scikit-topt/optimization.html)


## Acknowledgements
### Standing on the shoulders of proverbial giants
 This software does not exist in a vacuum.
Scikit-Topt is standing on the shoulders of proverbial giants. In particular, I want to thank the following projects for constituting the technical backbone of the project:
 - Scipy
 - Scikit-fem
 - PyAMG
 - Numba
 - MeshIO
 - Matplotlib
 - PyVista
 - Topology Optimization Community



## Documentation

- [Scikit-Topt](https://kevin-tofu.github.io/scitopt/)


## ToDo
- Make workflow for multi-load cases efficient
- Set break point from the optimization loop
- Organize documentation
- Add LevelSet