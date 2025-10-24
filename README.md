[![PyPI version](https://img.shields.io/pypi/v/scikit-topt.svg?cacheSeconds=60)](https://pypi.org/project/scikit-topt/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/957674835.svg)](https://doi.org/10.5281/zenodo.15441499)
[![Python Version](https://img.shields.io/pypi/pyversions/scikit-topt.svg)](https://pypi.org/project/scikit-topt/)
[![PyPI Downloads](https://static.pepy.tech/badge/scikit-topt)](https://pepy.tech/projects/scikit-topt)
![CI](https://github.com/kevin-tofu/scikit-topt/actions/workflows/python-tests.yml/badge.svg)
![CI](https://github.com/kevin-tofu/scikit-topt/actions/workflows/sphinx.yml/badge.svg)


# 🧠 Scikit Topt
**A lightweight, flexible Python library for topology optimization built on top of Scikit Libraries**
- [scipy](https://scipy.org/)
- [scikit-fem](https://github.com/kinnala/scikit-fem)

## Documentation
[Scikit-topt Documentation](https://scikit-topt.readthedocs.io/en/latest/)

## Examples and Fieatures
### Example 1 : Single Load Condition
<p align="center">
  <img src="assets/ex-pull-down-0.gif" alt="Optimization Process Pull-Down-0" width="400" style="margin-right: 20px;">
  <img src="assets/ex-pull-down-1.jpg" alt="Optimization Process Pull-Down-1" width="400">
</p>

### Example 2 : Multiple Load Condition
<p align="center">
  <img src="assets/ex-multi-load-condition.jpg" alt="multi-load-condition" width="400" style="margin-right: 20px;">
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
- able to handle multiple force condition
- High-performance computation using sparse matrices with Scipy and PyAMG  
- has a function to monitor the transition of parameters.

## Usage
### Install Package
```bash
pip install scikit-topt
poetry add scikit-topt
```


### How to define a task

### Load Mesh file from file.
```Python
import skfem
import sktopt

mesh_path = "./data/model.msh"
basis = sktopt.mesh.loader.basis_from_file(
  mesh_path, intorder=3
)
```

#### Create Mesh.
```Python
x_len, y_len, z_len = 1.0, 1.0, 1.0
element_size = 0.1

# import pathlib
# msh_path = "model.msh"
# mesh = skfem.MeshHex.load(pathlib.Path(msh_path))

# define basis
mesh = sktopt.mesh.toy_problem.create_box_hex(
  x_len, y_len, z_len, element_size
)
e = skfem.ElementVector(skfem.ElementHex1())
basis = skfem.Basis(mesh, e, intorder=2)
```

#### Set BCs and Force, and define task
```Python

# Specify Dirichlet Boundary Conditions
dirichlet_in_range = sktopt.mesh.utils.get_points_in_range(
    (0.0, 0.03), (0.0, y_len), (0.0, z_len)
)
force_in_range = sktopt.mesh.utils.get_points_in_range(
  (x_len, x_len),
  (y_len*2/5, y_len*3/5),
  (z_len*2/5, z_len*3/5)
)
boundaries = {
    "dirichlet": dirichlet_in_range,
    "force": force_in_range
}
mesh = mesh.with_boundaries(boundaries)
subdomains = {"design": np.array(range(mesh.nelements))}

mesh = mesh.with_subdomains(subdomains)
e = skfem.ElementVector(skfem.ElementHex1())
basis = skfem.Basis(mesh, e, intorder=2)
dirichlet_dir = "all"
force_dir_type = "u^1"
force_value = 100
# Define it as a task
tsk = sktopt.mesh.task.TaskConfig.from_mesh_tags(
    210e9,
    0.30,
    basis,
    dirichlet_dir,
    force_dir_type,
    force_value
)
```


### Optimize Toy Problem with Python Script

```Python
import sktopt

tsk = sktopt.mesh.toy_problem.toy1()
cfg = sktopt.core.optimizers.OC_Config(
  p=sktopt.tools.SchedulerConfig.step(
    init_value=1.0, target_value=3.0,
    num_steps=3
  ),
  vol_frac=sktopt.tools.SchedulerConfig.constant(
    target_value=0.6
  ),
)

optimizer = sktopt.core.OC_Optimizer(cfg, tsk)

optimizer.parameterize()
optimizer.optimize()
```


### Optimize Toy Problem with command line.
```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3  MKL_NUM_THREADS=3 PYTHONPATH=scikit-topt python ./scikit-topt/sktopt/core/optimizers/logmoc.py --dst_path ./result/toy2_moc 
 --interpolation SIMP  --vol_frac 0.40  --eta 0.8  --record_times 100  --max_iters 300  --mu_p 5  --task_name toy2  --solver_option spsolve  --rho_min 1e-2  --E0 210e9  --E_min 210e5  --design_dirichle
t true
```


## Optimization Algorithm
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
 - Joblib
 - Topology Optimization Community



## 📖 Citation

If you use Scikit Topt in your research or software, please cite it as:

```bibtex
@misc{scikit-topt2025,
  author       = {Watanabe, Kohei},
  title        = {Scikit-Topt: A Python Library for Algorithm Development in Topology Optimization},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15441499},
  url          = {https://doi.org/10.5281/zenodo.15441499},
  note         = {Version 0.2.13}
}
```

## ToDo
- Set break point from the optimization loop
- Add A feature to assign tags to nodes and cells
- Add Level Set
- Add other optimizers
  - Evolutionary Algorithms
  - Phase Field
  - MMA
- Add Multiple BC Conditions
- Add Constraint for Rotation
