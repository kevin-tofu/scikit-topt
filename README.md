![CI](https://github.com/kevin-tofu/scikit-topt/actions/workflows/python-tests.yml/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/scitopt.svg?cacheSeconds=60)](https://pypi.org/project/scitopt/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# 🧠 Scikit Topt
**A lightweight, flexible Python library for topology optimization built on top of [scikit-fem](https://github.com/kinnala/scikit-fem).**

<p align="center">
  <img src="assets/ex-pull-down-0.gif" alt="Optimization Process Pull-Down-0" width="500">
  <img src="assets/ex-pull-down-1.jpg" alt="Optimization Process Pull-Down-0" width="500">
</p>


## Features
 To contribute to the open-source community and education—which I’ve always benefited from—I decided to start this project. 
 
  The currently supported features are as follows:
- Coding with Python  
- Implement FEA on structured mesh using scikit-fem
- Topology optimization using the density method and its optimization algorithm
  - Optimality Criteria (OC) Method  
  - Modified OC Method
  - Lagrange Method
- Multiple objective functions (forces)  
- High-performance computation using sparse matrices with Scipy and Numba  
- easy installation with pip/poetry



## ToDo
- density interpolation
- coarse to fine optimization
- stabilize
- set break point from the optimization loop
- Add LevelSet

### Install Package
```bash
pip install scitopt
poetry add scitopt
```

### Optimize Toy Problem with command line.
```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3  MKL_NUM_THREADS=3 PYTHONPATH=./ python ./scitopt/core/optimizer/kkt.py \
 --dst_path ./result/test2_kkt1 \
 --interpolation SIMP \
 --p_init 1.2 \
 --p 3.0 \
 --p_step 3 \
 --filter_radius_init 1.50 \
 --filter_radius 0.05 \
 --filter_radius_step 3 \
 --move_limit_init 0.40 \
 --move_limit 0.10 \
 --move_limit_step 50 \
 --vol_frac_init 0.80 \
 --vol_frac 0.40 \
 --vol_frac_step 3 \
 --beta_init 0.1 \
 --beta 4.0 \
 --beta_step -5 \
 --percentile_init 70 \
 --percentile 90 \
 --percentile_step -5 \
 --eta 0.3 \
 --record_times 120 \
 --max_iters 300 \
 --lambda_v 0.01 \
 --lambda_decay  0.8 \
 --lambda_lower -100.0 \
 --lambda_upper 100.0 \
 --mu_p 2.5 \
 --export_img true \
 --task plate-0.2.msh \
 --design_dirichlet true
```

### Optimize Toy Problem with Python Script

```Python
import scitopt

tsk = scitopt.mesh.toy_problem.toy1()
cfg = scitopt.core.KKT_Config()

optimizer = scitopt.core.KKT_Optimizer(cfg, tsk)

optimizer.parameterize()
optimizer.optimize()
```



## Optiization Algorithm
### Density Method
#### Optimality Criteria (OC) Method

The **OC method** is a widely used algorithm for compliance minimization problems in structural topology optimization. It updates the material distribution (`density field`) based on a set of local update rules derived from optimality conditions.

**Key characteristics:**
- Simple and efficient to implement.
- Iteratively updates densities using sensitivity information (e.g., compliance derivatives).
- Often includes move limits to stabilize convergence.

**Update rule (simplified):**
```math
\rho_i^{(new)} = \text{clip}\left(\rho_i \cdot \left(-\frac{\partial C}{\partial \rho_i} / \lambda \right)^{\eta}, \rho_{min}, \rho_{max} \right)
```
where:
- ρ_i: density of element i
- dC/dρ_i: compliance sensitivity
- λ: Lagrange multiplier (to satisfy volume constraint)
- η: damping factor

---

#### Modified OC (MOC) Method

The **Modified OC method (MOC)** extends the classic OC method by introducing enhancements such as:
- **Log-domain updates** to improve numerical stability,
- **Dynamic lambda adjustment** to better handle volume constraints,
- **Stress constraints** or **connectivity penalties** (optional).

**Advantages of MOC:**
- Improved convergence in difficult optimization problems.
- Better control over numerical instability (e.g., checkerboard patterns).
- More flexibility to incorporate complex constraints.

---

Both methods are particularly suited for density-based approaches (e.g., SIMP), and can be combined with filters (e.g., sensitivity or density filters) to control minimum feature size and avoid numerical issues.

---

## Techinical Components
### Material Interpolation: SIMP and RAMP
In density-based topology optimization, the material stiffness is interpolated as a function of the element density.

#### SIMP (Solid Isotropic Material with Penalization)
SIMP is the most commonly used interpolation scheme:

```math
E(ρ) = ρ^p * E₀
```

- ρ: element density (range: 0 to 1)
- p: penalization factor (typically p ≥ 3)
- E0: Young’s modulus of solid material



This method penalizes intermediate densities and encourages a 0–1 (black-and-white) design.

####  RAMP (Rational Approximation of Material Properties)

RAMP is another interpolation scheme used to reduce numerical instabilities like checkerboarding:

```math
E(ρ) = E₀ * ρ / (1 + q * (1 - ρ))
```

- q: penalization parameter (higher q gives stronger 0–1 behavior)


RAMP can sometimes provide smoother convergence than SIMP.

---

### Heaviside Projection

Heaviside projection is used to **sharpen the boundaries** between solid and void regions after filtering:

```math
ρ̃ = (tanh(β * η) + tanh(β * (ρ - η))) / (tanh(β * η) + tanh(β * (1 - η)))
```

- ρ: filtered density
- ρ̃: projected density
- β: steepness parameter (higher = sharper transitions)
- η: threshold level (usually 0.5)

As beta → ∞, the projection approaches a binary function.

---

### Helmholtz Filter (Density Smoothing)

The **Helmholtz filter** smooths the density field to prevent checkerboard patterns and enforce a minimum feature size.

It solves the PDE:

```math
(-r² ∇² + 1) ρ̃ = ρ
```

- ρ: raw density field  
- ρ̃: filtered density  
- r: filter radius (controls the minimum length scale)

This is often implemented via solving a sparse linear system using finite elements.

**Benefits:**
- Enforces minimum feature size
- Suppresses numerical instabilities
- Improves manufacturability of the design

---



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


## 📖 Citation

If you use this repository in your work, please cite it as follows:  

If you use Scikit Topt in your research or software, please cite it as:

```bibtex
@misc{scikit-topt2025,
  author       = {Kohei Watanabe},
  title        = {Scikit Topt: A Python library for topology optimization using scikit-fem},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/kevin-tofu/scikit-topt}},
  note         = {Accessed: 2025-04-24}
}
```