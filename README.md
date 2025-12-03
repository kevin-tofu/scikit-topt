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
[Scikit-Topt Documentation](https://scikit-topt.readthedocs.io/en/latest/)

## Examples and Features
### Example 1 : Single Load Condition
<p align="center">
  <img src="https://media.githubusercontent.com/media/kevin-tofu/scikit-topt/master/assets/ex-pull-down-0.gif" alt="Optimization Process Pull-Down-0" width="400" style="margin-right: 20px;">
  <img src="https://media.githubusercontent.com/media/kevin-tofu/scikit-topt/master/assets/ex-pull-down-1.jpg" alt="Optimization Process Pull-Down-1" width="400">
</p>

### Example 2 : Multiple Load Condition
<p align="center">
  <img src="https://media.githubusercontent.com/media/kevin-tofu/scikit-topt/master/assets/ex-multi-load-condition.jpg" alt="multi-load-condition" width="400" style="margin-right: 20px;">
  <img src="https://media.githubusercontent.com/media/kevin-tofu/scikit-topt/master/assets/ex-multi-load-v-50.jpg" alt="multi-load-condition-distribution" width="400">
</p>

### Example 3 : Heat Conduction
<p align="center">
  <img src="https://media.githubusercontent.com/media/kevin-tofu/scikit-topt/master/assets/ex-heat-conduction-cond.jpg" alt="heat-conduction" width="400" style="margin-right: 20px;">
  <img src="https://media.githubusercontent.com/media/kevin-tofu/scikit-topt/master/assets/ex-heat-conduction-0.gif" alt="heat-conduction" width="400" style="margin-right: 20px;">
</p>

### Progress Report
<p align="center">
  <img src="https://media.githubusercontent.com/media/kevin-tofu/scikit-topt/master/assets/ex-progress-report.jpg" alt="multi-load-condition-progress" width="600">
</p>


## Features
 To contribute to the open-source community and education—which I’ve always benefited from—I decided to start this project. 
 
The currently supported features are as follows:
- Coding with Python  
- easy installation with pip/poetry
- Implement FEA on unstructured mesh using scikit-fem
- Structural Analysis / Heat Conduction Analysis
- Topology optimization using the density method and its optimization algorithm
  - Optimality Criteria (OC) Method  
  - (Log-Space) Modified OC Method 
- able to handle multiple force condition
- High-performance computation using sparse matrices with Scipy and PyAMG  
- has a function to monitor the transition of parameters.

## SetUp

You can install **Scikit-Topt** either via **pip** or **Poetry**.

#### Supported Python Versions

Scikit-Topt supports **Python 3.10–3.13**:

- **3.10–3.12** — fully supported and tested  
- **3.13** — core topology optimization works normally,  
  but **VTK-based features** (VTU export & image rendering using PyVista)  
  are temporarily unavailable because VTK/PyVista do not yet provide wheels  
  for Python 3.13.

You can still run the full optimization workflow on Python 3.13;  
only visualization-related features are restricted.


**Choose one of the following methods:**

### Using pip
```bash
pip install scikit-topt
```

### Using poetry
```bash
poetry add scikit-topt
```

### Optional: Enable off-screen rendering

If you want to visualize the optimized density distribution with mesh as an image,
you need to enable off-screen rendering using a virtual display.

On Debian/Ubuntu:
```bash
sudo apt install xvfb libgl1-mesa-glx
```

CentOS / RHL
```bash
sudo yum install xvfb libgl1-mesa-glx
```

## Usage
 See examples in example directory and README.md.
[README for Usage](https://github.com/kevin-tofu/scikit-topt/blob/joss-review/examples/README.md) 
[Examples](https://github.com/kevin-tofu/scikit-topt/tree/joss-review/examples/tutorial) 

## Algorithm for Optimization
Optimization Algorithms and Techniques are briefly summarized here.  
[Optimization Algorithms and Techniques](https://kevin-tofu.github.io/scikit-topt/optimization.html)


## Contributing

We are happy to welcome any contributions to the library.
You can contribute in various ways:

- Reporting bugs, opening pull requests, or starting discussions via [GitHub Issues](https://github.com/kevin-tofu/scikit-topt/issues)
- Writing new [examples](https://github.com/kevin-tofu/scikit-topt/tree/joss-review/examples)
- Improving the [tests](https://github.com/kevin-tofu/scikit-topt/tree/joss-review/tests)
- Enhancing the documentation or code readability [doc](https://scikit-topt.readthedocs.io/en/latest/)

By contributing code to **Scikit-Topt**, you agree to release it under the [Apache 2.0 License](https://github.com/kevin-tofu/scikit-topt/tree/master/LICENSE).


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

If you use Scikit Topt in your research or software, please cite it as:

```bibtex
@misc{Scikit-Topt2025,
  author       = {Watanabe, Kohei},
  title        = {Scikit-Topt: A Python Library for Algorithm Development in Topology Optimization},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15441499},
  url          = {https://doi.org/10.5281/zenodo.15441499},
  note         = {Version 0.3.5}
}
```

## ToDo
- Set break point from the optimization loop
- Add A feature to assign tags to nodes and cells
- Add Level Set
- Add other optimizers
  - Evolutionary Algorithms
  - MMA
- Add Multiple BC Conditions
- Add Unit Test
