Governing Equation and Weak Form
================================

This section summarizes the physical simulation models used in Scikit-Topt.
The framework currently supports:

- **Linear Elasticity** – deformation and structural mechanics  
- **Steady-State Heat Conduction** – temperature distribution and thermal flow  

Each model is formulated through a standard FEM pipeline:

1. Governing equation (strong form)  
2. Weak formulation  
3. Boundary conditions  
4. Matrix assembly  
5. Solving a linear system  

See the detailed pages below.

.. toctree::
   :maxdepth: 1

   physics-elasticity
   physics-heat
