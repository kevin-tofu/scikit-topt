The Weak Form of Linear Elasticity
====================================

This document provides an overview of the physics implemented in
**Scikit-Topt**, focusing on linear elasticity and its formulation in the
finite element method (FEM).  
We explain the governing equations, weak formulation, boundary conditions
(Neumann and Dirichlet), and the resulting linear system solved inside
the optimizer.

Linear Elasticity
-----------------

In linear elasticity, the displacement field
:math:`\mathbf{u} : \Omega \rightarrow \mathbb{R}^d`
describes how each point in the domain :math:`\Omega` moves under
external forces.

The stress–strain relationship follows Hooke’s law:

.. math::

   \boldsymbol{\sigma} = \mathbb{C} : \boldsymbol{\varepsilon},

where

- :math:`\boldsymbol{\sigma}` = Cauchy stress tensor  
- :math:`\boldsymbol{\varepsilon}` = infinitesimal strain tensor  
- :math:`\mathbb{C}` = 4th-order elasticity tensor  
  (depends on Young’s modulus :math:`E` and Poisson ratio :math:`\nu`).

The strain tensor is defined as:

.. math::

   \boldsymbol{\varepsilon}(\mathbf{u})
   = \frac{1}{2}(\nabla \mathbf{u} + \nabla\mathbf{u}^T).

Governing Equation (Strong Form)
--------------------------------

Mechanical equilibrium in absence of body forces is written as:

.. math::

   -\nabla \cdot \boldsymbol{\sigma} = \mathbf{0}
   \quad \text{in } \Omega.

Boundary conditions are applied on the domain boundary :math:`\partial\Omega`:

- **Dirichlet BC (fixed boundary)**:  
  :math:`\mathbf{u} = \bar{\mathbf{u}}` on :math:`\Gamma_D`

- **Neumann BC (traction/force)**:  
  :math:`\boldsymbol{\sigma}\mathbf{n} = \mathbf{t}` on :math:`\Gamma_N`

Here :math:`\mathbf{n}` is the outward normal vector.

Weak Formulation
----------------

We multiply the equilibrium equation by a test function
:math:`\mathbf{v}` and integrate over the domain:

.. math::

   \int_\Omega \boldsymbol{\sigma}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v}) \, d\Omega
   = \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, d\Gamma.

This is the *weak form* of linear elasticity.

Substituting Hooke’s law gives:

.. math::

   \int_\Omega (\mathbb{C}\boldsymbol{\varepsilon}(\mathbf{u}))
   : \boldsymbol{\varepsilon}(\mathbf{v}) \, d\Omega
   = \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, d\Gamma.

Finite Element Discretization
-----------------------------

We approximate the displacement field as:

.. math::

   \mathbf{u}_h(\mathbf{x})
   = \sum_{i=1}^{N} u_i \, \phi_i(\mathbf{x}),

where :math:`\phi_i` are FEM shape functions and :math:`u_i` the unknown nodal
degrees of freedom.

Inserting this into the weak form yields the discrete system:

.. math::

   \mathbf{K} \mathbf{u} = \mathbf{f},

where:

- :math:`\mathbf{K}` is the **stiffness matrix**

  .. math::

     \mathbf{K}_{ij}
     = \int_\Omega
       \boldsymbol{\varepsilon}(\phi_j)^T \,
       \mathbb{C} \,
       \boldsymbol{\varepsilon}(\phi_i)
       \, d\Omega.

- :math:`\mathbf{f}` is the **load vector**

  .. math::

     \mathbf{f}_i
     = \int_{\Gamma_N} \mathbf{t} \cdot \phi_i \, d\Gamma.

Boundary Conditions
-------------------

Dirichlet BC (fixed boundaries)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enforced by modifying the stiffness matrix system:

- Rows corresponding to fixed DOFs are zeroed
- Diagonal entries are set to 1
- Load vector entries for those DOFs are set to the prescribed value

Neumann BC (surface forces)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Traction loads add contributions to :math:`\mathbf{f}`:

Solving the Linear System
-------------------------

After applying boundary conditions, we solve:

.. math::

   \mathbf{K}_e \mathbf{u} = \mathbf{f}_e.

In Scikit-Topt:

- Default: sparse direct solver (``scipy.spsolve`` or ``splu``)
- Optional: iterative solvers or PyAMG

Typical code:

.. code-block:: python

   K_e, f_e = skfem.enforce(K, f, D=dirichlet_dofs)
   u = scipy.sparse.linalg.spsolve(K_e, f_e)
