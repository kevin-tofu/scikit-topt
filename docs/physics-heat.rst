The Weak Form of Heat Conduction
===================================

This section describes the physics of *steady-state heat conduction* used in
Scikit-Topt.  
We derive the governing equation, weak formulation, and treatment of
Dirichlet, Neumann, and Robin boundary conditions in the finite element
method (FEM).

Governing Equation (Strong Form)
--------------------------------

Let :math:`T : \Omega \rightarrow \mathbb{R}` be the temperature field.  
The steady-state heat conduction equation without internal heat source is:

.. math::

   -\nabla \cdot (k(\mathbf{x}) \nabla T) = 0
   \quad \text{in } \Omega,

where

- :math:`k(\mathbf{x})` is the thermal conductivity  
- :math:`T` is the unknown temperature

Boundary Conditions
-------------------

We consider three types of thermal boundary conditions:

1. **Dirichlet (Temperature BC)**  
   Prescribed temperature:

   .. math::

      T = T_0 \quad \text{on } \Gamma_D.

2. **Neumann (Heat Flux BC)**  
   Prescribed heat flux :math:`q`:

   .. math::

      -k \nabla T \cdot \mathbf{n} = q
      \quad \text{on } \Gamma_N.

3. **Robin (Convective BC)**  
   Newton’s cooling law:

   .. math::

      -k \nabla T \cdot \mathbf{n}
      = h(T - T_\infty)
      \quad \text{on } \Gamma_R,

   where

   - :math:`h` : heat transfer coefficient  
   - :math:`T_\infty` : ambient temperature

   This boundary condition models cooling by air or fluid.

Weak Formulation
----------------

Let :math:`v` be a test function.  
Multiply the strong form by :math:`v` and integrate over :math:`\Omega`:

.. math::

   \int_\Omega -\nabla \cdot (k \nabla T) \, v \, d\Omega = 0.

Using the divergence theorem:

.. math::

   \int_\Omega k \nabla T \cdot \nabla v \, d\Omega
   = \int_{\partial\Omega} (k \nabla T \cdot \mathbf{n}) v \, d\Gamma.

We now insert boundary conditions.

Dirichlet BC
------------
For :math:`\Gamma_D`, the temperature is prescribed.  
As usual in FEM, this is enforced through modification of the linear system
after assembling the matrix (see below).

Neumann BC
----------

On :math:`\Gamma_N`:

.. math::

   -k \nabla T \cdot \mathbf{n} = q.

Substituting into the boundary integral:

.. math::

   \int_{\Gamma_N} q v \, d\Gamma.

Robin BC
--------

Robin BC provides an additional contribution:

.. math::

   -k \nabla T \cdot \mathbf{n}
   = h(T - T_\infty)
   \quad \Rightarrow \quad
   k \nabla T \cdot \mathbf{n} = -h(T - T_\infty).

Plug into the boundary integral:

.. math::

   \int_{\Gamma_R} h(T - T_\infty) v \, d\Gamma.

This term splits into:

.. math::

   \int_{\Gamma_R} h T v \, d\Gamma
   - \int_{\Gamma_R} h T_\infty v \, d\Gamma.

As a result:

- The first part contributes to the **matrix**  
  (Robin = “boundary stiffness”)

  .. math::

     K_{ij}^{(R)} = \int_{\Gamma_R} h \phi_i \phi_j \, d\Gamma.

- The second part contributes to the **load vector**

  .. math::

     f_i^{(R)} = \int_{\Gamma_R} h T_\infty \phi_i \, d\Gamma.

FEM Discretization
------------------

Expand the temperature field as:

.. math::

   T_h(\mathbf{x}) = \sum_{j=1}^N T_j \phi_j(\mathbf{x}).

The weak form becomes:

.. math::

   \mathbf{K}\mathbf{T} = \mathbf{f},

where

- **stiffness matrix**

  .. math::

     K_{ij} = \int_\Omega k \nabla\phi_j \cdot \nabla\phi_i \, d\Omega
              + \int_{\Gamma_R} h \phi_j \phi_i \, d\Gamma.

- **load vector**

  .. math::

     f_i = \int_{\Gamma_N} q \phi_i \, d\Gamma
           + \int_{\Gamma_R} h T_\infty \phi_i \, d\Gamma.

Dirichlet Boundary Enforcement
------------------------------

Dirichlet temperature BCs (prescribed :math:`T=T_0`) are applied by modifying
the linear system:

- Zero out rows of fixed DOFs  
- Set diag = 1  
- Set load entry = prescribed value

In Scikit-FEM (as used by Scikit-Topt):


Solving the Linear System
-------------------------

The final assembled system is:

.. math::

   \mathbf{K}_e \mathbf{T} = \mathbf{f}_e.

In Scikit-Topt, the solver is typically:

.. code-block:: python

   T = scipy.sparse.linalg.spsolve(K_e, f_e)

Summary
-------

The heat conduction solver in Scikit-Topt follows the standard FEM procedure:

1. Governing equation  
2. Weak form  
3. Boundary conditions  
4. Robin BC → both stiffness + load contribution  
5. Assemble global matrices  
6. Enforce Dirichlet BC via ``skfem.enforce``  
7. Solve linear system :math:`K T = f`

This framework is used consistently for thermal topology optimization,
including SIMP-based and adjoint sensitivity analyses.
