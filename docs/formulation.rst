Topology Optimization Algorithm Overview
=================================================

This page provides a high-level explanation of the topology optimization routine implemented in the ``optimize`` method. The algorithm follows a density-based approach, using SIMP or RAMP interpolation, a Helmholtz filter, and Heaviside projection to iteratively update the design variables under compliance minimization.

Interpolation Setup
--------------------

The optimization supports two types of material interpolation:

- **SIMP**: :math:`E(\rho) = E_\text{min} + \rho^p (E_0 - E_\text{min})`
- **RAMP**: :math:`E(\rho) = \frac{\rho}{1 + q(1 - \rho)} (E_0 - E_\text{min}) + E_\text{min}`

Where:

- :math:`\rho` is the density variable.
- :math:`E_0` and :math:`E_\text{min}` are the maximum and minimum Young's moduli.
- :math:`p` and :math:`q` are penalization parameters.

Note: \(E_\text{min}\) must be strictly positive to avoid singular stiffness
matrices. A common choice is \(E_\text{min} \approx 10^{-3} E_0\).

Initialization
~~~~~~~~~~~~~~

- Design variable :math:`\rho` is initialized using the ``vol_frac_init`` or a fixed value depending on the interpolation scheme.
- Restart logic loads a previous design if specified.
- Dirichlet and force boundary elements are set to full material.

Helmholtz Filter
--------------------

- Smooths the density field to ensure minimum length scale.

In topology optimization, the Helmholtz filter is employed to smooth the density field, ensuring a minimum length scale in the design and mitigating numerical instabilities such as checkerboard patterns.
The filter is defined by the following partial differential equation (PDE):

.. math::

   -r^2 \nabla^2 \tilde{\rho} + \tilde{\rho} = \rho

where:

- :math:`\rho` is the original (unfiltered) density field,
- :math:`\tilde{\rho}` is the filtered density field,
- :math:`r` is the filter radius controlling the minimum feature size.

This PDE is subject to homogeneous Neumann boundary conditions:

.. math::

   \frac{\partial \tilde{\rho}}{\partial n} = 0 \quad \text{on} \quad \partial \Omega

where :math:`\partial \Omega` denotes the boundary of the design domain.

By solving this equation, the filtered density field :math:`\tilde{\rho}` is obtained, which is smoother than the original field :math:`\rho`. The filter radius :math:`r` determines the extent of smoothing; larger values of :math:`r` lead to smoother designs with larger minimum feature sizes.

This approach allows for efficient implementation within finite element frameworks and is particularly advantageous in parallel computing environments, as it requires only local mesh information for the finite element discretization of the problem.

Heaviside Projection
----------------------

In topology optimization, the Heaviside projection is employed to transform intermediate density values into values closer to 0 or 1, promoting a clear distinction between void and solid regions in the design. This approach enhances the manufacturability of the optimized structures by reducing gray areas.

The projection is defined by the following smooth approximation of the Heaviside step function:

.. math::

   \tilde{\rho} = \frac{\tanh(\beta \eta) + \tanh(\beta (\rho - \eta))}{\tanh(\beta \eta) + \tanh(\beta (1 - \eta))}

where:

- :math:`\rho` is the filtered density variable,
- :math:`\tilde{\rho}` is the projected density,
- :math:`\beta` is the projection sharpness parameter,
- :math:`\eta` is the threshold parameter determining the transition point.

The parameter :math:`\beta` controls the steepness of the projection function. A higher value of :math:`\beta` results in a steeper transition, effectively pushing the densities towards binary values. Typically, :math:`\beta` is gradually increased during the optimization process to allow the design to evolve smoothly before enforcing a more discrete solution.

The threshold parameter :math:`\eta` sets the midpoint of the projection. A common choice is :math:`\eta = 0.5`, which centers the projection around a density value of 0.5. Adjusting :math:`\eta` can shift the transition point, influencing the material distribution in the optimized design.

In the implementation, the threshold parameter :math:`\eta` corresponds to the variable ``beta_eta``. This variable defines the transition point in the Heaviside projection function, determining the density value at which the projection shifts from favoring void to favoring solid material. There is another \eta parameter in the optimization process, which is used with OC method, please do not take it mistakenly.

By applying the Heaviside projection, the optimization process encourages the formation of distinct solid and void regions, leading to designs that are more practical for manufacturing and exhibit improved structural performance.

Compliance Evaluation
------------------------

 In topology optimization involving multiple load cases, the compliance evaluation process includes the following steps:

- **Displacement Field Computation**: For each load case :math:`i`, compute the global displacement field :math:`\mathbf{u}_i` corresponding to the load vector :math:`\mathbf{f}_i`.

- **Compliance Calculation**: Evaluate the compliance for each load case as:

  .. math::

     C_i = \mathbf{f}_i^T \mathbf{u}_i

- **Total Compliance**: Sum the compliances over all :math:`N_L` load cases to obtain the total compliance:

  .. math::

     C = \sum_{i=1}^{N_L} \mathbf{f}_i^T \mathbf{u}_i

- **Element-wise strain energy**: it is calculated to derive sensitivity:

  .. math::

     \frac{\partial C}{\partial \rho} = -p \rho^{p-1} (E_0 - E_\text{min}) \epsilon^T \mathbf{D} \epsilon

Sensitivity Analysis with Backpropagation
--------------------------------------------

In practice, the sensitivities propagate through filtering and projection as

.. math::
   \frac{\partial C}{\partial \rho}
   = \frac{\partial C}{\partial \tilde{\rho}}
   \frac{\partial \tilde{\rho}}{\partial \rho_{\text{filtered}}}
   \frac{\partial \rho_{\text{filtered}}}{\partial \rho},

which makes the influence of both the Helmholtz filter and the Heaviside
projection explicit in the gradient computation.

In topology optimization, the sensitivity of the objective function (e.g.,
compliance) with respect to the design variable :math:`\rho` is computed by
applying the chain rule through each computational step. This is
conceptually similar to backpropagation in machine learning.

Assume the objective function :math:`C` depends on the displacement field
:math:`\mathbf{u}`, which in turn depends on the projected density
:math:`\tilde{\rho}`, which is computed from the filtered density
:math:`\rho_{\text{filtered}}`, and ultimately from the design variable
:math:`\rho`.

The total derivative is written as:

.. math::
   \frac{\partial C}{\partial \rho}
   =
   \frac{\partial C}{\partial \mathbf{u}}
   \cdot
   \frac{\partial \mathbf{u}}{\partial \tilde{\rho}}
   \cdot
   \frac{\partial \tilde{\rho}}{\partial \rho_{\text{filtered}}}
   \cdot
   \frac{\partial \rho_{\text{filtered}}}{\partial \rho}

Each term corresponds to:

- :math:`\frac{\partial C}{\partial \mathbf{u}}`: the derivative of compliance with respect to displacement, typically equal to the load vector :math:`\mathbf{f}`.
- :math:`\frac{\partial \mathbf{u}}{\partial \tilde{\rho}}`: the derivative of the displacement field with respect to material stiffness, obtained via the adjoint method or differentiating the FEM equilibrium equation.
- :math:`\frac{\partial \tilde{\rho}}{\partial \rho_{\text{filtered}}}`: the derivative of the Heaviside projection function.
- :math:`\frac{\partial \rho_{\text{filtered}}}{\partial \rho}`: the derivative of the Helmholtz filter, which is typically linear and defined via the solution of a diffusion-type PDE.

This chain rule composition allows the gradient of the objective function to be backpropagated from output (compliance) to input (design variable), and is essential for gradient-based optimization algorithms.

As post-processing:

- Gradients are filtered and averaged across load cases.
- Sensitivity filtering is optionally applied post-processing.

Density Update
~~~~~~~~~~~~~~

The densities are updated via a custom update rule using:

- Move limits
- Target volume fraction
- Sensitivity-based scaling
- Percentile clipping for stability

Stopping Criteria
~~~~~~~~~~~~~~~~~

The process repeats until the maximum number of iterations is reached or convergence criteria are satisfied.

.. note::

   This implementation uses a modular framework where filtering, projection, and updating are decoupled for clarity and extensibility.

Next Steps
~~~~~~~~~~

Refer to the individual components below for deeper explanation:

.. - :ref:`interp_schemes`
.. - :ref:`helmholtz_filter`
.. - :ref:`heaviside_proj`
.. - :ref:`density_update`
