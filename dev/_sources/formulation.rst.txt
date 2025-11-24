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

Sensitivity Propagation Through Filtering and Projection
--------------------------------------------------------

In density-based topology optimization, the compliance sensitivity computed
from FEM does not act directly on the design density. Instead, the design
variable passes through two transformations:

.. math::
   \rho
   \;\xrightarrow{\text{filter}}\;
   \tilde{\rho}
   \;\xrightarrow{\text{projection}}\;
   \hat{\rho}.

The total derivative uses the chain rule:

.. math::
   \frac{\partial C}{\partial \rho}
   =
   \frac{\partial C}{\partial \hat{\rho}}
   \cdot
   \frac{\partial \hat{\rho}}{\partial \tilde{\rho}}
   \cdot
   \frac{\partial \tilde{\rho}}{\partial \rho}.

Meaning of each term:

* :math:`\partial C / \partial \hat{\rho}`  
  Raw compliance sensitivity obtained from FEM (already aggregated over load cases).

* :math:`\partial \hat{\rho} / \partial \tilde{\rho}`  
  Derivative of the Heaviside-type projection used for black–white refinement.

* :math:`\partial \tilde{\rho} / \partial \rho`  
  Adjoint of the spatial or Helmholtz density filter.  
  For the linear Helmholtz filter, this corresponds to applying ``F^T`` via
  ``filter.gradient()``.

Thus the final sensitivity in Scikit-Topt is computed as:

.. math::
   \frac{\partial C}{\partial \rho}
   = F^T \left(
       \frac{\partial \hat{\rho}}{\partial \tilde{\rho}}
       \circ
       \frac{\partial C}{\partial \hat{\rho}}
     \right)

where :math:`\circ` denotes element-wise multiplication.

This “chain rule through filtering and projection’’ ensures that the update
step (OC or MOC) is consistent with the filtered and projected densities used
in the state equation

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


Convergence Criteria
--------------------

In addition to the fixed iteration limit (``max_iters``), Scikit-Topt supports
two standard convergence checks widely used in OC/MOC methods:

**Maximum density change**

.. math::

   \max_e \left| \rho_e^{(t+1)} - \rho_e^{(t)} \right|
   < \varepsilon_\rho.

**KKT residual**

.. math::

   \left\| \nabla_\rho L(\rho, \lambda) \right\|
   < \varepsilon_{\mathrm{KKT}}.


If :class:`~sktopt.core.optimizers.DensityMethodConfig.check_convergence` is set to ``True``, convergence is declared only when both of the following conditions are satisfied:

Next Steps
~~~~~~~~~~

Refer to the individual components below for deeper explanation:

.. - :ref:`interp_schemes`
.. - :ref:`helmholtz_filter`
.. - :ref:`heaviside_proj`
.. - :ref:`density_update`
