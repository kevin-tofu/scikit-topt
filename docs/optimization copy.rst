Optimization Algorithms in Density-Based Topology Optimization
================================================================

Density-based topology optimization relies on continuous design variables to represent material distribution, and various optimization algorithms have been developed to update these densities effectively.

One of the most widely used methods is the **Optimality Criteria (OC) method**, which updates the density based on a closed-form expression derived from the Karush-Kuhn-Tucker (KKT) conditions. The OC method is simple to implement and provides fast convergence in many structural problems. It typically uses a multiplicative update rule that enforces a volume constraint while improving the compliance.

Another approach is the **Modified Optimality Criteria (MOC)** method, which extends the OC framework by incorporating additional strategies for handling constraints, improving stability, or enhancing convergence. For example, some MOC variants integrate projection and filtering directly into the update step or apply move limits and continuation schemes to control intermediate densities more robustly.

In addition to these, gradient-based methods using standard optimization libraries (e.g., MMA: Method of Moving Asymptotes) are also popular in density-based formulations, particularly when dealing with multiple constraints or noncompliance objectives.

Each of these methods offers different trade-offs in terms of implementation complexity, convergence speed, and robustness. The choice of optimization algorithm can significantly affect the quality and performance of the final design.

Optimality Criteria
-----------------------------------

Optimality Criteria Method (OC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Optimality Criteria (OC) method is a widely used algorithm in density-based topology optimization due to its simplicity and effectiveness. 
It provides a closed-form multiplicative update rule for the design variables, derived from the Karush–Kuhn–Tucker (KKT) optimality conditions of the compliance minimization problem.

Assume the goal is to minimize compliance :math:`C(\rho)` subject to a volume constraint:

.. math::

   \min_{\rho} \quad C(\rho) = \mathbf{f}^T \mathbf{u}(\rho) \\
   \text{subject to} \quad \sum_e \rho_e v_e \leq V_{\text{max}}, 
   \quad \rho_{\text{min}} \leq \rho_e \leq 1.

The KKT conditions lead to an element-wise update rule involving the compliance sensitivity and a Lagrange multiplier :math:`\lambda`:

.. math::

   \rho_e^{(t+1)} = \text{clip}\left(
   \rho_e^{(t)} \cdot 
   \left( \frac{-\partial C / \partial \rho_e}{\lambda} \right)^{\eta},\ 
   \rho_{\text{min}},\ 1
   \right).

Here:

- :math:`\partial C / \partial \rho_e` is the sensitivity of compliance with respect to the design variable,
- :math:`\lambda` is a Lagrange multiplier chosen (typically by bisection) to satisfy the volume constraint,
- :math:`\eta` is a numerical damping or scaling factor (commonly :math:`\eta = 1`),
- ``clip`` ensures the updated density stays within prescribed bounds,
- optional move limits may also be enforced to restrict the maximum per-iteration change in :math:`\rho_e`.

The OC method requires only element-wise sensitivities and a scalar multiplier update, 
and therefore avoids reliance on general-purpose gradient-based optimizers such as MMA or SQP.

Advantages
^^^^^^^^^^

- **Simplicity**: Easy to implement and understand.
- **Efficiency**: Fast convergence in practice for compliance minimization.
- **Closed-form updates**: Only sensitivities and a scalar Lagrange multiplier search are required, no external optimization solver.

Disadvantages
^^^^^^^^^^^^^

- **Not general-purpose**: Difficult to extend to multiple constraints or noncompliance objectives.
- **Heuristic parameters**: Requires tuning of damping factors and move limits for stability.
- **Constraint enforcement**: The volume constraint is enforced only up to the accuracy of the multiplier search.
- **Restricted update form**: Limited to multiplicative updates, which makes integration with modern trust-region or line-search strategies nontrivial.

Despite these limitations, the OC method remains one of the most popular approaches for compliance-based topology optimization, 
especially when a robust and lightweight heuristic is sufficient for prototyping or academic demonstration.


Modified Optimality Criteria (MOC) Variants
-------------------------------------------

In density-based topology optimization, the Modified Optimality Criteria (MOC) method can be implemented in several ways. 
One common idea is to reformulate the OC update in **log-space**, which improves numerical stability while preserving its multiplicative structure.

1. Log-space Update Method
~~~~~~~~~~~~~~~~~~~~~~~~~~

This variant applies the OC update to the logarithm of the density rather than the density itself:

.. math::

   \log \rho_e^{(t+1)} 
   = \log \rho_e^{(t)} 
   + \eta \cdot \log\!\left(\frac{-\partial C / \partial \rho_e}{\lambda_v}\right).

After exponentiation, the update is equivalent to the standard multiplicative OC rule:

.. math::

   \rho_e^{(t+1)} 
   = \rho_e^{(t)} 
   \left( \frac{-\partial C / \partial \rho_e}{\lambda_v} \right)^{\eta}.

Here:

- :math:`\rho_e^{(t)}` is the current density,
- :math:`\partial C / \partial \rho_e` is the compliance sensitivity,
- :math:`\lambda_v` is a constraint-related parameter (e.g., a Lagrange multiplier chosen by bisection, or a penalty-based approximation),
- :math:`\eta` is a damping or learning-rate parameter.

This log-space formulation guarantees positivity of :math:`\rho` and improves numerical robustness by working in an additive domain for what is fundamentally a multiplicative update.

**Smoothing of the dual/penalty parameter**

Because the update is multiplicative, instabilities can arise if the constraint-related parameter :math:`\lambda_v` changes too abruptly. 
To mitigate this, a smoothing strategy such as an **Exponential Moving Average (EMA)** can be applied:

.. math::

   \lambda_v^{(t)} 
   = \lambda_\text{decay} \, \lambda_v^{(t-1)} 
   + (1 - \lambda_\text{decay}) \, \hat{\lambda}_v^{(t)},

where :math:`\hat{\lambda}_v^{(t)}` is the current estimate (e.g., from the volume constraint violation) and :math:`\lambda_\text{decay} \in (0, 1]` is a smoothing factor.  
This is not part of the classical OC method but can improve stability in practical implementations.

**Advantages**:

- Guarantees :math:`\rho > 0` automatically.
- Preserves the multiplicative structure of OC while improving numerical stability.
- Easy to implement in a vectorized, in-place fashion.

**Disadvantages**:

- Requires careful tuning of :math:`\eta` and the handling of :math:`\lambda_v`.
- Volume constraint is not enforced exactly unless :math:`\lambda_v` is solved consistently (e.g., via bisection).
- Convergence may be sensitive to filtering and parameter settings.


2. Linear-Space Update Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative to the multiplicative OC update is to use a **linear (additive) update** rule in the physical density domain. 
Here, the design variables are updated by adding an explicit increment :math:`\Delta \rho` to the current density:

.. math::

   \rho_e^{(t+1)} = \rho_e^{(t)} + \Delta \rho_e,

with the increment defined as

.. math::

   \Delta \rho_e = -\eta \, \left( \frac{\partial C}{\partial \rho_e} + \lambda_v \frac{\partial g}{\partial \rho_e} \right),

where:

- :math:`\partial C / \partial \rho_e` is the compliance sensitivity,
- :math:`g(\rho)` is the volume constraint function,
- :math:`\lambda_v` is a constraint-related parameter (dual variable in a Lagrangian formulation, or a penalty weight in a penalty-based approximation),
- :math:`\eta` is a step size (learning rate).

In practice, the update increment :math:`\Delta \rho_e` is often **clipped** to enforce move limits or bound constraints:

.. math::

   \rho_e^{(t+1)} = \text{clip}\!\left(\rho_e^{(t)} + \Delta \rho_e,\; \rho_{\min},\; 1\right).

This formulation resembles a gradient-descent step and is structurally simpler than multiplicative OC, 
which makes it more flexible for integration with additional constraint-handling or stabilization techniques.

**Advantages**:

- Direct control over the update magnitude through :math:`\eta`.
- Easier to integrate with projections, filters, or move-limit strategies.
- Conceptually simple and convenient for algorithmic experimentation.

**Disadvantages**:

- Positivity of :math:`\rho` is not guaranteed unless explicitly enforced by clipping.
- Volume constraints are only approximately satisfied unless post-processing or multiplier updates are applied.
- Sensitive to parameter tuning; damping or stabilization is usually required for robust performance.


Log-space Lagrangian Method
-----------------------------------

Log-space Lagrangian Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method is a variant of the Modified Optimality Criteria (MOC) approach that performs density updates in **log-space** rather than directly in the physical domain. The goal is to improve numerical stability and ensure that the updated density field remains strictly positive.

Instead of updating the density :math:`\rho` directly, the update is applied to its logarithm :math:`\log \rho`. This transforms the multiplicative update behavior into an additive one in log-space.

The method computes the Lagrangian gradient :math:`dL` as the sum of the compliance sensitivity :math:`dC` and the derivative of the volume penalty term :math:`\lambda_v`. The update rule is:

.. math::

   \log \rho^{(t+1)} = \log \rho^{(t)} - \eta \cdot (dC + \lambda_v)

The density is then recovered by exponentiation:

.. math::

   \rho^{(t+1)} = \exp\left( \log \rho^{(t)} - \eta \cdot (dC + \lambda_v) \right)
               = \rho^{(t)} \cdot \exp\left( -\eta \cdot (dC + \lambda_v) \right)

Here:

- :math:`\rho^{(t)}` is the current density at iteration :math:`t`,
- :math:`dC` is the derivative of compliance with respect to density,
- :math:`\lambda_v` is the derivative of the volume constraint penalty,
- :math:`\eta` is a scalar step size (analogous to a learning rate).

Optional clipping is applied in log-space to limit excessive updates and preserve stability:

.. math::

   \log \rho^{(t+1)} = \text{clip}\left( \log \rho^{(t+1)},\ \log \rho_{\min},\ \log \rho_{\max} \right)

Finally, move limits can also be enforced using:

.. math::

   \rho^{(t+1)} = \text{clip}\left( \rho^{(t+1)},\ \rho^{(t)} - \Delta \rho_{\max},\ \rho^{(t)} + \Delta \rho_{\max} \right)

**Advantages**:

- Naturally ensures :math:`\rho > 0` without additional constraints.
- Suitable for in-place and vectorized computation in large-scale problems.
- Converts multiplicative effects into additive updates, improving numerical robustness.

**Disadvantages**:

- Sensitive to step size :math:`\eta` and penalty weight :math:`\lambda_v`.
- Volume constraints are only enforced implicitly via penalty.
- Requires careful initialization and parameter tuning to ensure convergence.
