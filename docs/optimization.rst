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

The Optimality Criteria (OC) method is a widely used algorithm in density-based topology optimization due to its simplicity and effectiveness. It provides a closed-form update rule for the design variables based on the optimality conditions derived from the Lagrangian of the problem.

Assume the goal is to minimize compliance :math:`C(\rho)` subject to a volume constraint:

.. math::

   \min_{\rho} \quad C(\rho) = \mathbf{f}^T \mathbf{u}(\rho) \\
   \text{subject to} \quad \sum_e \rho_e v_e \leq V_{\text{max}}, \quad \rho_{\text{min}} \leq \rho_e \leq 1

The KKT conditions yield an optimality criterion involving the derivative of the objective function and a Lagrange multiplier :math:`\lambda`. Based on this, the OC method defines the update rule for each element :math:`e` as:

.. math::

   \rho_e^{(t+1)} = \text{clip}\left(
   \rho_e^{(t)} \cdot \left( \frac{-\partial C / \partial \rho_e}{\lambda} \right)^{\eta},\ 
   \rho_{\text{min}},\ 1
   \right)

where:

- :math:`\partial C / \partial \rho_e` is the sensitivity of compliance with respect to the design variable,
- :math:`\lambda` is chosen to satisfy the volume constraint (e.g., by bisection),
- :math:`\eta` is a numerical damping or scaling factor (commonly :math:`\eta = 1`),
- `clip` ensures the updated value stays within allowable bounds.

The OC update rule is applied iteratively, and move limits may also be enforced to restrict the change in :math:`\rho_e` between iterations.

Advantages
^^^^^^^^^^

- **Simplicity**: Easy to implement and understand.
- **Efficiency**: Fast convergence in practice for compliance minimization.
- **No gradient solver needed**: Only objective sensitivities are required, not the full gradient descent machinery.

Disadvantages
^^^^^^^^^^^^^

- **Not general-purpose**: Hard to extend to problems with multiple constraints or noncompliance objectives.
- **Heuristic parameters**: Requires tuning of parameters like damping factor, move limits.
- **Volume constraint enforcement**: Inexact unless a Lagrange multiplier search is properly implemented.
- **Limited to multiplicative updates**: Cannot easily incorporate modern optimization strategies (e.g., trust regions or line search).

Despite its limitations, the OC method remains popular for compliance-based problems, especially when a quick and robust heuristic method is desired for early prototyping or academic demonstration.


Modified Optimality Criteria (MOC) Variants
-------------------------------------------

In density-based topology optimization, the Modified Optimality Criteria (MOC) method can be implemented in several ways. I implemented 2 variants which are log-space update and direct additive update. These differ in how they incorporate sensitivity information and handle volume constraints.

1. Log-space Update Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method modifies the OC update by applying it in **log-space**. Instead of directly updating the density :math:`\rho`, it performs the update on :math:`\log \rho`, ensuring better numerical stability and multiplicative behavior.

It computes the Lagrangian gradient :math:`dL` as the sum of the compliance sensitivity :math:`dC` and a volume penalty term :math:`\lambda_v`, and applies the update in log-space:

.. math::

   \log \rho^{(t+1)} = \log \rho^{(t)} - \eta \cdot (dC + \lambda_v)

Then, the updated density is recovered by exponentiation:

.. math::

   \rho^{(t+1)} = \exp\left( \log \rho^{(t)} - \eta \cdot (dC + \lambda_v) \right)
               = \rho^{(t)} \cdot \exp\left( -\eta \cdot (dC + \lambda_v) \right)

Here:

- :math:`\rho^{(t)}` is the current density field,
- :math:`dC` is the sensitivity of the compliance with respect to density,
- :math:`\lambda_v` is the derivative of the volume penalty,
- :math:`\eta` is a step size or learning rate.

This approach improves numerical stability and ensures that the density remains positive throughout the optimization process.

**Advantages**:

- Efficient in-place update suitable for large-scale problems.
- Ensures positivity of design variables automatically.
- Straightforward to implement using standard vector operations.

**Disadvantages**:

- Requires careful tuning of :math:`\eta` and :math:`\lambda_v`.
- Does not enforce volume constraints exactly—relies on penalty balancing.
- Convergence behavior may vary depending on the problem and filter.

2. Linear-Space Update Method (Not Implemented yet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This variant formulates the update as an explicit increment :math:`\Delta \rho` added to the current density. The update is defined as:

.. math::

   \rho^{(t+1)} = \rho^{(t)} + \Delta \rho

where:

.. math::

   \Delta \rho = -\eta \cdot (dC + \lambda_v)

This form is structurally simpler and better suited for integration with additional constraint handling mechanisms. The update increment :math:`\Delta \rho` may be further clipped to enforce move limits or bound constraints.

**Advantages**:

- Direct control over the update magnitude.
- Easier to incorporate projection, filtering, or move limits.
- Simple to interpret and modify in algorithmic experiments.

**Disadvantages**:

- Can violate positivity if not carefully bounded.
- Volume constraint is only approximately satisfied unless post-processing is added.
- Requires stabilization strategies (e.g., clipping or damping) for robust performance.

Summary
~~~~~~~

Both update strategies aim to descend along the objective gradient while respecting volume constraints and maintaining stability. The choice depends on implementation goals:

- Use the log-space update when prioritizing positivity and multiplicative structure.
- Use the additive update when flexibility, constraint control, or custom damping is desired.

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
