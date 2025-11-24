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


### Bisection for the Volume Constraint

The multiplier :math:`\lambda` enforcing the volume constraint is found by scalar bisection.
This procedure is identical for both OC and MOC updates, as the volume constraint enters only through the scalar multiplier.
Each trial :math:`\lambda` yields updated densities via the OC/MOC rule; the volume is checked and the bounds adjusted until the target is met.
The process is inexpensive, as it requires only vectorized density updates and no additional FEA solves.


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

Log-space Update Method
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


## Practical Notes for OC and MOC Implementations

Both the classical OC method and its modified variants (MOC) rely on a few practical numerical settings that significantly affect robustness and convergence.
The following guidelines summarize commonly used choices across the literature.

### Parameter Guidelines

Several heuristic parameters appear in OC/MOC-style multiplicative updates:

* **Move limits ** (move_limit, controls per-iteration change): typically **0.2–0.5**.
  Smaller limits improve stability; larger limits accelerate convergence.

* **Exponent parameter :math:`\eta`** (eta): often **1.0**, with **0.5–1.0** used when sensitivities are noisy.

* **Minimum density :math:`\rho_{\min}`** (rho_min): usually **10⁻³–10⁻⁴** to avoid singular stiffness matrices.

* **Volume fraction :math:`v_{\text{frac}}`** (vol_frac): typically **0.3–0.8** depending on the desired material usage.
  Very low targets may cause discontinuities; very high targets reduce optimization benefit.

* **SIMP penalization :math:`p`**: commonly continued from **1.0** to a final value of **3–4** over a few steps.
  Higher values promote a sharper black–white design but increase non-convexity.

* **Projection sharpness :math:`\beta`** (beta, Heaviside): usually increased from **1.0** to **2–8** through continuation.
  Larger final values sharpen the design but require stronger filtering to maintain stability.

* **Filter radius**  (filter_radius, spatial/Helmholtz filters): typically **1.5–3.0** times the local element size,
  corresponding to **0.01–0.1** for normalized domains.
  Larger radii suppress numerical artifacts but remove small features.

* **Weak material coefficient :math:`E_{\min}`** (E_min, void stiffness): usually **10^{-3}–10^{-4}**.
  Too small values lead to ill-conditioned stiffness matrices; too large values blur the void region.


These ranges are not strict but offer reliable starting points for most compliance-based problems.

### Recommended Range for the Multiplier Bounds

The Lagrange multiplier :math:`\lambda` used in OC/MOC updates must be **strictly positive**, since the multiplicative update
.. math::
\rho_e^{(t+1)} = \rho_e^{(t)}\left(\frac{-\partial C/\partial \rho_e}{\lambda}\right)^\eta
requires :math:`\lambda > 0` to keep the update well-defined and to preserve the monotonic relationship between
:math:`\lambda` and the resulting volume.

A wide positive bracket such as

```
lambda_lower = 1e−7
lambda_upper = 1e+7
```

is sufficient for most problems. This logarithmic range covers all practical sensitivity scales,
and because the OC/MOC update is monotonic in :math:`\lambda`,
using a wide positive interval does not increase computational cost.

For implementation details and the correspondence with actual parameter
definitions, see the :ref:`core module <core_api>`.