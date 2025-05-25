Tutorial Page
===============


Optimization
-----------------

Below is a minimal working example that performs a topology optimization.
This will run a compliance minimization with OC method.

Optimizer Configuration, and Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import scitopt

   cfg = scitopt.core.optimizers.OC_Config()

   optimizer = scitopt.core.OC_Optimizer(cfg, tsk)
   optimizer.parameterize()
   optimizer.optimize()

But before running the optimization, we need to set up the task configuration and the design variables.

Task Definition
-----------------

Shape modeling and its basis function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import skfem
   import scitopt

   x_len, y_len, z_len = 1.0, 1.0, 1.0
    element_size = 0.1
    e = skfem.ElementVector(skfem.ElementHex1())
    mesh = scitopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, element_size
    )
    basis = skfem.Basis(mesh, e, intorder=3)


Task Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dirichlet_points = scitopt.mesh.utils.get_point_indices_in_range(
        basis, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_nodes = basis.get_dofs(nodes=dirichlet_points).all()

    # Specify Force Vector
    F_points = scitopt.mesh.utils.get_point_indices_in_range(
        basis,
        (x_len, x_len),
        (y_len*2/5, y_len*3/5),
        (z_len*2/5, z_len*3/5)
    )
    F_nodes = basis.get_dofs(nodes=F_points).nodal["u^1"]
    F = 100

    # Specify Design Field
    design_elements = scitopt.mesh.utils.get_elements_in_box(
        mesh,
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )
    mytask = scitopt.mesh.task.TaskConfig.from_defaults(
        210e9,
        0.30,
        basis,
        dirichlet_points,
        dirichlet_nodes,
        F_points,
        F_nodes,
        F,
        design_elements
    )

Results and Visualization
-----------------------------

Results and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The results of the optimization are stored in the directory specified by cfg.dst_path.
For example, it contains visualizations of the density distribution, as well as graphs showing the evolution of various parameters during the optimization process, such as the density field, volume fraction, and sensitivity values.

.. image:: _static/ex-progress-report.jpg
   :alt: Alternative text
   :width: 400px
   :align: center

.. image:: _static/ex-pull-down-1.gif
   :alt: Alternative text
   :width: 400px
   :align: center
