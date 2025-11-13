Tutorial Page
===============


Optimization
-----------------

Below is a minimal working example that performs a topology optimization.
This will run a compliance minimization with OC method.

Optimizer Configuration, and Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import sktopt

   cfg = sktopt.core.optimizers.LogMOC_Config(
        vol_frac=vol_frac=sktopt.tools.SchedulerConfig.constant(
            target_value=0.6
        ),
        max_iters=40,
        record_times=40,
        export_img=True
    )
    optimizer = sktopt.core.LogMOC_Optimizer(cfg, mytask)
    optimizer.parameterize()
    optimizer.optimize()

But before running the optimization, we need to set up the task configuration and the design variables.

Task Definition
-----------------

Shape modeling and its basis function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import skfem
    import sktopt

    x_len = 8.0
    y_len = 8.0
    z_len = 1.0
    mesh_size = 0.2

    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, mesh_size
    )
    e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=2)


Load Basis from Model File 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import skfem
    import sktopt

    mesh_path = "./data/model.msh"
    basis = sktopt.mesh.loader.basis_from_file(mesh_path, intorder=3)


Task Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dirichlet_in_range = sktopt.mesh.utils.get_points_in_range(
        (0.0, 0.05), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_dir = "all"

    eps = mesh_size
    force_in_range_0 = sktopt.mesh.utils.get_points_in_range(
        (x_len, x_len), (y_len-eps, y_len), (0, z_len)
    )
    force_in_range_1 = sktopt.mesh.utils.get_points_in_range(
        (x_len, x_len), (0, eps), (0, z_len)
    )
    force_dir_type = ["u^2", "u^2"]
    force_value = [-100, 100]

    boundaries = {
        "dirichlet": dirichlet_in_range,
        "force_0": force_in_range_0,
        "force_1": force_in_range_1
    }
    mesh = mesh.with_boundaries(boundaries)
    subdomains = {"design": np.array(range(mesh.nelements))}
    mesh = mesh.with_subdomains(subdomains)

    e = skfem.ElementVector(skfem.ElementHex1())
    basis = skfem.Basis(mesh, e, intorder=2)
    E0 = 210e3
    mytask = task.LinearElastisicity.from_mesh_tags(
        basis,
        "all",
        force_dir_type,
        force_value,
        E0,
        0.30,
    )


Results and Visualization
-----------------------------

Results and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The results of the optimization are stored in the directory specified by cfg.dst_path.
For example, it contains visualizations of the density distribution, as well as graphs showing the evolution of various parameters during the optimization process, such as the density field, volume fraction, and sensitivity values.

.. image:: https://raw.githubusercontent.com/kevin-tofu/scikit-topt/master/assets/ex-multi-load-condition.jpg
   :alt: multi-load-condition
   :width: 400px
   :align: center

.. image:: https://raw.githubusercontent.com/kevin-tofu/scikit-topt/master/assets/ex-multi-load-v-50.jpg
   :alt: Multi-Load-condition-Density-Distribution
   :width: 400px
   :align: center

.. raw:: html

   <video width="640" height="360" controls>
     <source src="_static/animation-box-rho.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>