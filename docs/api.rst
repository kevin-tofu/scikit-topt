API Reference
=============

This section summarizes the main public API of Scikit-Topt.
For full details, see the module-specific pages below.

Core Optimizers
---------------

Key classes:

- :class:`sktopt.core.OC_Optimizer`
- :class:`sktopt.core.LogMOC_Optimizer`
- :class:`sktopt.core.DensityMethodConfig`
- :class:`sktopt.core.OptimizationHistory`

Full API documentation: :mod:`sktopt.core`

.. toctree::
   :maxdepth: 1
   :caption: Core Optimizers

   core


Finite Element Solvers
----------------------

Key classes:

- :class:`sktopt.fea.composer.LinearElasticity`
- :class:`sktopt.fea.composer.SteadyStateHeatConduction`

Full API documentation: :mod:`sktopt.fea`

.. toctree::
   :maxdepth: 1
   :caption: Finite Element Solvers

   fea


Tools
-----

Key utilities:

- :class:`sktopt.tools.SchedulerConfig`
- :class:`sktopt.tools.HistoryLogger`
- :mod:`sktopt.tools.logconf`

.. toctree::
   :maxdepth: 1
   :caption: Tools to manage optimization

   tools


Mesh and Task Utilities
-----------------------

Key utilities:

- :mod:`sktopt.mesh.toy_problem`
- :mod:`sktopt.mesh.loader`
- :mod:`sktopt.mesh.utils`

.. toctree::
   :maxdepth: 1
   :caption: Mesh and Task Utilities

   mesh
   mesh.utils
