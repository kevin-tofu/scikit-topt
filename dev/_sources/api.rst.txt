API Reference
=============

This section summarizes the main public API of Scikit-Topt.
For full details, see the module-specific pages below.

Full API documentation: :mod:`sktopt.core`

.. toctree::
   :maxdepth: 1
   :caption: Core Optimizers

   core

Key classes:

- :class:`sktopt.DensityMethodConfig`
- :class:`sktopt.DensityMethod_OC_Config`
- :class:`sktopt.DensityMethod`
- :class:`sktopt.OC_Config`
- :class:`sktopt.OC_Optimizer`
- :class:`sktopt.LogMOC_Config`
- :class:`sktopt.LogMOC_Optimizer`


Full API documentation: :mod:`sktopt.fea`

.. toctree::
   :maxdepth: 1
   :caption: Finite Element Solvers

   fea


Key classes:

- :class:`sktopt.fea.FEM_SimpLinearElasticity`
- :class:`sktopt.fea.FEM_SimpLinearHeatConduction`


.. toctree::
   :maxdepth: 1
   :caption: Tools to manage optimization

   tools

Key utilities:

- :class:`sktopt.tools.HistoryCollection`
- :class:`sktopt.tools.SchedulerConfig`
- :class:`sktopt.tools.SchedulerStepAccelerating`
- :class:`sktopt.tools.SchedulerStep`
- :class:`sktopt.tools.SchedulerSawtoothDecay`

.. toctree::
   :maxdepth: 1
   :caption: Mesh and Task Utilities

   mesh
   mesh.utils

Key utilities:

- :class:`sktopt.tools.FEMDomain`
- :class:`sktopt.tools.LinearElasticity`
- :class:`sktopt.tools.LinearHeatConduction`
- :mod:`sktopt.mesh.toy_problem`
- :mod:`sktopt.mesh.loader`


