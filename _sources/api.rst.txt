API Reference
=============

This section summarizes the main public API of Scikit-Topt.
For full details, see the module-specific pages below.


.. toctree::
   :maxdepth: 1
   :caption: Core Optimizers

   core

.. currentmodule:: sktopt.core

**Key Classes**

- :class:`DensityMethodConfig`
- :class:`DensityMethod_OC_Config`
- :class:`DensityMethod`
- :class:`OC_Config`
- :class:`OC_Optimizer`
- :class:`LogMOC_Config`
- :class:`LogMOC_Optimizer`


.. toctree::
   :maxdepth: 1
   :caption: Finite Element Solvers

   fea

.. currentmodule:: sktopt.fea

**Key Classes**

- :class:`FEM_SimpLinearElasticity`
- :class:`FEM_SimpLinearHeatConduction`


.. toctree::
   :maxdepth: 1
   :caption: Tools to manage optimization

   tools

.. currentmodule:: sktopt.tools

**Key Utilities**

- :class:`HistoryCollection`
- :class:`SchedulerConfig`
- :class:`SchedulerStep`
- :class:`SchedulerStepAccelerating`
- :class:`SchedulerStepDecelerating`
- :class:`SchedulerSawtoothDecay`


.. toctree::
   :maxdepth: 1
   :caption: Mesh and Task Utilities

   mesh

.. currentmodule:: sktopt.mesh

**Key Modules**

- :class:`FEMDomain`
- :class:`LinearElasticity`
- :class:`LinearHeatConduction`
