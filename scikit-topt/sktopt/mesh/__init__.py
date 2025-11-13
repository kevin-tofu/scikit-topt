
from sktopt.mesh.task_common import FEMDomain
from sktopt.mesh.task_elastic import LinearElasticity
from sktopt.mesh.task_heat import LinearHeatConduction
from sktopt.mesh import toy_problem
from sktopt.mesh import loader

FEMDomain.__module__ = "sktopt.mesh"
LinearElasticity.__module__ = "sktopt.mesh"
LinearHeatConduction.__module__ = "sktopt.mesh"

__all__ = [
    "FEMDomain",
    "LinearElasticity",
    "LinearHeatConduction",
    "toy_problem",
    "loader"
]
