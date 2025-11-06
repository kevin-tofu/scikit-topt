from typing import Union, List, Literal, Optional
from dataclasses import dataclass

import numpy as np
import skfem

from skfem import FacetBasis, asm, LinearForm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sktopt.mesh.task_common import FEMDomain


def setdiff1d(a, b):
    mask = ~np.isin(a, b)
    a = a[mask]
    return np.ascontiguousarray(a)


_lit_bc = Literal['u^1', 'u^2', 'u^3', 'all']
# _lit_robin = Literal['u^1', 'u^2', 'u^3']


def assemble_surface_robin(
    basis,
    robin_facets_ids: Union[np.ndarray, List[np.ndarray]],
    robin_coefficient: Union[float, List[float]],
    robin_bc_value: Union[float, List[float]]
):
    def _to_list(x):
        return x if isinstance(x, list) else [x]

    facets_list = _to_list(robin_facets_ids)
    robin_h_list = _to_list(robin_coefficient)
    robin_Tenv_list = _to_list(robin_bc_value)

    if not (len(facets_list) == len(robin_h_list) == len(robin_Tenv_list)):
        raise ValueError(
            "Lengths of robin_facets_ids and robin_value\
                must match when lists."
        )

    linear_list = list()
    bilinear_list = list()
    for facets, h, Tenv in zip(
        facets_list, robin_h_list, robin_Tenv_list
    ):
        fb = FacetBasis(
            basis.mesh, basis.elem,
            facets=np.asarray(facets, dtype=int)
        )

        @skfem.BilinearForm
        def robin_form(u, v, w):
            # Surface integral (heat radiation term)
            return h * u * v

        @skfem.LinearForm
        def robin_load(v, w):
            # Temperature difference from outside temperature
            return h * Tenv * v

        bilinear_list.append(asm(robin_form, fb))
        linear_list.append(asm(robin_load, fb))

    if len(bilinear_list) == 1:
        bilinear_list = bilinear_list[0]
        linear_list = linear_list[0]
    return bilinear_list, linear_list


@dataclass
class LinearHeatConduction():

    k: float  # thermal conductivity
    q: np.ndarray | None = None  # heat source per volume
    robin_bilinear: np.array
    robin_linear: np.array

    @classmethod
    def from_facets(
        cls,
        k: float,
        q: Optional[np.ndarray],
        basis: skfem.Basis,
        dirichlet_facets_ids: np.ndarray | list[np.ndarray],
        dirichlet_dir: _lit_bc | list[_lit_bc],
        robin_facets_ids: np.ndarray | list[np.ndarray],
        robin_coefficient: float | list[float],
        robin_bc_value: float | list[float],
        design_elements: np.ndarray,
    ) -> 'LinearHeatConduction':

        base = FEMDomain.from_facets(
            basis,
            dirichlet_facets_ids,
            dirichlet_dir,
            None, None, None, 
            robin_facets_ids,
            robin_coefficient,
            robin_bc_value,
            design_elements
        )

        robin_bilinear, robin_linear = assemble_surface_robin(
            base.basis,
            robin_facets_ids=robin_facets_ids,
            robin_coefficient=base.robin_coefficient,
            robin_bc_value=base.robin_bc_value
        )
        return cls(
            base.basis,
            base.dirichlet_nodes,
            base.dirichlet_dofs,
            base.dirichlet_elements,
            base.neumann_nodes,
            base.neumann_elements,
            base.neumann_dir_type,
            base.neumann_values,
            base.robin_nodes,
            base.robin_elements,
            base.robin_coefficient,
            base.robin_bc_value,
            base.design_elements,
            base.free_dofs,
            base.free_elements,
            base.all_elements,
            base.fixed_elements,
            base.dirichlet_neumann_elements,
            base.elements_volume,
            k, q, robin_bilinear, robin_linear
        )
