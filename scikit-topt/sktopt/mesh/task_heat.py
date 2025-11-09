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


def assemble_surface_neumann(
    basis: skfem.Basis,
    neumann_facets_ids: Union[np.ndarray, List[np.ndarray]],
    neumann_value: Union[float, List[float]]
) -> list:
    def _to_list(x):
        return x if isinstance(x, list) else [x]

    facets_list = _to_list(neumann_facets_ids)
    vals_list = _to_list(neumann_value)

    if not (len(facets_list) == len(vals_list)):
        # print("len(facets_list) : ", len(facets_list))
        # print("len(dirs_list) : ", len(dirs_list))
        # print("len(vals_list) : ", len(vals_list))
        raise ValueError(
            "Lengths of facets_list and vals_list\
                must match when lists."
        )

    heat_list = list()
    for facets, q_flux in zip(facets_list, vals_list):
        fb = FacetBasis(
            basis.mesh, basis.elem,
            facets=np.asarray(facets, dtype=int)
        )

        @skfem.LinearForm
        def surface_heat_source(v, w):
            return q_flux * v  # q_n * v

        heat = asm(surface_heat_source, fb)
        heat_list.append(heat)

    return heat_list[0] if (len(heat_list) == 1) else heat_list


def assemble_surface_robin(
    basis,
    robin_facets_ids: np.ndarray | List[np.ndarray],
    robin_coefficient: float | List[float],
    robin_bc_value: float | List[float]
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
            return w.h * u * v

        @skfem.LinearForm
        def robin_load(v, w):
            # Temperature difference from outside temperature
            return w.h * w.Tenv * v

        bilinear_list.append(asm(robin_form, fb, h=h))
        linear_list.append(asm(robin_load, fb, h=h, Tenv=Tenv))

    if len(bilinear_list) == 1:
        bilinear_list = bilinear_list[0]
        linear_list = linear_list[0]
    return bilinear_list, linear_list


@dataclass
class LinearHeatConduction(FEMDomain):

    k: float  # thermal conductivity
    robin_bilinear: Optional[np.array]
    robin_linear: Optional[np.array]

    @property
    def material_coef(self) -> float:
        return self.k

    @property
    def n_tasks(self) -> int:
        ret = 1 if isinstance(self.dirichlet_values, float) \
            else len(self.dirichlet_values)
        return ret

    @classmethod
    def from_facets(
        cls,
        k: float,
        basis: skfem.Basis,
        dirichlet_facets_ids: np.ndarray | list[np.ndarray],
        dirichlet_values: float | list[float],
        # neumann_facets_ids: np.ndarray | list[np.ndarray],
        # neumann_values: float | list[float],
        robin_facets_ids: np.ndarray | list[np.ndarray] | None,
        robin_coefficient: float | list[float] | None,
        robin_bc_value: float | list[float] | None,
        design_elements: np.ndarray,
    ) -> 'LinearHeatConduction':

        dirichlet_dir = None
        neumann_facets_ids = None
        neumann_dir = None
        neumann_values = None
        base = FEMDomain.from_facets(
            basis,
            dirichlet_facets_ids,
            dirichlet_dir,
            dirichlet_values,
            neumann_facets_ids, neumann_dir, neumann_values,
            robin_facets_ids,
            robin_coefficient,
            robin_bc_value,
            design_elements
        )
        # heat_source_list = assemble_surface_neumann(
        #     base.basis,
        #     neumann_facets_ids,
        #     neumann_values
        # )

        if robin_facets_ids is not None:
            robin_bilinear, robin_linear = assemble_surface_robin(
                base.basis,
                robin_facets_ids=robin_facets_ids,
                robin_coefficient=base.robin_coefficient,
                robin_bc_value=base.robin_bc_value
            )
        else:
            robin_bilinear, robin_linear = None, None

        return cls(
            base.basis,
            base.dirichlet_nodes,
            base.dirichlet_dofs,
            base.dirichlet_elements,
            base.dirichlet_values,
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
            k, robin_bilinear, robin_linear
        )

    @classmethod
    def from_mesh_tags(
        cls,
        k: float,
        basis: skfem.Basis,
        dirichlet_values: float | list[float],
        robin_coefficient: float | list[float],
        robin_bc_value: float | list[float],
    ) -> 'FEMDomain':
        import re

        design_elements = basis.mesh.subdomains["design"]
        keys = basis.mesh.boundaries.keys()

        # dirichlet
        keys = basis.mesh.boundaries.keys()
        dirichlet_keys = sorted(
            [k for k in keys if re.match(r"dirichlet_\d+$", k)],
            key=lambda x: int(re.search(r"\d+$", x).group())
        )
        if dirichlet_keys:
            dirichlet_facets_ids = [
                basis.mesh.boundaries[k] for k in dirichlet_keys
            ]
        elif "dirichlet" in keys:
            dirichlet_facets_ids = basis.mesh.boundaries["dirichlet"]
        else:
            dirichlet_facets_ids = None

        # robin
        robin_keys = sorted(
            [k for k in keys if re.match(r"robin_\d+$", k)],
            key=lambda x: int(re.search(r"\d+$", x).group())
        )
        if robin_keys:
            robin_facets_ids = [
                basis.mesh.boundaries[k] for k in robin_keys
            ]
        elif "robin" in keys:
            robin_facets_ids = basis.mesh.boundaries["robin"]
        else:
            robin_facets_ids = None

        # neumann
        # neumann_keys = sorted(
        #     [k for k in keys if re.match(r"neumann_\d+$", k)],
        #     key=lambda x: int(re.search(r"\d+$", x).group())
        # )
        # if neumann_keys:
        #     neumann_facets_ids = [
        #         basis.mesh.boundaries[k] for k in neumann_keys
        #     ]
        # elif "neumann" in keys:
        #     neumann_facets_ids = [basis.mesh.boundaries["neumann"]]
        # else:
        #     neumann_facets_ids = None

        return cls.from_facets(
            k, basis,
            dirichlet_facets_ids,
            dirichlet_values,
            robin_facets_ids,
            robin_coefficient,
            robin_bc_value,
            design_elements
        )
