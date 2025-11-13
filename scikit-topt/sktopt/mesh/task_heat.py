from typing import Union, List, Literal, Optional
from dataclasses import dataclass

import numpy as np
import skfem

from skfem import FacetBasis, asm, LinearForm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sktopt.mesh import FEMDomain


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
    robin_bc_value: float | List[float],
    rho: Optional[np.ndarray] = None,
    p: Optional[float] = None
):
    def _to_list(x):
        return x if isinstance(x, list) else [x]

    facets_list = _to_list(robin_facets_ids)
    robin_h_list = [
        robin_coefficient for _ in range(len(facets_list))
    ] if isinstance(robin_coefficient, float) else robin_coefficient
    robin_Tenv_list = [
        robin_bc_value for _ in range(len(facets_list))
    ] if isinstance(robin_bc_value, float) else robin_bc_value

    if not (len(facets_list) == len(robin_h_list) == len(robin_Tenv_list)):
        raise ValueError(
            "Lengths of robin_facets_ids and robin_value\
                must match when lists."
        )

    rho_field = None
    if rho is not None:
        # nodal field if given as element-wise array
        if rho.shape[0] == basis.mesh.nelements:
            # element平均ρ → 節点に平均化
            rho_nodal = np.zeros(basis.mesh.nvertices)
            count = np.zeros(basis.mesh.nvertices)
            t = basis.mesh.t
            np.add.at(rho_nodal, t.ravel(), np.repeat(rho, t.shape[0]))
            np.add.at(count, t.ravel(), 1)
            rho_nodal /= np.maximum(count, 1)
        else:
            rho_nodal = rho

        # rho_field = basis.interpolate(rho_nodal)

    linear_list = list()
    bilinear_list = list()
    for facets, h, Tenv in zip(
        facets_list, robin_h_list, robin_Tenv_list
    ):
        fb = FacetBasis(
            basis.mesh, basis.elem,
            facets=np.asarray(facets, dtype=int)
        )
        if rho_field is not None:
            rho_facet = fb.interpolate(rho_nodal)

        @skfem.BilinearForm
        def robin_form(u, v, w):
            if rho_field is None:
                h_eff = w.h
            else:
                h_eff = w.h * w.rho**p
            return h_eff * u * v

        @skfem.LinearForm
        def robin_load(v, w):
            if rho_field is None:
                h_eff = w.h
            else:
                h_eff = w.h * w.rho**p
            return h_eff * w.Tenv * v

        w_dict = {'h': h, 'Tenv': Tenv}
        if rho_field is not None:
            w_dict['rho'] = rho_facet

        bilinear_list.append(skfem.asm(robin_form, fb, **w_dict))
        linear_list.append(skfem.asm(robin_load, fb, **w_dict))

    # if len(bilinear_list) == 1:
    #     bilinear_list = bilinear_list[0]
    #     linear_list = linear_list[0]
    return bilinear_list, linear_list


@dataclass
class LinearHeatConduction(FEMDomain):
    k: float  # thermal conductivity
    robin_bilinear: Optional[list] = None
    robin_linear: Optional[list] = None
    objective: Literal["compliance"] = "compliance"

    def update_robin_bc(self, rho: np.ndarray, p: float):
        robin_bilinear, robin_linear = assemble_surface_robin(
            self.basis,
            robin_facets_ids=self.robin_facets_ids,
            robin_coefficient=self.robin_coefficient,
            robin_bc_value=self.robin_bc_value,
            rho=rho, p=p
        )
        self.robin_bilinear = robin_bilinear
        self.robin_linear = robin_linear

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
        basis: skfem.Basis,
        dirichlet_facets_ids: np.ndarray | list[np.ndarray],
        dirichlet_values: float | list[float],
        robin_facets_ids: np.ndarray | list[np.ndarray] | None,
        robin_coefficient: float | list[float] | None,
        robin_bc_value: float | list[float] | None,
        design_robin_boundary: bool | None,
        design_elements: np.ndarray,
        k: float,
        objective: Literal["compliance"] = "compliance"
    ) -> 'LinearHeatConduction':

        if objective not in ["compliance"]:
            raise ValueError("should be compliance")

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
            design_robin_boundary,
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
            base.robin_facets_ids,
            base.robin_nodes,
            base.robin_elements,
            base.robin_coefficient,
            base.robin_bc_value,
            base.design_robin_boundary,
            base.design_elements,
            base.free_dofs,
            base.free_elements,
            base.all_elements,
            base.fixed_elements,
            base.dirichlet_neumann_elements,
            base.elements_volume,
            k, robin_bilinear, robin_linear, objective
        )

    @classmethod
    def from_mesh_tags(
        cls,
        basis: skfem.Basis,
        dirichlet_values: float | list[float],
        robin_coefficient: float | list[float],
        robin_bc_value: float | list[float],
        design_robin_boundary: bool | None,
        k: float,
        objective: Literal["compliance"] = "compliance"
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
            basis,
            dirichlet_facets_ids,
            dirichlet_values,
            robin_facets_ids,
            robin_coefficient,
            robin_bc_value,
            design_robin_boundary,
            design_elements,
            k, objective
        )
