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
_lit_neumann = Literal['u^1', 'u^2', 'u^3']


def assemble_surface_neumanns(
    basis,
    neumann_facets_ids: Union[np.ndarray, List[np.ndarray]],
    neumann_dir_type: Union[str, List[str]],
    neumann_value: Union[float, List[float]],
    *,
    treat_value_as_total_neumann: bool = True,
):
    def _to_list(x):
        return x if isinstance(x, list) else [x]

    def _dir_to_comp(s: str) -> int:
        if not (isinstance(s, str) and s.startswith('u^') and s[2:].isdigit()):
            raise ValueError(
                f"neumann_dir_type must be like 'u^1','u^2','u^3', got: {s}"
            )
        c = int(s[2:]) - 1
        if c < 0:
            raise ValueError(f"Invalid component index parsed from {s}")
        return c

    facets_list = _to_list(neumann_facets_ids)
    dirs_list = _to_list(neumann_dir_type)
    vals_list = _to_list(neumann_value)

    if not (len(facets_list) == len(dirs_list) == len(vals_list)):
        # print("len(facets_list) : ", len(facets_list))
        # print("len(dirs_list) : ", len(dirs_list))
        # print("len(vals_list) : ", len(vals_list))
        raise ValueError(
            "Lengths of neumann_facets_ids, neumann_dir_type, and neumann_value\
                must match when lists."
        )

    @skfem.Functional
    def l_one(w):
        return 1.0

    F_list = list()
    for facets, dir_s, val in zip(facets_list, dirs_list, vals_list):
        comp = _dir_to_comp(dir_s)
        fb = FacetBasis(
            basis.mesh, basis.elem,
            facets=np.asarray(facets, dtype=int)
        )

        A = asm(l_one, fb)
        if A <= 0.0:
            raise ValueError(
                "Selected facets have zero total area; check facet indices or geometry."
            )

        if treat_value_as_total_neumann:
            pressure = float(val) / A
        else:
            pressure = float(val)

        @LinearForm
        def l_comp(v, w):
            # print(f"v.shape {v.shape}")
            # print(f"w.n.shape {w.n.shape}")
            # v.shape (3, 46, 16)
            # w.n.shape (3, 46, 16)
            # return pressure * skfem.helpers.dot(w.n, v)
            return pressure * v[comp]

        F = asm(l_comp, fb)
        F_list.append(F)

        # ndim = basis.mesh.dim()
        # The order of F is [u1_x, u1_y, u1_z, u2_x, u2_y, u2_z, ...]
        # F_blocks = np.vstack([
        #     F[comp::ndim] for comp in range(ndim)
        # ])

        # print("x-block nonzero:", (abs(F_blocks[0]) > 1e-12).any())
        # print("y-block nonzero:", (abs(F_blocks[1]) > 1e-12).any())
        # print("z-block nonzero:", (abs(F_blocks[2]) > 1e-12).any())

    return F_list[0] if (len(F_list) == 1) else F_list


@dataclass
class LinearHeatConduction():

    k: float # thermal conductivity
    q: np.ndarray | None = None # heat source per volume

    @classmethod
    def from_facets(
        cls,
        k: float,
        q: Optional[np.ndarray],
        basis: skfem.Basis,
        dirichlet_facets_ids: np.ndarray | list[np.ndarray],
        dirichlet_dir: _lit_bc | list[_lit_bc],
        robin_facets_ids: np.ndarray | list[np.ndarray],
        robin_dir_type: str | list[str],
        robin_value: float | list[float],
        design_elements: np.ndarray,
    ) -> 'LinearHeatConduction':

        base = FEMDomain.from_facets(
            basis,
            dirichlet_facets_ids,
            dirichlet_dir,
            None, None, None, None,
            robin_facets_ids,
            robin_dir_type,
            robin_value,
            None,
            design_elements
        )

        robin_vector = assemble_surface_neumanns(
            base.basis,
            robin_facets_ids=robin_facets_ids,
            robin_dir_type=base.robin_dir_type,
            robin_value=base.robin_values
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
            base.neumann_vector,
            base.robin_nodes,
            base.robin_elements,
            base.robin_dir_type,
            base.robin_values,
            robin_vector,
            base.design_elements,
            base.free_dofs,
            base.free_elements,
            base.all_elements,
            base.fixed_elements,
            base.dirichlet_neumann_elements,
            base.elements_volume,
            k, q
        )
