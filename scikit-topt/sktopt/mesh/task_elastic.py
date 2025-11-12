from typing import Union, List, Literal
from dataclasses import dataclass

import numpy as np
import skfem

from skfem import FacetBasis, asm, LinearForm
from sktopt.mesh import FEMDomain


_lit_bc = Literal['u^1', 'u^2', 'u^3', 'all']
_lit_force = Literal['u^1', 'u^2', 'u^3']


def assemble_surface_forces(
    basis: skfem.Basis,
    force_facets_ids: Union[np.ndarray, List[np.ndarray]],
    force_dir_type: Union[str, List[str]],
    force_value: Union[float, List[float]]
) -> list:
    def _to_list(x):
        return x if isinstance(x, list) else [x]

    def _dir_to_comp(s: str) -> int:
        if not (isinstance(s, str) and s.startswith('u^') and s[2:].isdigit()):
            raise ValueError(
                f"force_dir_type must be like 'u^1','u^2','u^3', got: {s}"
            )
        c = int(s[2:]) - 1
        if c < 0:
            raise ValueError(f"Invalid component index parsed from {s}")
        return c

    facets_list = _to_list(force_facets_ids)
    dirs_list = _to_list(force_dir_type)
    vals_list = _to_list(force_value)

    if not (len(facets_list) == len(dirs_list) == len(vals_list)):
        # print("len(facets_list) : ", len(facets_list))
        # print("len(dirs_list) : ", len(dirs_list))
        # print("len(vals_list) : ", len(vals_list))
        raise ValueError(
            "Lengths of force_facets_ids, force_dir_type, and force_value\
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
        pressure = float(val) / A
        # pressure = float(val)

        @LinearForm
        def l_comp(v, w):
            return pressure * v[comp]
            # return pressure * dot(v, n)

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
class LinearElasticity(FEMDomain):
    """
    Container for storing finite element and optimization-related data
    used in topology optimization tasks.

    This class holds material properties, boundary condition information,
    designable and non-designable element indices, as well as force vectors
    and volume data for each element. It is typically constructed using
    `LinearElasticity.from_defaults`.

    Attributes
    ----------
    E : float
        Young's modulus of the base material.
    nu : float
        Poisson's ratio of the base material.
    """

    E: float
    nu: float
    neumann_linear: list[np.array]
    body_force: np.ndarray | None = None

    @property
    def material_coef(self) -> float:
        return self.E

    @property
    def n_tasks(self) -> int:
        return len(self.neumann_linear)

    @property
    def force(self):
        return self.neumann_values

    @force.setter
    def force(self, value):
        self.neumann_values = value

    @property
    def force_elements(self):
        return self.neumann_elements

    @force_elements.setter
    def force_elements(self, value):
        self.neumann_elements = value

    @property
    def force_nodes(self):
        return self.neumann_elements

    @force_nodes.setter
    def force_nodes(self, value):
        self.neumann_nodes = value

    @property
    def dirichlet_force_elements(self):
        return self.dirichlet_neumann_elements

    @dirichlet_force_elements.setter
    def dirichlet_force_elements(self, value):
        self.dirichlet_neumann_elements = value

    @classmethod
    def from_facets(
        cls,
        basis: skfem.Basis,
        dirichlet_facets_ids: np.ndarray | list[np.ndarray],
        dirichlet_dir: _lit_bc | list[_lit_bc],
        force_facets_ids: np.ndarray | list[np.ndarray],
        force_dir_type: str | list[str],
        force_value: float | list[float],
        design_elements: np.ndarray,
        E: float,
        nu: float,
    ) -> 'LinearElasticity':
        """
        Create a TaskConfig from facet-based boundary-condition specifications.

        This constructor allows you to specify Dirichlet and Neumann boundaries
        directly via facet indices (rather than node indices). It will internally:

        - Convert `dirichlet_facets_ids` into the corresponding Dirichlet node set.
        - Resolve Dirichlet DOFs from those nodes and `dirichlet_dir`.
        - Convert `force_facets_ids` into the corresponding force node set.
        - Assemble the Neumann (surface) load vector(s) using
            `assemble_surface_forces`.
        - Forward all data to `TaskConfig.neumann_` to build the final config.

        Parameters
        ----------
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
        basis : skfem.Basis
            Finite-element space providing mesh and DOFs.
        dirichlet_facets_ids : np.ndarray or list[np.ndarray]
            Indices of facets subject to Dirichlet boundary conditions. If a list
            is given, each entry corresponds to a boundary region.
        dirichlet_dir : _lit_bc or list[_lit_bc]
            Direction specifier(s) for Dirichlet constraints.  
            `_lit_bc = Literal['u^1', 'u^2', 'u^3', 'all']`  
            - `'all'` fixes all displacement components.  
            - `'u^1'`, `'u^2'`, `'u^3'` fix the respective component only.
        force_facets_ids : np.ndarray or list[np.ndarray]
            Indices of facets subject to Neumann (surface) forces. A list denotes
            multiple load regions.
        force_dir_type : str or list[str]
            Direction specifier(s) for each force region.  
            `_lit_force = Literal['u^1', 'u^2', 'u^3']`  
            Indicates along which component the surface load is applied.
        force_value : float or list[float]
            Magnitude(s) of the surface forces, one per region if multiple.
        design_elements : np.ndarray
            Element indices initially considered designable. Force-touching
            elements will be excluded downstream.

        Returns
        -------
        TaskConfig
            A fully initialized TaskConfig, equivalent to what
            `TaskConfig.neumann_` produces but constructed from facet-based
            specifications.
        """
        base = FEMDomain.from_facets(
            basis,
            dirichlet_facets_ids,
            dirichlet_dir,
            None,
            force_facets_ids,
            force_dir_type,
            force_value,
            None, None, None, None,
            design_elements
        )

        neumann_linear = assemble_surface_forces(
            base.basis,
            force_facets_ids=force_facets_ids,
            force_dir_type=base.neumann_dir_type,
            force_value=base.neumann_values
        )
        if isinstance(neumann_linear, np.ndarray):
            neumann_linear = [neumann_linear]
        elif isinstance(neumann_linear, list):
            pass

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
            E, nu, neumann_linear
        )

    @property
    def force_elements_all(self) -> np.ndarray:
        return self.neumann_elements_all

    @classmethod
    def from_mesh_tags(
        cls,
        basis: skfem.Basis,
        dirichlet_dir: _lit_bc | list[_lit_bc],
        neumann_dir_type: str | list[str],
        neumann_values: float | list[float],
        E: float,
        nu: float,
    ) -> 'FEMDomain':
        import re

        # dirichlet_facets_ids: np.ndarray | list[np.ndarray]
        # neumann_facets_ids: np.ndarray | list[np.ndarray]
        # design_elements: np.ndarray

        design_elements = basis.mesh.subdomains["design"]
        keys = basis.mesh.boundaries.keys()
        dirichlet_keys = sorted(
            [k for k in keys if re.match(r"dirichlete_\d+$", k)],
            key=lambda x: int(re.search(r"\d+$", x).group())
        )
        if dirichlet_keys:
            dirichlet_facets_ids = [
                basis.mesh.boundaries[k] for k in dirichlet_keys
            ]
        elif "dirichlet" in keys:
            dirichlet_facets_ids = basis.mesh.boundaries["dirichlet"]
        else:
            dirichlet_facets_ids = np.array([])
        # 
        neumann_keys = sorted(
            [k for k in keys if re.match(r"neumann_\d+$", k)],
            key=lambda x: int(re.search(r"\d+$", x).group())
        )
        if neumann_keys:
            neumann_facets_ids = [
                basis.mesh.boundaries[k] for k in neumann_keys
            ]
        elif "neumann" in keys:
            neumann_facets_ids = [basis.mesh.boundaries["neumann"]]
        else:
            neumann_facets_ids = np.array([])
        return cls.from_facets(
            basis,
            dirichlet_facets_ids,
            dirichlet_dir,
            neumann_facets_ids,
            neumann_dir_type,
            neumann_values,
            design_elements,
            E, nu
        )
