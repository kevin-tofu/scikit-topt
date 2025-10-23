from typing import Union, List, Literal
from dataclasses import dataclass

import numpy as np
import skfem

from skfem import FacetBasis, asm, LinearForm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sktopt.mesh import utils
from sktopt.fea import composer


def setdiff1d(a, b):
    mask = ~np.isin(a, b)
    a = a[mask]
    return np.ascontiguousarray(a)


_lit_bc = Literal['u^1', 'u^2', 'u^3', 'all']
_lit_force = Literal['u^1', 'u^2', 'u^3']


def assemble_surface_forces(
    basis,
    force_facets_ids: Union[np.ndarray, List[np.ndarray]],
    force_dir_type: Union[str, List[str]],
    force_value: Union[float, List[float]],
    *,
    treat_value_as_total_force: bool = True,
):
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
        if A <= 0.0:
            raise ValueError(
                "Selected facets have zero total area; check facet indices or geometry."
            )

        if treat_value_as_total_force:
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
class TaskConfig():
    """
    Container for storing finite element and optimization-related data
    used in topology optimization tasks.

    This class holds material properties, boundary condition information,
    designable and non-designable element indices, as well as force vectors
    and volume data for each element. It is typically constructed using
    `TaskConfig.from_defaults`.

    Attributes
    ----------
    E : float
        Young's modulus of the base material.
    nu : float
        Poisson's ratio of the base material.
    basis : skfem.Basis
        Finite element basis object associated with the mesh and function space.
    dirichlet_dofs : np.ndarray
        Degrees of freedom constrained by Dirichlet (displacement) boundary conditions.
    dirichlet_elements : np.ndarray
        Elements that contain Dirichlet boundary points.
    force_elements : np.ndarray
        Elements that contain the force application points.
    force : np.ndarray or list of np.ndarray
        External force vector(s) applied to the system.
        A list is used when multiple load cases are present.
    design_elements : np.ndarray
        Indices of elements that are considered designable in the optimization.
    free_dofs : np.ndarray
        Degrees of freedom that are not fixed by boundary conditions.
    free_elements : np.ndarray
        Elements associated with the free degrees of freedom.
    all_elements : np.ndarray
        Array of all element indices in the mesh.
    fixed_elements : np.ndarray
        Elements excluded from the design domain.
    dirichlet_force_elements : np.ndarray
        Union of Dirichlet and force elements.
        Useful for identifying constrained and loaded regions.
    elements_volume : np.ndarray
        Volume of each finite element, used in volume constraints and integration.
    """

    E: float
    nu: float
    basis: skfem.Basis
    dirichlet_nodes: np.ndarray
    dirichlet_dofs: np.ndarray
    dirichlet_elements: np.ndarray
    force_nodes: np.ndarray | list[np.ndarray] 
    force_elements: np.ndarray
    force: np.ndarray | list[np.ndarray]
    design_elements: np.ndarray
    free_dofs: np.ndarray
    free_elements: np.ndarray
    all_elements: np.ndarray
    fixed_elements: np.ndarray
    dirichlet_force_elements: np.ndarray
    elements_volume: np.ndarray

    @property
    def design_mask(self):
        return np.isin(self.all_elements, self.design_elements)

    @property
    def mesh(self):
        return self.basis.mesh

    @classmethod
    def from_nodes(
        cls,
        E: float,
        nu: float,
        basis: skfem.Basis,
        dirichlet_nodes: np.ndarray | list[np.ndarray],
        dirichlet_dir: _lit_bc | list[_lit_bc],
        force_nodes: np.ndarray | list[np.ndarray],
        force: np.ndarray | list[np.ndarray],
        design_elements: np.ndarray,
    ) -> 'TaskConfig':
        """
        Create a TaskConfig from material parameters and boundary-condition
        specifications expressed by **node-index** sets.

        This constructor:
          - Resolves Dirichlet DOFs from `dirichlet_nodes` and `dirichlet_dir`.
          - Identifies Dirichlet/Neumann (force) **elements** that touch those nodes.
          - Builds load vector(s) from `force` and filters out non-designable elements.
          - Derives sets such as free DOFs/elements and fixed/design elements.
          - Precomputes per-element volumes.

        Parameters
        ----------
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
        basis : skfem.Basis
            Finite-element space (provides mesh, DOFs, etc.).
        dirichlet_nodes : np.ndarray or list[np.ndarray]
            Global **node indices** that define Dirichlet boundaries.
            If a list is provided, it must align one-to-one with `dirichlet_dir`.
        dirichlet_dir : _lit_bc or list[_lit_bc]
            Direction specifier(s) for Dirichlet constraints.  
            `_lit_bc = Literal['u^1', 'u^2', 'u^3', 'all']`  
            Use `'all'` to fix all components, or `'u^1'|'u^2'|'u^3'` to fix a
            single component (compatible with `basis.get_dofs(...).nodal[...]`).
            When `dirichlet_nodes` is a list, this must be a list of equal length.
        force_nodes : np.ndarray or list[np.ndarray]
            Global **node indices** where Neumann (external) loads are applied.
            A list denotes multiple load regions (load cases).
        force : np.ndarray or list[np.ndarray]
            Load vector(s) associated with `force_nodes`. Provide a list to match
            multiple regions/cases. Each vector should already be assembled in the
            global DOF ordering expected by the solver.
        design_elements : np.ndarray
            Element indices initially considered designable. Elements touching
            `force_nodes` are removed from this set.

        Returns
        -------
        TaskConfig
            A fully initialized configuration containing:
              - material parameters (`E`, `nu`)
              - mesh/basis
              - `dirichlet_nodes`, `dirichlet_dofs`, `dirichlet_elements`
              - `force_nodes`, `force_elements`, `force`
              - filtered `design_elements`
              - `free_dofs`, `free_elements`
              - `all_elements`, `fixed_elements`, `dirichlet_force_elements`
              - `elements_volume`

        Notes
        -----
        - Element sets are computed via `utils.get_elements_by_nodes(...)`,
          i.e., elements **touching** the provided node sets.
        - `free_dofs` = all DOFs `dirichlet_dofs`; `free_elements` are elements
          touching `free_dofs` (convenience for downstream assembly/solves).
        - If you prefer facet-based specification (with direction literals for
          surface forces, e.g., `_lit_force = Literal['u^1','u^2','u^3']`),
          consider using `TaskConfig.from_facets(...)` which will assemble
          surface loads internally.
        """
        if isinstance(dirichlet_nodes, list):
            assert isinstance(dirichlet_dir, list)
            assert len(dirichlet_nodes) == len(dirichlet_dir)
        elif isinstance(dirichlet_nodes, np.ndarray):
            assert isinstance(dirichlet_dir, str)
            # assert dirichlet_dir in _lit_bc
        else:
            raise ValueError("dirichlet_nodes should be list or np.ndarray")

        #
        # Dirichlet
        #
        if isinstance(dirichlet_nodes, list):
            dirichlet_dofs = [
                basis.get_dofs(nodes=nodes).all() if direction == 'all'
                else basis.get_dofs(nodes=nodes).nodal[direction]
                for nodes, direction in zip(dirichlet_nodes, dirichlet_dir)
            ]
            dirichlet_dofs = np.concatenate(dirichlet_dofs)
            dirichlet_nodes = np.concatenate(dirichlet_nodes)
        elif isinstance(dirichlet_nodes, np.ndarray):
            dofs = basis.get_dofs(nodes=dirichlet_nodes)
            dirichlet_dofs = dofs.all() if dirichlet_dir == 'all' \
                else dofs.nodal[dirichlet_dir]
        else:
            raise ValueError("dirichlet_nodes is not np.ndarray or of list")

        dirichlet_elements = utils.get_elements_by_nodes(
            basis.mesh, [dirichlet_nodes]
        )
        #
        # Force
        #
        if isinstance(force_nodes, np.ndarray):
            force_elements = utils.get_elements_by_nodes(
                basis.mesh, [force_nodes]
            )
        elif isinstance(force_nodes, list):
            force_elements = utils.get_elements_by_nodes(
                basis.mesh, force_nodes
            )
        if force_elements.shape[0] == 0:
            raise ValueError("force_elements has not been set.")

        #
        # Design Field
        #
        design_elements = setdiff1d(design_elements, force_elements)
        if len(design_elements) == 0:
            error_msg = "⚠️Warning: `design_elements` is empty"
            raise ValueError(error_msg)

        all_elements = np.arange(basis.mesh.nelements)
        fixed_elements = setdiff1d(all_elements, design_elements)
        dirichlet_force_elements = np.concatenate(
            [dirichlet_elements, force_elements]
        )
        free_dofs = setdiff1d(np.arange(basis.N), dirichlet_dofs)
        free_elements = utils.get_elements_by_nodes(
            basis.mesh, [free_dofs]
        )
        elements_volume = composer.get_elements_volume(basis.mesh)
        print(
            f"all_elements: {all_elements.shape}",
            f"design_elements: {design_elements.shape}",
            f"fixed_elements: {fixed_elements.shape}",
            f"dirichlet_force_elements: {dirichlet_force_elements.shape}",
            f"force_elements: {force_elements}"
        )
        return cls(
            E,
            nu,
            basis,
            dirichlet_nodes,
            dirichlet_dofs,
            dirichlet_elements,
            force_nodes,
            force_elements,
            force,
            design_elements,
            free_dofs,
            free_elements,
            all_elements,
            fixed_elements,
            dirichlet_force_elements,
            elements_volume
        )

    @classmethod
    def from_facets(
        cls,
        E: float,
        nu: float,
        basis: skfem.Basis,
        dirichlet_facets_ids: np.ndarray | list[np.ndarray],
        dirichlet_dir: _lit_bc | list[_lit_bc],
        force_facets_ids: np.ndarray | list[np.ndarray],
        force_dir_type: str | list[str],
        force_value: float | list[float],
        design_elements: np.ndarray,
    ) -> 'TaskConfig':
        """
        Create a TaskConfig from facet-based boundary-condition specifications.

        This constructor allows you to specify Dirichlet and Neumann boundaries
        directly via facet indices (rather than node indices). It will internally:

        - Convert `dirichlet_facets_ids` into the corresponding Dirichlet node set.
        - Resolve Dirichlet DOFs from those nodes and `dirichlet_dir`.
        - Convert `force_facets_ids` into the corresponding force node set.
        - Assemble the Neumann (surface) load vector(s) using
            `assemble_surface_forces`.
        - Forward all data to `TaskConfig.from_nodes` to build the final config.

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
            `TaskConfig.from_nodes` produces but constructed from facet-based
            specifications.
        """

        facets = basis.mesh.facets
        if isinstance(dirichlet_facets_ids, list):
            dirichlet_nodes = list()
            for dirichlet_facets_ids_loop in dirichlet_facets_ids:
                dirichlet_nodes.append(
                    np.unique(facets[:, dirichlet_facets_ids_loop].ravel())
                )
        elif isinstance(dirichlet_facets_ids, np.ndarray):
            dirichlet_nodes = np.unique(facets[:, dirichlet_facets_ids].ravel())
        else:
            raise ValueError(
                "dirichlet_facets_ids should be list[np.ndarray] or np.ndarray"
            )

        force_facets_ids_concat = np.concatenate(force_facets_ids) \
            if isinstance(force_facets_ids, list) else force_facets_ids
        force_nodes = np.unique(facets[:, force_facets_ids_concat].ravel())
        force = assemble_surface_forces(
            basis,
            force_facets_ids=force_facets_ids,
            force_dir_type=force_dir_type,
            force_value=force_value
        )

        return cls.from_nodes(
            E, nu, basis,
            dirichlet_nodes, dirichlet_dir,
            force_nodes, force,
            design_elements
        )

    @classmethod
    def from_mesh_tags(
        cls,
        E: float,
        nu: float,
        basis: skfem.Basis,
        dirichlet_dir: _lit_bc | list[_lit_bc],
        force_dir_type: str | list[str],
        force_value: float | list[float],
    ) -> 'TaskConfig':
        import re

        # dirichlet_facets_ids: np.ndarray | list[np.ndarray]
        # force_facets_ids: np.ndarray | list[np.ndarray]
        # design_elements: np.ndarray

        design_elements = basis.mesh.subdomains["design"]
        dirichlet_facets_ids = basis.mesh.boundaries["dirichlet"]
        keys = basis.mesh.boundaries.keys()
        # 
        dirichlet_keys = sorted(
            [k for k in keys if re.match(r"dirichlete_\d+$", k)],
            key=lambda x: int(re.search(r"\d+$", x).group())
        )
        if dirichlet_keys:
            force_facets_ids = [
                basis.mesh.boundaries[k] for k in dirichlet_keys
            ]
        elif "dirichlet" in keys:
            force_facets_ids = [basis.mesh.boundaries["dirichlet"]]
        else:
            force_facets_ids = np.array([])
        # 
        force_keys = sorted(
            [k for k in keys if re.match(r"force_\d+$", k)],
            key=lambda x: int(re.search(r"\d+$", x).group())
        )
        if force_keys:
            force_facets_ids = [basis.mesh.boundaries[k] for k in force_keys]
        elif "force" in keys:
            force_facets_ids = [basis.mesh.boundaries["force"]]
        else:
            force_facets_ids = np.array([])
        return cls.from_facets(
            E, nu, basis,
            dirichlet_facets_ids,
            dirichlet_dir,
            force_facets_ids,
            force_dir_type,
            force_value,
            design_elements
        )

    @classmethod
    def from_json(self, path: str):
        raise NotImplementedError("not implmented yet")

    @property
    def force_nodes_all(self) -> np.ndarray:
        if isinstance(self.force_nodes, list):
            return np.unique(np.concatenate(self.force_nodes))
        else:
            return self.force_nodes

    def export_analysis_condition_on_mesh(
        self, dst_path: str
    ):
        import meshio
        mesh = self.basis.mesh
        if isinstance(mesh, skfem.MeshTet):
            cell_type = "tetra"
        elif isinstance(mesh, skfem.MeshHex):
            cell_type = "hexahedron"
        else:
            raise ValueError("Unsupported mesh type for VTU export.")

        # Points (shape: [n_nodes, dim])
        points = mesh.p.T
        node_colors_df = np.zeros(mesh.p.shape[1], dtype=int)
        node_colors_df[self.force_nodes_all] = 1
        node_colors_df[self.dirichlet_nodes] = 2
        point_outputs = dict()
        point_outputs["node_color"] = node_colors_df

        # Elements
        element_colors_df = np.zeros(mesh.nelements, dtype=int)
        element_colors_df[self.free_elements] = 1
        element_colors_df[self.fixed_elements] = 2
        element_colors_df[self.design_elements] = 3
        cells = [(cell_type, mesh.t.T)]
        cell_outputs = dict()
        cell_outputs["condition"] = [element_colors_df]

        meshio_mesh = meshio.Mesh(
            points=points,
            cells=cells,
            point_data=point_outputs,
            cell_data=cell_outputs
        )
        meshio_mesh.write(f"{dst_path}/condition.vtu")

    def exlude_dirichlet_from_design(self):
        self.design_elements = setdiff1d(
            self.design_elements, self.dirichlet_elements
        )

    def scale(
        self,
        L_scale: float,
        F_scale: float
    ):
        # this wont work
        # self.basis.mesh.p /= L_scale
        mesh = self.basis.mesh
        p_scaled = mesh.p * L_scale
        mesh_scaled = type(mesh)(p_scaled, mesh.t)
        basis_scaled = skfem.Basis(mesh_scaled, self.basis.elem)
        self.basis = basis_scaled

        if isinstance(self.force, np.ndarray):
            self.force *= F_scale
        elif isinstance(self.force, list):
            for loop in range(len(self.force)):
                self.force[loop] *= F_scale
        else:
            raise ValueError("should be ndarray or list of ndarray")

    def nodes_and_elements_stats(self, dst_path: str):
        node_points = self.basis.mesh.p.T  # shape = (n_points, 3)
        tree_nodes = cKDTree(node_points)
        dists_node, _ = tree_nodes.query(node_points, k=2)
        node_nearest_dists = dists_node[:, 1]

        element_centers = np.mean(
            self.basis.mesh.p[:, self.basis.mesh.t], axis=1
        ).T
        tree_elems = cKDTree(element_centers)
        dists_elem, _ = tree_elems.query(element_centers, k=2)
        element_nearest_dists = dists_elem[:, 1]

        print("===Distance between nodes ===")
        print(f"min:    {np.min(node_nearest_dists):.4f}")
        print(f"max:    {np.max(node_nearest_dists):.4f}")
        print(f"mean:   {np.mean(node_nearest_dists):.4f}")
        print(f"median: {np.median(node_nearest_dists):.4f}")
        print(f"std:    {np.std(node_nearest_dists):.4f}")

        print("\n=== Distance between elements ===")
        print(f"min:    {np.min(element_nearest_dists):.4f}")
        print(f"max:    {np.max(element_nearest_dists):.4f}")
        print(f"mean:   {np.mean(element_nearest_dists):.4f}")
        print(f"median: {np.median(element_nearest_dists):.4f}")
        print(f"std:    {np.std(element_nearest_dists):.4f}")

        plt.clf()
        fig, axs = plt.subplots(2, 3, figsize=(14, 6))

        axs[0, 0].hist(node_nearest_dists, bins=30, edgecolor='black')
        axs[0, 0].set_title("Nearest Neighbor Distance (Nodes)")
        axs[0, 0].set_xlabel("Distance")
        axs[0, 0].set_ylabel("Count")
        axs[0, 0].grid(True)

        axs[0, 1].hist(element_nearest_dists, bins=30, edgecolor='black')
        axs[0, 1].set_title("Nearest Neighbor Distance (Element Centers)")
        axs[0, 1].set_xlabel("Distance")
        axs[0, 1].set_ylabel("Count")
        axs[0, 1].grid(True)

        axs[1, 0].hist(
            self.elements_volume, bins=30, edgecolor='black'
        )
        axs[1, 0].set_title("elements_volume - all")
        axs[1, 0].set_xlabel("Volume")
        axs[1, 0].set_ylabel("Count")
        axs[1, 0].grid(True)
        axs[1, 1].hist(
            self.elements_volume[self.design_elements],
            bins=30, edgecolor='black'
        )
        axs[1, 1].set_title("elements_volume - design")
        axs[1, 1].set_xlabel("Volume")
        axs[1, 1].set_ylabel("Count")
        axs[1, 1].grid(True)
        items = [
            "all", "dirichlet", "force", "design"
        ]
        values = [
            np.sum(self.elements_volume),
            np.sum(self.elements_volume[self.dirichlet_elements]),
            np.sum(self.elements_volume[self.force_elements]),
            np.sum(self.elements_volume[self.design_elements])
        ]
        bars = axs[1, 2].bar(items, values)
        # axs[1, 0].bar_label(bars)
        for bar in bars:
            yval = bar.get_height()
            axs[1, 2].text(
                bar.get_x() + bar.get_width()/2,
                yval + 0.5, f'{yval:.2g}', ha='center', va='bottom'
            )

        axs[1, 2].set_title("THe volume difference elements")
        axs[1, 2].set_xlabel("Elements Attribute")
        axs[1, 2].set_ylabel("Volume")

        fig.tight_layout()
        fig.savefig(f"{dst_path}/info-nodes-elements.jpg")
        plt.close("all")
