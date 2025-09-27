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
    force_facets: Union[np.ndarray, List[np.ndarray]],
    force_dir_type: Union[str, List[str]],
    force_value: Union[float, List[float]],
    *,
    treat_value_as_total_force: bool = True,
):
    def _to_list(x):
        return x if isinstance(x, list) else [x]

    def _dir_to_comp(s: str) -> int:
        if not (isinstance(s, str) and s.startswith('u^') and s[2:].isdigit()):
            raise ValueError(f"force_dir_type must be like 'u^1','u^2','u^3', got: {s}")
        c = int(s[2:]) - 1
        if c < 0:
            raise ValueError(f"Invalid component index parsed from {s}")
        return c

    facets_list = _to_list(force_facets)
    dirs_list = _to_list(force_dir_type)
    vals_list = _to_list(force_value)

    if not (len(facets_list) == len(dirs_list) == len(vals_list)):
        raise ValueError(
            "Lengths of force_facets, force_dir_type, and force_value must match when lists."
        )

    @LinearForm
    def l_one(v, w):
        return 1.0

    F_list = []

    for facets, dir_s, val in zip(facets_list, dirs_list, vals_list):
        comp = _dir_to_comp(dir_s)
        fb = FacetBasis(basis.mesh, basis.elem, facets=np.asarray(facets, dtype=int))

        A_arr = asm(l_one, fb)
        A = float(np.sum(A_arr))  # sum up all contributions to get scalar area

        if A <= 0.0:
            raise ValueError(
                "Selected facets have zero total area; check facet indices or geometry."
            )

        if treat_value_as_total_force:
            p = float(val) / A
        else:
            p = float(val)

        @LinearForm
        def l_comp(v, w):
            return p * v[comp]

        F = asm(l_comp, fb)
        F_list.append(F)

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
        force_facets: np.ndarray | list[np.ndarray],
        force_dir_type: str | list[str],
        force_value: float | list[float],
        design_elements: np.ndarray,
    ):
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

        # if isinstance(force_nodes, list):
        #     force_dofs = [
        #         basis.get_dofs(nodes=n_loop).nodal[dir_loop]
        #         for n_loop, dir_loop in zip(
        #             force_nodes, force_dir
        #         )
        #     ]
        # elif isinstance(force_nodes, np.ndarray):
        #     force_dofs = basis.get_dofs(nodes=force_nodes).nodal[force_dir]
        # else:
        #     raise ValueError("force_nodes is not np.ndarray or of list")        

        return cls.from_defaults(
            E, nu, basis,
            dirichlet_nodes, dirichlet_dofs,
            force_facets, force_dir_type, force_value,
            design_elements
        )

    @classmethod
    def from_defaults(
        cls,
        E: float,
        nu: float,
        basis: skfem.Basis,
        dirichlet_nodes: np.ndarray,
        dirichlet_dofs: np.ndarray,
        force_facets: np.ndarray | list[np.ndarray],
        force_dir_type: str | list[str],
        force_value: float | list[float],
        design_elements: np.ndarray,
    ) -> 'TaskConfig':
        """Create a TaskConfig instance using basic material parameters and \
        boundary conditions.

        This method automatically computes the sets of Dirichlet and \
            Neumann (force) elements,
        filters out non-designable elements, and constructs force vectors.

        Parameters
        ----------
        E : float
            Young's modulus of the material.
        nu : float
            Poisson's ratio of the material.
        basis : skfem.Basis
            Basis object from scikit-fem representing the finite element \
                space.
        dirichlet_nodes : np.ndarray
            Coordinates used to determine which elements are subject to \
                Dirichlet boundary conditions.
        dirichlet_dofs : np.ndarray
            Degrees of freedom fixed under Dirichlet boundary conditions.
        force_nodes : np.ndarray or list of np.ndarray
            Coordinates used to determine which elements are subject to \
                external forces.
        force_value : float or list of float
            Magnitude(s) of the external force(s). If multiple load cases \
                are used, provide a list.
        design_elements : np.ndarray
            Initial set of element indices considered designable.

        Returns
        -------
        TaskConfig
            A fully initialized TaskConfig object containing mesh, boundary condition, and load data.

        Raises
        ------
        ValueError
            If no force elements are found or if design elements are empty after filtering.
        """

        force_facets_ids = np.concatenate(force_facets) \
            if isinstance(force_facets, list) else force_facets
        facets = basis.mesh.facets                  # facet→node 接続
        force_nodes = np.unique(facets[:, force_facets_ids].ravel())
        force = assemble_surface_forces(
            basis,
            force_facets=force_facets,
            force_dir_type=force_dir_type,
            force_value=force_value
        )

        #
        # Dirichlet
        #
        dirichlet_elements = utils.get_elements_with_nodes_fast(
            basis.mesh, [dirichlet_nodes]
        )
        #
        # Force
        #
        if isinstance(force_nodes, np.ndarray):
            force_elements = utils.get_elements_with_nodes_fast(
                basis.mesh, [force_nodes]
            )
        else:
            force_elements = utils.get_elements_with_nodes_fast(
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
        free_elements = utils.get_elements_with_nodes_fast(
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
