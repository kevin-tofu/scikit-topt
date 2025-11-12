from typing import Union, List, Literal, Optional
from dataclasses import dataclass

import numpy as np
import skfem

from skfem import FacetBasis, asm, LinearForm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sktopt.mesh import utils
from sktopt.fea import composer


_lit_bc = Literal['u^1', 'u^2', 'u^3', 'all']
_lit_force = Literal['u^1', 'u^2', 'u^3']


def setdiff1d(a, b):
    mask = ~np.isin(a, b)
    a = a[mask]
    return np.ascontiguousarray(a)


@dataclass
class FEMDomain():
    """
    Container for storing finite element and optimization-related data
    used in topology optimization tasks.

    This class holds material properties, boundary condition information,
    designable and non-designable element indices, as well as force vectors
    and volume data for each element. It is typically constructed using
    `LinearElastisicity.from_defaults`.

    Attributes
    ----------
    basis : skfem.Basis
        Finite element basis object associated with the mesh and function space.
    dirichlet_dofs : np.ndarray
        Degrees of freedom constrained by Dirichlet (displacement) boundary conditions.
    dirichlet_elements : np.ndarray
        Elements that contain Dirichlet boundary points.
    neumann_elements : np.ndarray
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
    dirichlet_neumann_elements : np.ndarray
        Union of Dirichlet and force elements.
        Useful for identifying constrained and loaded regions.
    elements_volume : np.ndarray
        Volume of each finite element, used in volume constraints and integration.
    """
    basis: skfem.Basis
    dirichlet_nodes: np.ndarray | list[np.ndarray] | None
    dirichlet_dofs: np.ndarray | list[np.ndarray] | None  # The nessecity is not high?
    dirichlet_elements: np.ndarray | None
    dirichlet_values: float | list[float] | None

    neumann_nodes: np.ndarray | list[np.ndarray] | None
    neumann_elements: np.ndarray | None
    neumann_dir_type: str | list[str] | None
    neumann_values: float | list[float] | None
    # neumann_vector: Optional[np.ndarray | list[np.ndarray]]

    robin_facets_ids: np.ndarray | list[np.ndarray] | None
    robin_nodes: np.ndarray | list[np.ndarray] | None
    robin_elements: np.ndarray | None
    robin_coefficient: float | list[float] | None
    robin_bc_value: float | list[float] | None
    design_robin_boundary: bool | None
    # robin_vector: Optional[np.ndarray | list[np.ndarray]]

    design_elements: np.ndarray
    free_dofs: np.ndarray
    free_elements: np.ndarray
    all_elements: np.ndarray
    fixed_elements: np.ndarray
    dirichlet_neumann_elements: np.ndarray
    elements_volume: np.ndarray

    @property
    def n_tasks(self) -> int:
        raise NotImplementedError("")

    @property
    def design_mask(self):
        return np.isin(self.all_elements, self.design_elements)

    @property
    def mesh(self):
        return self.basis.mesh

    @classmethod
    def from_nodes(
        cls,
        basis: skfem.Basis,
        dirichlet_nodes: np.ndarray | list[np.ndarray] | None,
        dirichlet_dir: _lit_bc | list[_lit_bc] | None,
        dirichlet_values: float | list[float] | None,
        neumann_nodes: np.ndarray | list[np.ndarray] | None,
        neumann_dir_type: str | list[str] | None,
        neumann_values: np.ndarray | list[np.ndarray] | None,
        robin_facets_ids: np.ndarray | list[np.ndarray] | None,
        robin_nodes: np.ndarray | list[np.ndarray] | None,
        robin_coefficient: float | list[float] | None,
        robin_bc_value: float | list[float] | None,
        design_robin_boundary: bool | None,
        design_elements: np.ndarray,
    ) -> 'FEMDomain':

        if dirichlet_nodes is not None:
            if isinstance(dirichlet_nodes, list):
                if dirichlet_dir is None:
                    is_list = isinstance(dirichlet_values, list)
                    is_float = isinstance(dirichlet_values, float)
                    assert is_list | is_float
                    if is_list:
                        assert len(dirichlet_nodes) == len(dirichlet_values)
                if dirichlet_values is None:
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
                if dirichlet_dir is not None:
                    dirichlet_dofs = [
                        basis.get_dofs(nodes=nodes).all() if direction == 'all'
                        else basis.get_dofs(nodes=nodes).nodal[direction]
                        for nodes, direction in zip(
                            dirichlet_nodes, dirichlet_dir
                        )
                    ]
                else:
                    dirichlet_dofs = [
                        basis.get_dofs(nodes=nodes).all()
                        for nodes in dirichlet_nodes
                    ]

                # dirichlet_dofs = np.concatenate(dirichlet_dofs)
                # dirichlet_nodes = np.concatenate(dirichlet_nodes)
            elif isinstance(dirichlet_nodes, np.ndarray):
                dofs = basis.get_dofs(nodes=dirichlet_nodes)
                dirichlet_dofs = dofs.all() if dirichlet_dir == 'all' \
                    else dofs.nodal[dirichlet_dir]
            else:
                raise ValueError("dirichlet_nodes is not np.ndarray or of list")

            dirichlet_elements = utils.get_elements_by_nodes(
                basis.mesh, [dirichlet_nodes]
            )
        else:
            dirichlet_dofs = None
            dirichlet_elements = None
            # dirichlet_elements = np.array([])

        #
        # neumann
        #
        if neumann_nodes is not None:
            if isinstance(neumann_nodes, np.ndarray):
                neumann_elements = utils.get_elements_by_nodes(
                    basis.mesh, [neumann_nodes]
                )
            elif isinstance(neumann_nodes, list):
                neumann_elements = utils.get_elements_by_nodes(
                    basis.mesh, neumann_nodes
                )
            if neumann_elements.shape[0] == 0:
                raise ValueError("neumann_elements has not been set.")
        else:
            neumann_elements = None

        #
        # robin
        #
        if robin_nodes is not None:
            if isinstance(robin_nodes, np.ndarray):
                robin_elements = utils.get_elements_by_nodes(
                    basis.mesh, [robin_nodes]
                )
            elif isinstance(robin_nodes, list):
                robin_elements = utils.get_elements_by_nodes(
                    basis.mesh, robin_nodes
                )
            if robin_elements.shape[0] == 0:
                raise ValueError("robin_elements has not been set.")
        else:
            robin_elements = None

        #
        # Design Field
        #
        elements_excluded = np.array([])
        if neumann_elements is not None:
            elements_excluded = np.concatenate(
                [elements_excluded, neumann_elements]
            )
        if design_robin_boundary is False:
            elements_excluded = np.concatenate(
                [elements_excluded, robin_elements]
            )

        design_elements = setdiff1d(design_elements, elements_excluded)
        if len(design_elements) == 0:
            error_msg = "⚠️Warning: `design_elements` is empty"
            raise ValueError(error_msg)

        all_elements = np.arange(basis.mesh.nelements)
        fixed_elements = setdiff1d(all_elements, design_elements)
        valid = [
            l for l in [dirichlet_elements, neumann_elements] if l is not None and len(l) > 0
        ]
        dirichlet_neumann_elements = np.concatenate(valid) \
            if len(valid) > 0 else np.array([], dtype=int)

        free_dofs = setdiff1d(np.arange(basis.N), dirichlet_dofs)
        free_elements = utils.get_elements_by_nodes(
            basis.mesh, [free_dofs]
        )
        elements_volume = composer.get_elements_volume(basis.mesh)
        print(
            f"all_elements: {all_elements.shape}",
            f"design_elements: {design_elements.shape}",
            f"fixed_elements: {fixed_elements.shape}",
            f"dirichlet_neumann_elements: {dirichlet_neumann_elements.shape}",
            f"neumann_elements: {neumann_elements}",
            f"robin_elements: {robin_elements}"
        )
        return cls(
            basis,
            dirichlet_nodes,
            dirichlet_dofs,
            dirichlet_elements,
            dirichlet_values,
            neumann_nodes,
            neumann_elements,
            neumann_dir_type,
            neumann_values,
            robin_facets_ids,
            robin_nodes,
            robin_elements,
            robin_coefficient,
            robin_bc_value,
            design_robin_boundary,
            design_elements,
            free_dofs,
            free_elements,
            all_elements,
            fixed_elements,
            dirichlet_neumann_elements,
            elements_volume
        )

    @classmethod
    def from_facets(
        cls,
        basis: skfem.Basis,
        dirichlet_facets_ids: np.ndarray | list[np.ndarray] | None,
        dirichlet_dir: _lit_bc | list[_lit_bc] | None,
        dirichlet_values: float | list[float] | None,
        neumann_facets_ids: np.ndarray | list[np.ndarray] | None,
        neumann_dir_type: str | list[str] | None,
        neumann_values: float | list[float] | None,
        robin_facets_ids: np.ndarray | list[np.ndarray] | None,
        robin_coefficient: float | list[float] | None,
        robin_bc_value: float | list[float] | None,
        design_robin_boundary: bool | None,
        design_elements: np.ndarray,
    ) -> 'FEMDomain':

        facets = basis.mesh.facets

        if dirichlet_facets_ids is None:
            dirichlet_nodes = None
        else:
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

        if neumann_facets_ids is not None:
            neumann_facets_ids_concat = np.concatenate(neumann_facets_ids) \
                if isinstance(neumann_facets_ids, list) else neumann_facets_ids
            neumann_nodes = np.unique(
                facets[:, neumann_facets_ids_concat].ravel()
            )
            # neumann_vector = None
        else:
            neumann_nodes = None
            neumann_dir_type = None
            neumann_values = None
            # neumann_vector = None

        if robin_facets_ids is not None:
            robin_facets_ids_concat = np.concatenate(robin_facets_ids) \
                if isinstance(robin_facets_ids, list) else robin_facets_ids
            robin_nodes = np.unique(
                facets[:, robin_facets_ids_concat].ravel()
            )
            # robin_vector = None
        else:
            robin_nodes = None
            robin_coefficient = None
            robin_bc_value = None
            design_robin_boundary = None

        return cls.from_nodes(
            basis,
            dirichlet_nodes, dirichlet_dir, dirichlet_values,
            neumann_nodes,
            neumann_dir_type,
            neumann_values,
            # neumann_vector,
            robin_facets_ids,
            robin_nodes,
            robin_coefficient,
            robin_bc_value,
            design_robin_boundary,
            # robin_vector,
            design_elements
        )

    @classmethod
    def from_json(self, path: str):
        raise NotImplementedError("not implmented yet")

    @property
    def neumann_nodes_all(self) -> np.ndarray:
        if isinstance(self.neumann_nodes, list):
            return np.unique(np.concatenate(self.neumann_nodes))
        else:
            return self.neumann_nodes

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
        node_colors_df[self.neumann_nodes_all] = 1
        node_colors_df[self.dirichlet_nodes] = 2
        node_colors_df[self.robin_nodes] = 3
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
            np.sum(self.elements_volume[self.neumann_elements]),
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
