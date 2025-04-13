import pathlib
from dataclasses import dataclass
import numpy as np
import skfem
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import meshio
from scitopt import tools
from scitopt.mesh import utils
from scitopt.fea import composer


def setdiff1d(a, b):
    mask = ~np.isin(a, b)
    a = a[mask]
    return np.ascontiguousarray(a)


@dataclass
class TaskConfig():
    E0: float
    nu0: float
    Emin: float
    mesh: skfem.Mesh
    basis: skfem.Basis
    dirichlet_points: np.ndarray
    dirichlet_nodes: np.ndarray
    dirichlet_elements: np.ndarray
    dirichlet_adj_elements: np.ndarray
    force_points: np.ndarray | list[np.ndarray]
    force_nodes: np.ndarray | list[np.ndarray]
    force_elements: np.ndarray
    force: np.ndarray | list[np.ndarray]
    design_elements: np.ndarray
    free_nodes: np.ndarray
    free_elements: np.ndarray
    all_elements: np.ndarray
    fixed_elements_in_rho: np.ndarray
    dirichlet_force_elements: np.ndarray
    elements_volume: np.ndarray

    @classmethod
    def from_defaults(
        cls,
        E0: float,
        nu0: float,
        Emin: float,
        mesh: skfem.Mesh,
        basis: skfem.Basis,
        dirichlet_points: np.ndarray,
        dirichlet_nodes: np.ndarray,
        force_points: np.ndarray | list[np.ndarray],
        force_nodes: np.ndarray | list[np.ndarray],
        force_value: float | list[float],
        design_elements: np.ndarray,
    ) -> 'TaskConfig':
        dirichlet_elements = utils.get_elements_with_points_fast(
            mesh, [dirichlet_points]
        )
        adjacency = utils.build_element_adjacency_matrix_fast(mesh)
        # Elements that are next to boundary condition
        dirichlet_adj_elements = utils.get_adjacent_elements_fast(adjacency, dirichlet_elements)
        if isinstance(force_points, np.ndarray):
            force_elements = utils.get_elements_with_points_fast(
                mesh, [force_points]
            )
        else:
            force_elements = utils.get_elements_with_points_fast(
                mesh, force_points
            )
            
        # elements_related_with_bc = np.concatenate([bc_elements, dirichlet_adj_elements, force_elements])
        
        # design_elements = np.setdiff1d(design_elements, elements_related_with_bc)
        design_elements = setdiff1d(design_elements, force_elements)
        # design_elements = setdiff1d(design_elements, elements_related_with_bc)
        

        if len(design_elements) == 0:
            error_msg = "⚠️Warning: `design_elements` is empty"
            raise ValueError(error_msg)
        
        all_elements = np.arange(mesh.nelements)
        fixed_elements_in_rho = setdiff1d(all_elements, design_elements)
        dirichlet_force_elements = np.concatenate([dirichlet_elements, force_elements])
        print(
            f"all_elements: {all_elements.shape}",
            f"design_elements: {design_elements.shape}",
            f"fixed_elements_in_rho: {fixed_elements_in_rho.shape}",
            f"dirichlet_force_elements: {dirichlet_force_elements.shape}"
        )
        # free_nodes = np.setdiff1d(np.arange(basis.N), dirichlet_nodes)
        free_nodes = setdiff1d(np.arange(basis.N), dirichlet_nodes)
        free_elements = utils.get_elements_with_points_fast(mesh, [free_nodes])
        if isinstance(force_nodes, np.ndarray):
            if isinstance(force_value, (float, int)):
                force = np.zeros(basis.N)
                force[force_nodes] = force_value / len(force_nodes)
            elif isinstance(force_value, list):
                force = list()
                for fv in force_value:
                    print("fv", fv)
                    f_temp = np.zeros(basis.N)
                    f_temp[force_nodes] = fv / len(force_nodes)
                    force.append(f_temp)    
        elif isinstance(force_nodes, list):
            force = list()
            for fn_loop, fv in zip(force_nodes, force_value):
                f_temp = np.zeros(basis.N)
                f_temp[fn_loop] = fv / len(fn_loop)
                force.append(f_temp)
            

        elements_volume = composer.get_elements_volume(mesh)
        return cls(
            E0,
            nu0,
            Emin,
            mesh,
            basis,
            dirichlet_points,
            dirichlet_nodes,
            dirichlet_elements,
            dirichlet_adj_elements,
            force_points,
            force_nodes,
            force_elements,
            force,
            design_elements,
            free_nodes,
            free_elements,
            all_elements,
            fixed_elements_in_rho,
            dirichlet_force_elements,
            elements_volume
        )


    @property
    def dirichlet_and_adj_elements(self):
        return np.concatenate(
            [self.dirichlet_elements, self.dirichlet_adj_elements]
        )
        
    
    def exlude_dirichlet_from_design(self):
        self.design_elements = setdiff1d(
            self.design_elements, self.dirichlet_elements
        )
        
    def nodes_and_elements_stats(self, dst_path: str):
        node_points = self.mesh.p.T  # shape = (n_points, 3)
        tree_nodes = cKDTree(node_points)
        dists_node, _ = tree_nodes.query(node_points, k=2)
        node_nearest_dists = dists_node[:, 1]

        element_centers = np.mean(self.mesh.p[:, self.mesh.t], axis=1).T
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
            self.elements_volume[self.design_elements], bins=30, edgecolor='black'
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
            axs[1, 2].text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2g}', ha='center', va='bottom')

        axs[1, 2].set_title("THe volume difference elements")
        axs[1, 2].set_xlabel("Elements Attribute")
        axs[1, 2].set_ylabel("Volume")

        fig.tight_layout()
        fig.savefig(f"{dst_path}/info-nodes-elements.jpg")
        plt.close("all")

