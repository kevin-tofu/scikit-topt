import pathlib
from dataclasses import dataclass
import numpy as np
import skfem
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import meshio
from scitopt import tools
from scitopt.mesh import utils



@dataclass
class TaskConfig():
    E0: float
    nu0: float
    Emin: float
    mesh: skfem.Mesh
    basis: skfem.Basis
    dirichlet_points: np.ndarray
    dirichlet_nodes: np.ndarray
    force_points: np.ndarray
    force_nodes: np.ndarray
    force: np.ndarray
    design_elements: np.ndarray
    free_nodes: np.ndarray
    free_elements: np.ndarray
    all_elements: np.ndarray
    fixed_elements_in_rho: np.ndarray
    

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
        force_points: np.ndarray,
        force_nodes: np.ndarray,
        force_value: float,
        design_elements: np.ndarray,
    ) -> 'TaskConfig':
        
        bc_elements = utils.get_elements_with_points(
            mesh, [dirichlet_points]
        )
        bc_elements_adj = utils.get_adjacent_elements(mesh, bc_elements)
        force_elements = utils.get_elements_with_points(
            mesh, [force_points]
        )
        elements_related_with_bc = np.concatenate([bc_elements, bc_elements_adj, force_elements])
        
        design_elements = np.setdiff1d(design_elements, elements_related_with_bc)
        if len(design_elements) == 0:
            error_msg = "⚠️Warning: `design_elements` is empty"
            raise ValueError(error_msg)
        
        all_elements = np.arange(mesh.nelements)
        fixed_elements_in_rho = np.setdiff1d(all_elements, design_elements)
        print(
            f"all_elements: {all_elements.shape}",
            f"design_elements: {design_elements.shape}",
            f"fixed_elements_in_rho: {fixed_elements_in_rho.shape}"
        )
        free_nodes = np.setdiff1d(np.arange(basis.N), dirichlet_nodes)
        free_elements = utils.get_elements_with_points(mesh, [free_nodes])
        F = np.zeros(basis.N)
        F[force_nodes] = force_value / len(force_nodes)

        return cls(
            E0,
            nu0,
            Emin,
            mesh,
            basis,
            dirichlet_points,
            dirichlet_nodes,
            force_points,
            force_nodes,
            F,
            design_elements,
            free_nodes,
            free_elements,
            all_elements,
            fixed_elements_in_rho
        )
        
    def nodes_stats(self, dst_path: str):
        points = self.mesh.p.T  # shape = (n_points, 3)
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        nearest_dists = dists[:, 1]  # shape = (n_points,)

        print(f"The minimum distance: {np.min(nearest_dists):.4f}")
        print(f"The maximum distance: {np.max(nearest_dists):.4f}")
        print(f"The average distance: {np.mean(nearest_dists):.4f}")
        print(f"The median distance: {np.median(nearest_dists):.4f}")
        print(f"The std distance: {np.std(nearest_dists):.4f}")

        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.hist(nearest_dists, bins=30, edgecolor='black')
        ax.set_xlabel("Distance from nearest node")
        ax.set_ylabel("Number of Nodes")
        ax.set_title("The histogram of nearest neighbors")
        ax.grid(True)

        fig.tight_layout() 
        fig.savefig(f"{dst_path}/nodes_stats.jpg")
        plt.close("all")

