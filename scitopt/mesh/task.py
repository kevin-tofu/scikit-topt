import pathlib
from dataclasses import dataclass
import numpy as np
import skfem
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import meshio
from scitopt import tools
from scitopt.mesh import utils


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
    force_points: np.ndarray | list[np.ndarray]
    force_nodes: np.ndarray | list[np.ndarray]
    force: np.ndarray | list[np.ndarray]
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
        force_points: np.ndarray | list[np.ndarray],
        force_nodes: np.ndarray | list[np.ndarray],
        force_value: float | list[float],
        design_elements: np.ndarray,
    ) -> 'TaskConfig':
        bc_elements = utils.get_elements_with_points_fast(
            mesh, [dirichlet_points]
        )
        adjacency = utils.build_element_adjacency_matrix_fast(mesh)
        # Elements that are next to boundary condition
        bc_elements_adj = utils.get_adjacent_elements_fast(adjacency, bc_elements)
        if isinstance(force_points, np.ndarray):
            force_elements = utils.get_elements_with_points_fast(
                mesh, [force_points]
            )
        else:
            force_elements = utils.get_elements_with_points_fast(
                mesh, force_points
            )
            
        elements_related_with_bc = np.concatenate([bc_elements, bc_elements_adj, force_elements])
        
        # design_elements = np.setdiff1d(design_elements, elements_related_with_bc)
        design_elements = setdiff1d(design_elements, force_elements)

        if len(design_elements) == 0:
            error_msg = "⚠️Warning: `design_elements` is empty"
            raise ValueError(error_msg)
        
        all_elements = np.arange(mesh.nelements)
        # fixed_elements_in_rho = np.setdiff1d(all_elements, design_elements)
        fixed_elements_in_rho = setdiff1d(all_elements, design_elements)
        print(
            f"all_elements: {all_elements.shape}",
            f"design_elements: {design_elements.shape}",
            f"fixed_elements_in_rho: {fixed_elements_in_rho.shape}"
        )
        # free_nodes = np.setdiff1d(np.arange(basis.N), dirichlet_nodes)
        free_nodes = setdiff1d(np.arange(basis.N), dirichlet_nodes)
        free_elements = utils.get_elements_with_points_fast(mesh, [free_nodes])
        if isinstance(force_nodes, np.ndarray):
            if isinstance(force_value, float):
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
            force,
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

