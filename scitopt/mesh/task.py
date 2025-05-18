from dataclasses import dataclass
import numpy as np
import skfem
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scitopt.mesh import utils
from scitopt.fea import composer


def setdiff1d(a, b):
    mask = ~np.isin(a, b)
    a = a[mask]
    return np.ascontiguousarray(a)


@dataclass
class TaskConfig():
    E: float
    nu: float
    basis: skfem.Basis
    # dirichlet_points: np.ndarray
    dirichlet_dofs: np.ndarray
    dirichlet_elements: np.ndarray
    # force_points: np.ndarray | list[np.ndarray]
    force_dofs: np.ndarray | list[np.ndarray]
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
    def from_defaults(
        cls,
        E: float,
        nu: float,
        basis: skfem.Basis,
        dirichlet_points: np.ndarray,
        dirichlet_dofs: np.ndarray,
        force_points: np.ndarray | list[np.ndarray],
        force_dofs: np.ndarray | list[np.ndarray],
        force_value: float | list[float],
        design_elements: np.ndarray,
    ) -> 'TaskConfig':
        #
        # Dirichlet
        #
        dirichlet_elements = utils.get_elements_with_points_fast(
            basis.mesh, [dirichlet_points]
        )
        #
        # Force
        #
        if isinstance(force_points, np.ndarray):
            force_elements = utils.get_elements_with_points_fast(
                basis.mesh, [force_points]
            )
        else:
            force_elements = utils.get_elements_with_points_fast(
                basis.mesh, force_points
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
        free_elements = utils.get_elements_with_points_fast(
            basis.mesh, [free_dofs]
        )
        if isinstance(force_dofs, np.ndarray):
            if isinstance(force_value, (float, int)):
                force = np.zeros(basis.N)
                force[force_dofs] = force_value / len(force_dofs)
            elif isinstance(force_value, list):
                force = list()
                for fv in force_value:
                    print("fv", fv)
                    f_temp = np.zeros(basis.N)
                    f_temp[force_dofs] = fv / len(force_dofs)
                    force.append(f_temp)
        elif isinstance(force_dofs, list):
            force = list()
            for fn_loop, fv in zip(force_dofs, force_value):
                f_temp = np.zeros(basis.N)
                f_temp[fn_loop] = fv / len(fn_loop)
                force.append(f_temp)

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
            dirichlet_dofs,
            dirichlet_elements,
            force_dofs,
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
