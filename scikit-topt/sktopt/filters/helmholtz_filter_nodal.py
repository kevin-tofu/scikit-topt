from typing import Optional, Literal
from dataclasses import dataclass
import logging

import numpy as np
from scipy.sparse.linalg import spsolve

from sktopt.skfem_loader import skfem
from sktopt.skfem_loader import dot, grad

from sktopt.filters.base import BaseFilter


logger = logging.getLogger(__name__)


def infer_element_from_mesh(mesh: skfem.Mesh):
    if isinstance(mesh, skfem.MeshTri):
        return skfem.ElementTriP1()
    elif isinstance(mesh, skfem.MeshQuad):
        return skfem.ElementQuad1()
    elif isinstance(mesh, skfem.MeshTet):
        return skfem.ElementTetP1()
    elif isinstance(mesh, skfem.MeshHex):
        return skfem.ElementHex1()
    else:
        raise ValueError(f"Unknown mesh type: {type(mesh)}")


def node_to_element_density(mesh, rho_node):
    return np.mean(rho_node[mesh.t], axis=0)


def element_to_node_density_averaging(
    mesh: skfem.Mesh,
    elements_volume: np.ndarray,
    rho_elem: np.ndarray,
    design_mask: np.ndarray | None = None,
    weighted: bool = True,
    fixed_value_for_design: float = 1.0
) -> np.ndarray:
    n_nodes = mesh.p.shape[1]
    rho_node = np.zeros(n_nodes)
    wsum = np.zeros(n_nodes)

    t = mesh.t.T

    if design_mask is None:
        design_mask = np.ones(len(t), dtype=bool)

    for e, nodes in enumerate(t):
        rho_val = rho_elem[e] if design_mask[e] else fixed_value_for_design
        # rho_val = rho_elem[e] if design_mask[e] else 0.0
        w = elements_volume[e] if weighted else 1.0
        rho_node[nodes] += w * rho_val
        wsum[nodes] += w

    wsum[wsum == 0.0] = 1.0
    rho_node /= wsum
    return rho_node


#
# Need to be fixed
#
# def project_element_density_to_nodes(mesh, rho_elem, design_mask=None):
#     basis = skfem.Basis(mesh, infer_element_from_mesh(mesh))
#     n_elem = mesh.t.shape[1]
#     if design_mask is None:
#         design_mask = np.ones(n_elem, dtype=bool)

#     @skfem.BilinearForm
#     def mass(u, v, w):
#         return dot(u, v)

#     @skfem.LinearForm
#     def rhs_design(v, w):
#         return rho_elem[w.idx] * v

#     @skfem.LinearForm
#     def rhs_fixed(v, w):
#         return 1.0 * v

#     M = skfem.asm(mass, basis)
#     F_design = skfem.asm(rhs_design, basis.with_elements(np.nonzero(design_mask)[0]))
#     F_fixed = skfem.asm(rhs_fixed, basis.with_elements(np.nonzero(~design_mask)[0]))

#     F = F_design + F_fixed
#     rho_node = spsolve(M, F)
#     return rho_node


# def solve_helmholtz(
#     case: Literal["forward", "gradient"],
#     mesh: skfem.Mesh, rho_node: np.ndarray,
#     r_min: float, design_mask: Optional[np.ndarray] = None,
# ):
#     const_value = 1.0 if case == "forward" else -1e-15
#     basis = skfem.Basis(mesh, infer_element_from_mesh(mesh))
#     elements = mesh.t.shape[1]

#     if design_mask is None:
#         design_mask = np.ones(elements, dtype=bool)

#     rho_field = basis.interpolate(rho_node)

#     @skfem.BilinearForm
#     def a(u, v, w):
#         return u * v + (r_min ** 2) * dot(grad(u), grad(v))

#     @skfem.LinearForm
#     def L(v, w):
#         return rho_field * v

#     A = skfem.asm(a, basis)
#     b = skfem.asm(L, basis)
#     fixed_nodes = np.unique(mesh.t[:, ~design_mask].ravel())
#     D = fixed_nodes
#     x0 = np.zeros(A.shape[0])
#     x0[fixed_nodes] = const_value
#     rho_filtered = skfem.solve(*skfem.condense(A, b, D=D, x=x0))
#     return rho_filtered


def solve_helmholtz(
    case: Literal["forward", "gradient"],
    mesh: skfem.Mesh,
    rho_node: np.ndarray,
    r_min: float,
    design_mask: Optional[np.ndarray] = None,
):
    basis = skfem.Basis(mesh, infer_element_from_mesh(mesh))
    if design_mask is None:
        fixed_nodes = np.array([], dtype=int)
    else:
        fixed_nodes = np.unique(mesh.t[:, ~design_mask].ravel())

    rho_field = basis.interpolate(rho_node)

    @skfem.BilinearForm
    def a(u, v, w):
        return u * v + (r_min ** 2) * dot(grad(u), grad(v))

    @skfem.LinearForm
    def L(v, w):
        return rho_field * v

    A = skfem.asm(a, basis)
    b = skfem.asm(L, basis)

    if len(fixed_nodes) > 0:
        D = fixed_nodes
        x0 = np.zeros(A.shape[0])
        x0[fixed_nodes] = 1.0 if case == "forward" else 0.0
        rho_filtered = skfem.solve(*skfem.condense(A, b, D=D, x=x0))
    else:
        # Neumann
        # print("Neumann")
        rho_filtered = skfem.solve(A, b)

    return rho_filtered



@dataclass
class HelmholtzFilterNodal(BaseFilter):

    def update_radius(
        self,
        radius: float,
        **args
    ):
        self.radius = radius

    @classmethod
    def from_defaults(
        cls,
        mesh: skfem.Mesh, elements_volume: np.ndarray,
        radius: float = 0.3,
        design_mask: Optional[np.ndarray] = None
    ) -> 'HelmholtzFilterNodal':
        return cls(mesh, elements_volume, radius, design_mask)

    def forward(self, rho_element: np.ndarray):
        rho_node = element_to_node_density_averaging(
            self.mesh, self.elements_volume, rho_element,
            design_mask=self.design_mask,
            fixed_value_for_design=1.0
        )
        # rho_node = project_element_density_to_nodes(
        #     self.mesh, rho_element,
        #     design_mask=self.design_mask
        # )
        rho_node_filtered = solve_helmholtz(
            "forward",
            self.mesh, rho_node, r_min=self.radius,
            design_mask=self.design_mask
        )
        rho_elem_filtered = node_to_element_density(
            self.mesh, rho_node_filtered
        )
        return rho_elem_filtered

    def gradient(self, v_ele: np.ndarray) -> np.ndarray:
        # print("np.sum(v_ele > 0):", np.sum(v_ele > 0))
        # element to node
        v_node = element_to_node_density_averaging(
            self.mesh,
            self.elements_volume,
            v_ele,
            design_mask=self.design_mask,
            weighted=True,
            fixed_value_for_design=0.0
        )
        # if element_to_node_density_averaging is proper, may not necessary.
        # print("np.sum(v_node > 0):", np.sum(v_node > 0))
        # v_node = np.minimum(v_node, 0.0)

        # Helmholtz PDE on nodes
        v_node_filtered = solve_helmholtz(
            "gradient",
            self.mesh,
            v_node,
            r_min=self.radius,
            design_mask=None
        )
        # print("np.sum(v_node_filtered > 0):", np.sum(v_node_filtered > 0))

        # node to element
        v_elem_filtered = node_to_element_density(
            self.mesh,
            v_node_filtered,
        )
        # print("np.sum(v_elem_filtered > 0):", np.sum(v_elem_filtered > 0))
        # print("v_elem_filtered:min/max", v_elem_filtered.min(), v_elem_filtered.max())
        v_elem_filtered = np.minimum(v_elem_filtered, 0.0)
        return v_elem_filtered

    def gradient_exact(self, v_ele: np.ndarray) -> np.ndarray:
        """Apply the transpose Jacobian of :meth:`forward` to ``v_ele``.

        ``gradient`` above is a sensitivity smoothing heuristic used by the
        legacy OC update.  MMA needs the derivative of the actual filtered
        density map, so this method applies the adjoint of each linear stage.
        """
        basis = skfem.Basis(self.mesh, infer_element_from_mesh(self.mesh))
        n_nodes = self.mesh.p.shape[1]
        elements = self.mesh.t.T
        values = np.asarray(v_ele, dtype=float)
        if values.shape != (elements.shape[0],):
            raise ValueError("v_ele must have one entry per mesh element")

        # Adjoint of node_to_element_density: each element stores the mean of
        # its nodal values.
        node_adjoint = np.zeros(n_nodes, dtype=float)
        np.add.at(node_adjoint, elements.ravel(), np.repeat(
            values / elements.shape[1], elements.shape[1]
        ))

        @skfem.BilinearForm
        def helmholtz(u, v, w):
            return u * v + (self.radius ** 2) * dot(grad(u), grad(v))

        @skfem.BilinearForm
        def mass(u, v, w):
            return u * v

        operator = skfem.asm(helmholtz, basis).tocsr()
        mass_matrix = skfem.asm(mass, basis).tocsr()
        if self.design_mask is None:
            fixed_nodes = np.array([], dtype=int)
        else:
            fixed_nodes = np.unique(
                self.mesh.t[:, ~self.design_mask].ravel()
            )
        free_nodes = np.setdiff1d(np.arange(n_nodes), fixed_nodes)
        if free_nodes.size == 0:
            return np.zeros_like(values)

        from scipy.sparse.linalg import spsolve

        adjoint_free = spsolve(
            operator[free_nodes][:, free_nodes].T,
            node_adjoint[free_nodes],
        )
        # The forward Helmholtz RHS is the consistent mass matrix times the
        # nodal input, including contributions from prescribed output nodes.
        input_node_adjoint = np.asarray(
            mass_matrix[free_nodes, :].T @ adjoint_free
        ).reshape(-1)

        # Adjoint of volume-weighted element_to_node_density_averaging.
        result = np.zeros_like(values)
        if self.design_mask is None:
            design_mask = np.ones(values.size, dtype=bool)
        else:
            design_mask = np.asarray(self.design_mask, dtype=bool)
        node_weight = np.zeros(n_nodes, dtype=float)
        np.add.at(
            node_weight,
            elements.ravel(),
            np.repeat(self.elements_volume, elements.shape[1]),
        )
        node_weight[node_weight == 0.0] = 1.0
        element_contribution = np.sum(
            input_node_adjoint[elements] / node_weight[elements], axis=1
        ) * self.elements_volume
        result[design_mask] = element_contribution[design_mask]
        return result


def test_main():

    import logging
    logging.getLogger("skfem").setLevel(logging.WARNING)
    import sktopt
    from sktopt.fea import composer
    x_len, y_len, z_len = 1.0, 1.0, 1.0
    element_size = 0.1
    e = skfem.ElementVector(skfem.ElementHex1())
    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, element_size
    )
    elements_volume = composer.get_elements_volume(mesh)

    filter_0 = HelmholtzFilterNodal.from_defaults(
        mesh, elements_volume=elements_volume, radius=0.04
    )
    filter_1 = HelmholtzFilterNodal.from_defaults(
        mesh, elements_volume=elements_volume, radius=0.08
    )
    rho = np.random.rand(mesh.t.shape[1])
    rho_0 = np.copy(rho)
    for loop in range(1, 21):
        rho_0 = filter_0.forward(rho_0)
        rho_var = np.var(rho_0)
        logger.info("loop: %s rho_var: %04f", loop, rho_var)

    rho_1 = np.copy(rho)
    for loop in range(1, 21):
        rho_1 = filter_1.forward(rho_1)
        rho_var = np.var(rho_1)
        logger.info("loop: %s rho_var: %04f", loop, rho_var)

    #
    # compare analytic gradient with numeric
    #
    eps = 1e-6
    v = np.random.rand(mesh.t.shape[1])
    v_grad = filter_0.gradient(v)

    # finite-diff check
    fwd1 = filter_0.forward(rho + eps * v)
    fwd2 = filter_0.forward(rho - eps * v)
    fd = (fwd1 - fwd2) / (2 * eps)

    logger.info("dot(fd, v) = %s", np.dot(fd, v))
    logger.info("dot(grad, v) = %s", np.dot(v_grad, v))


if __name__ == '__main__':
    test_main()
