from typing import Optional, Literal
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve

import skfem
from skfem.helpers import dot, grad

from sktopt.filters.base import BaseFilter


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
        print(f"loop: {loop} rho_var: {rho_var:04f}")

    rho_1 = np.copy(rho)
    for loop in range(1, 21):
        rho_1 = filter_1.forward(rho_1)
        rho_var = np.var(rho_1)
        print(f"loop: {loop} rho_var: {rho_var:04f}")

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

    print("dot(fd, v) =", np.dot(fd, v))
    print("dot(grad, v) =", np.dot(v_grad, v))


if __name__ == '__main__':
    test_main()
