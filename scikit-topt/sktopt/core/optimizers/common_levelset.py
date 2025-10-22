from typing import Callable

import numpy as np

import skfem
from skfem import *
from skfem.helpers import ddot, sym_grad, trace, eye, dot, identity, grad
from skfem.models.elasticity import linear_elasticity, lame_parameters



def get_shape_derivative_assembler(lam: float, mu: float) -> Callable:

    @skfem.LinearForm
    def shape_derivative_assembler(v, w):
        εu = sym_grad(w.u_interp)
        σ = lam * trace(εu) * identity(εu) + 2 * mu * εu
        energy = dot(εu, σ)
        energy = np.sum(energy, axis=0)
        φ = w.φ_interp.value
        φ_grad = grad(w.φ_interp)
        φ_grad_norm = np.linalg.norm(φ_grad, axis=0)

        q = w.q_interp.value
        c = energy * q * φ * φ_grad_norm
        return c * v

    return shape_derivative_assembler


def test():
    import skfem
    import sktopt

    x_len, y_len, z_len = 1.0, 1.0, 1.0
    element_size = 0.1
    e_v = skfem.ElementVector(skfem.ElementHex1())
    e_s = skfem.ElementHex1()
    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, element_size
    )
    basis_v = skfem.Basis(mesh, e_v, intorder=2)
    basis_s = skfem.Basis(mesh, e_s, intorder=2)

    def is_dirichlet_surface(x):
        return x[0] == 0.0

    def is_force_surface(x):
        return x[0] == 1.0

    # --- Material parameters ---
    E = 200e9
    nu = 0.3
    lam, mu = lame_parameters(E, nu)

    shape_derivative_assembler = get_shape_derivative_assembler(lam, mu)

    # --- Assemble stiffness matrix ---
    K = skfem.asm(linear_elasticity(lam, mu), basis_v)

    # --- Boundary conditions ---
    left = mesh.facets_satisfying(is_dirichlet_surface)
    D = basis_v.get_dofs(left)

    F_facet_ids = mesh.facets_satisfying(is_force_surface)
    fbasis = skfem.FacetBasis(mesh, e_v, facets=F_facet_ids)
    total_force = 100.0  # [N]

    @skfem.Functional
    def surface_measure(w):
        return 1.0
    A_proj_z = surface_measure.assemble(fbasis)
    pressure = total_force / A_proj_z

    @skfem.LinearForm
    def l_comp(v, w):
        return pressure * v[2]

    F = l_comp.assemble(fbasis)

    # --- Condense & solve (modern API) ---
    Kc, Fc, uc, I = skfem.condense(K, F, D=D)
    u = skfem.solve(Kc, Fc, uc, I)

    φ = np.zeros(u.shape[0] // 3)
    q = np.zeros(u.shape[0] // 3)
    params = dict(
        u_interp=basis_v.interpolate(u),
        φ_interp=basis_s.interpolate(φ),
        q_interp=basis_s.interpolate(q)
    )

    dJ = skfem.asm(shape_derivative_assembler, basis_s, **params)
    print(f"dJ: {dJ.shape}")


if __name__ == '__main__':
    test()