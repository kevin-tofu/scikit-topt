from typing import Callable

import numpy as np

import skfem
from skfem import *
from skfem.helpers import ddot, sym_grad, trace, eye, dot, identity, grad
from skfem.models.elasticity import linear_elasticity, lame_parameters

import sktopt


def get_shape_derivative_assembler(lam: float, mu: float) -> Callable:

    # dJ(q,u,φ) = ∫((C ⊙ ε(u) ⊙ ε(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
    @skfem.LinearForm
    def shape_derivative_assembler(v, w):
        εu = sym_grad(w.u_interp)
        σ = lam * trace(εu) * identity(εu) + 2 * mu * εu
        energy = dot(εu, σ)
        energy = np.sum(energy, axis=0)
        # energy = np.einsum("ijxy,ijxy->xy", σ, εu)
        φ = w.φ_interp.value
        φ_grad = grad(w.φ_interp)
        φ_grad_norm = np.linalg.norm(φ_grad, axis=0)
        n = φ_grad / (φ_grad_norm + 1e-12)

        q = w.q_interp.value
        q_n = np.einsum("ixy,ixy->xy", q, n)
        # q_n = np.linalg.norm(q, axis=0)

        # The derivative of Heaviside for smoothing
        # beta = 0.1
        # DH = (1 / (np.pi * beta)) * (1 / (1 + (φ / beta) ** 2))
        DH = φ
        c = energy * q_n * DH * φ_grad_norm

        return c * v

    return shape_derivative_assembler


def get_dVol_assembler() -> Callable:

    # dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
    @skfem.LinearForm
    def dVol_assembler(v, w):
        φ_val = w.φ_interp.value                      # (nqp, nelems)
        φ_grad = grad(w.φ_interp)                     # (dim, nqp, nelems)
        φ_grad_norm = np.linalg.norm(φ_grad, axis=0)  # (nqp, nelems)
        n = φ_grad / (φ_grad_norm + 1e-12)            # (dim, nqp, nelems)

        q_val = w.q_interp.value                      # (dim, nqp, nelems)
        q_n = np.einsum("ixy,ixy->xy", q_val, n)      # q·n → (nqp, nelems)

        # The derivative of Heaviside for smoothing
        beta = 0.1
        DH = (1 / (np.pi * beta)) * (1 / (1 + (φ_val / beta)**2))  # (nqp, nelems)

        # integrand
        c = -1.0 / w.vol * q_n * DH * φ_grad_norm    # (nqp, nelems)
        return c * v                                 # LinearForm expects (nqp, nelems)


    return dVol_assembler


def heaviside(phi, beta=0.1):
    return 0.5 * (1 + (2 / np.pi) * np.arctan(phi / beta))


def test():

    # J(u,φ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(u)))dΩ
    # dJ(q,u,φ) = ∫((C ⊙ ε(u) ⊙ ε(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
    # Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
    # dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
    x_len, y_len, z_len = 8.0, 1.0, 1.0
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
        return x[0] == x_len

    # --- Material parameters ---
    E = 200e9
    nu = 0.3
    lam, mu = lame_parameters(E, nu)

    shape_derivative_assembler = get_shape_derivative_assembler(lam, mu)
    dVol_assembler = get_dVol_assembler()

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
        return pressure * v[0]

    F = l_comp.assemble(fbasis)

    Kc, Fc, uc, I = skfem.condense(K, F, D=D)
    u = skfem.solve(Kc, Fc, uc, I)

    q = -u.copy()

    φ = np.zeros(u.shape[0] // 3)
    params = dict(
        u_interp=basis_v.interpolate(u),
        φ_interp=basis_s.interpolate(φ),
        q_interp=basis_v.interpolate(q),
        vol=100.0
    )

    dJ = skfem.asm(shape_derivative_assembler, basis_s, **params)
    print(f"dJ: {dJ.shape}")
    dVol = skfem.asm(dVol_assembler, basis_s, **params)
    print(f"dVol: {dVol.shape}")


if __name__ == '__main__':
    test()
