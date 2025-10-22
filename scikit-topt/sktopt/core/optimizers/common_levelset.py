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


def test_fem():
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
    return lam, mu, mesh, basis_v, basis_s, u


def test_assembling(lam, mu, mesh, basis_v, basis_s, u, φ, q):

    # J(u,φ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(u)))dΩ
    # dJ(q,u,φ) = ∫((C ⊙ ε(u) ⊙ ε(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
    # Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
    # dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
    shape_derivative_assembler = get_shape_derivative_assembler(lam, mu)
    dVol_assembler = get_dVol_assembler()

    
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


def test_HJ_WENO_like(basis_s, φ, V):

    # add non-linear diffusion on Hamilton-Jacobi
    # ∂t∂ϕ​+V∣∇ϕ∣ = ∇⋅(D(∇ϕ)∇ϕ)
    # D(∇ϕ)=α/(1+∣∇ϕ∣^2)
    # (M+ΔtK(D(∇φ)))φn+1=Mφn−ΔtM(V∣∇φ∣)

    alpha = 0.1

    @skfem.BilinearForm
    def a_weno_like(u, v, w):
        φ_grad = grad(w.φ)
        D = alpha / (1.0 + np.sum(φ_grad**2, axis=0))
        return D * np.einsum("ijk,ijk->jk", grad(u), grad(v))


    @skfem.LinearForm
    def l_weno_like(v, w):
        # φ_grad = grad(w.φ)
        φ_grad = w.φ.grad

        φ_grad_norm = np.linalg.norm(φ_grad, axis=0)
        return (-w.V * φ_grad_norm) * v

    K = skfem.asm(a_weno_like, basis_s, φ=basis_s.interpolate(φ))
    M = skfem.asm(lambda u, v, w: u*v, basis_s)
    rhs = skfem.asm(l_weno_like, basis_s, V=V, φ=basis_s.interpolate(φ))

    dt = 1e-4
    A = M + dt*K
    b = M @ φ - dt*rhs
    φ_new = skfem.solve(A, b)
    return φ_new


def test_ReInit_WENO_like(basis_s, φ):

    # ∂τ∂ϕ​=sign(ϕ0​)(1−∣∇ϕ∣)+∇⋅(D(∇ϕ)∇ϕ)
    alpha = 1e-3
    eps = 1e-6
    φ0 = φ.copy()

    @skfem.BilinearForm
    def a_weno_like(u, v, w):
        grad_phi = w["grad_phi"]
        D = alpha / (1.0 + np.sum(grad_phi**2, axis=0))
        return D * dot(grad(u), grad(v))

    @skfem.LinearForm
    def rhs_reinit_weno(v, w):
        return w["sign_phi0"] * (1.0 - w["grad_norm"]) * v

    grad_φ = basis_s.interpolate(φ).grad
    grad_norm = np.sqrt(np.sum(grad_φ**2, axis=0))
    sign_phi0 = φ0 / np.sqrt(φ0**2 + φ0**2)

    K = skfem.asm(a_weno_like, basis_s, grad_phi=grad_φ)
    M = skfem.asm(lambda u, v, w: u*v, basis_s)
    rhs = skfem.asm(
        rhs_reinit_weno, basis_s, sign_phi0=sign_phi0, grad_norm=grad_norm
    )

    dt = 1e-3
    A = M + dt * K
    b = M @ φ + dt * rhs
    phi_new = skfem.solve(A, b)
    return phi_new


if __name__ == '__main__':
    
    lam, mu, mesh, basis_v, basis_s, u = test_fem()
    φ = np.zeros(u.shape[0] // 3)
    q = -u.copy()
    # V = np.ones_like(u)
    # V = np.ones(basis_v.mesh.t.shape[1])
    V = np.ones(u.shape[0] // 3)

    test_assembling(lam, mu, mesh, basis_v, basis_s, u, φ, q)
    test_HJ_WENO_like(basis_s, φ, V)
    test_ReInit_WENO_like(basis_s, φ)
