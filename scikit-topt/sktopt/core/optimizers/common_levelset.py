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


def assemble_stiffness_matrix_with_φ(
    basis_v: skfem.Basis,
    basis_s: skfem.Basis,
    φ: np.ndarray,
    Emax: float, Emin: float, nu: float
):
    print("φ.shape =", φ.shape)
    print("basis.N =", basis_s.N)
    print("mesh.p.shape =", basis_s.mesh.p.shape)

    φ_elem = basis_s.interpolate(φ).value
    E_elem = Emin + (Emax - Emin) * heaviside(φ_elem)
    lam = (nu * E_elem) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_elem / (2.0 * (1.0 + nu))

    # 3. BilinearForm
    @skfem.BilinearForm
    def stiffness_form(u, v, w):
        strain_u = sym_grad(u)
        strain_v = sym_grad(v)
        volume_term = lam * trace(strain_u) * trace(strain_v)
        return volume_term + 2.0 * mu * ddot(strain_u, strain_v)

    K = skfem.asm(stiffness_form, basis_v)
    return K


def test_fem(
    mesh: skfem.Mesh,
    basis_v: skfem.Basis, basis_s: skfem.Basis,
    is_dirichlet_surface: Callable,
    is_force_surface: Callable,
    φ: np.ndarray,
    Emax=200e9,
    Emin=200e4,
    nu=0.3,
    F_total=100.0
):
    # --- Assemble stiffness matrix ---
    # lam, mu = lame_parameters(E, nu)
    # K = skfem.asm(linear_elasticity(lam, mu), basis_v)
    K = assemble_stiffness_matrix_with_φ(basis_v, basis_s, φ, Emax, Emin, nu)

    # --- Boundary conditions ---
    left = mesh.facets_satisfying(is_dirichlet_surface)
    D = basis_v.get_dofs(left)

    F_facet_ids = mesh.facets_satisfying(is_force_surface)
    fbasis = skfem.FacetBasis(mesh, e_v, facets=F_facet_ids)

    @skfem.Functional
    def surface_measure(w):
        return 1.0
    A_proj_z = surface_measure.assemble(fbasis)
    pressure = F_total / A_proj_z

    @skfem.LinearForm
    def l_comp(v, w):
        return pressure * v[2]

    F = l_comp.assemble(fbasis)

    Kc, Fc, uc, I = skfem.condense(K, F, D=D)
    u = skfem.solve(Kc, Fc, uc, I)
    return u


def test_assembling(E, nu, basis_v, basis_s, u, q, φ):

    # J(u,φ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(u)))dΩ
    # dJ(q,u,φ) = ∫((C ⊙ ε(u) ⊙ ε(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
    # Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
    # dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

    @skfem.LinearForm
    def liner_v(v, w):
        return v

    @skfem.LinearForm
    def linar_φ_grad_norm(v, w):
        φ_grad_norm = np.linalg.norm(grad(w.φ_interp), axis=0)
        return φ_grad_norm * v

    L_base = skfem.asm(liner_v, basis_s)
    vol = L_base * heaviside(φ)
    # vol = L_base * np.ones_like(φ)

    lam, mu = lame_parameters(E, nu)
    shape_derivative_assembler = get_shape_derivative_assembler(lam, mu)
    dVol_assembler = get_dVol_assembler()
    params = dict(
        u_interp=basis_v.interpolate(u),
        φ_interp=basis_s.interpolate(φ),
        q_interp=basis_v.interpolate(q),
        vol=vol
    )

    dJ = skfem.asm(shape_derivative_assembler, basis_s, **params)
    dVol = skfem.asm(dVol_assembler, basis_s, **params)
    φ_grad_norm = skfem.asm(linar_φ_grad_norm, basis_s, **params)
    # print(f"dJ: {dJ.shape}")
    # print(f"dVol: {dVol.shape}")
    # print(f"φ_grad_norm: {φ_grad_norm.shape}")
    # print(f"vol: {vol.shape} {np.sum(vol)}")

    return dJ, dVol, φ_grad_norm


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
        φ_grad = grad(w.φ)

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


def test_ReInit_WENO_like(basis_s, φ, φ0, eps: float = 1e-4):

    # ∂τ∂ϕ​=sign(ϕ0​)(1−∣∇ϕ∣)+∇⋅(D(∇ϕ)∇ϕ)
    alpha = 1e-3
    eps = 1e-6

    @skfem.BilinearForm
    def a_weno_like(u, v, w):
        D = alpha / (1.0 + np.sum(w.φ_grad_norm**2, axis=0))
        return D * dot(grad(u), grad(v))

    @skfem.LinearForm
    def rhs_reinit_weno(v, w):
        return w.sign_φ0 * (1.0 - w.φ_grad_norm) * v

    @skfem.LinearForm
    def linar_φ_grad_norm(v, w):
        φ_grad_norm = np.linalg.norm(grad(w.φ), axis=0)
        return φ_grad_norm * v

    φ_grad_norm = skfem.asm(
        linar_φ_grad_norm, basis_s, φ=basis_s.interpolate(φ)
    )
    sign_φ0 = φ0 / np.sqrt(φ0**2 + eps**2)

    K = skfem.asm(a_weno_like, basis_s, φ_grad_norm=φ_grad_norm)
    M = skfem.asm(lambda u, v, w: u*v, basis_s)
    rhs = skfem.asm(
        rhs_reinit_weno, basis_s, sign_φ0=sign_φ0, grad_norm=φ_grad_norm
    )

    dt = 1e-3
    A = M + dt * K
    b = M @ φ + dt * rhs
    phi_new = skfem.solve(A, b)
    return phi_new


def HJ_RK4_rhs(
    Emax, nu, basis_v, basis_s, λ, u, q, φ
):

    dJ, dVol, φ_grad_norm = test_assembling(
        Emax, nu, basis_v, basis_s, u, q, φ
    )
    velocity = - (dJ + λ * dVol * φ_grad_norm)
    return velocity


def HJ_RK4(
    Emax, nu, basis_v, basis_s, λ, u, q,
    φ,
    dt
):
    k1 = HJ_RK4_rhs(
        Emax, nu, basis_v, basis_s, λ, u, q,
        φ
    )
    k2 = HJ_RK4_rhs(
        Emax, nu, basis_v, basis_s, λ, u, q,
        φ - 0.5 * dt * k1
    )
    k3 = HJ_RK4_rhs(
        Emax, nu, basis_v, basis_s, λ, u, q,
        φ - 0.5 * dt * k2
    )
    k4 = HJ_RK4_rhs(
        Emax, nu, basis_v, basis_s, λ, u, q,
        φ - dt * k3
    )
    return φ - dt / 6 * (k1 + 2*k2 + 2*k3 + k4)


def REINIT_RK4_rhs(basis_s, φ0, φ, eps: float = 1e-4):
    grad_φ = basis_s.interpolate(φ).grad
    φ_grad_norm = np.sqrt(np.sum(grad_φ**2, axis=0))
    sign_φ0 = φ0 / np.sqrt(φ0**2 + eps**2)  # smooth sign
    return - sign_φ0 * (φ_grad_norm - 1.0)


def REINIT_RK4(
    basis_s, φ0, φ, dt,
    max_iters: int = 5
):
    for loop in range(max_iters):
        k1 = REINIT_RK4_rhs(
            basis_s, φ0,
            φ
        )
        k2 = REINIT_RK4_rhs(
            basis_s, φ0,
            φ - 0.5 * dt * k1
        )
        k3 = REINIT_RK4_rhs(
            basis_s, φ0,
            φ - 0.5 * dt * k2
        )
        k4 = REINIT_RK4_rhs(
            basis_s, φ0,
            φ - dt * k3
        )
        φ = φ - dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    return φ


if __name__ == '__main__':

    x_len, y_len, z_len = 8.0, 1.0, 1.0
    element_size = 0.1
    e_v = skfem.ElementVector(skfem.ElementHex1())
    e_s = skfem.ElementHex1()
    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, element_size
    )
    Emax = 210e9
    Emin = Emax * 1e-4
    nu = 0.3
    basis_v = skfem.Basis(mesh, e_v, intorder=2)
    basis_s = skfem.Basis(mesh, e_s, intorder=2)

    def is_dirichlet_surface(x):
        return x[0] == 0.0

    def is_force_surface(x):
        return x[0] == x_len

    φ = np.ones(mesh.p.shape[1])

    u = test_fem(
        mesh, basis_v, basis_s,
        is_dirichlet_surface,
        is_force_surface,
        φ,
        Emax=Emax, Emin=Emin, nu=nu
    )
    q = -u.copy()

    # dJ, dVol, φ_grad_norm = test_assembling(
    #     Emax, nu, basis_v, basis_s, u, q, φ
    # )

    λ = 0.0
    dt = 1e-5
    φ_updated = HJ_RK4(
        Emax, nu, basis_v, basis_s, λ, u, q,
        φ,
        dt
    )

    φ0 = φ.copy()
    φ_updated = REINIT_RK4(
        Emax, nu, basis_v, basis_s, λ, u, q,
        φ0,
        φ,
        dt
    )
