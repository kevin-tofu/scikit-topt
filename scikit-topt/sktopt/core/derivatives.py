import numpy as np


def dE_drho_simp(rho, E0, Emin, p):
    """
    Derivative of SIMP-style E(rho)
    E(rho) = Emin + (E0 - Emin) * rho^p
    """
    return p * (E0 - Emin) * np.maximum(rho, 1e-6) ** (p - 1)


def _E_simp(rho, E0, Emin, p):
    """
    SIMP interpolation evaluated at rho.
    """
    return Emin + (E0 - Emin) * np.maximum(rho, 1e-6) ** p


def dC_drho_simp(rho, strain_energy, E0, Emin, p):
    """
    Compliance sensitivity for SIMP, assuming ``strain_energy`` is the
    physical strain energy computed with the penalized stiffness
    (i.e., ``0.5 * u^T K(ρ) u``).

    Since ``K(ρ)`` scales linearly with ``E(ρ)``, the adjoint term
    ``u^T dK/drho u`` becomes ``(dE/drho / E(ρ)) * u^T K(ρ) u``.
    Here ``u^T K(ρ) u = 2 * strain_energy``, hence the extra factor.
    """
    dE_drho = dE_drho_simp(rho, E0, Emin, p)
    E_elem = _E_simp(rho, E0, Emin, p)
    return -2.0 * strain_energy * dE_drho / np.maximum(E_elem, 1e-12)


# def dE_drho_rationalSIMP(rho, E0, Emin, p):
def dE_drho_ramp(rho, E0, Emin, p):
    """
    Derivative of Rational SIMP-style E(rho)
    E(rho) = Emin + (E0 - Emin) * rho / (1 + p * (1 - rho))
    """
    denom = 1.0 + p * (1.0 - rho)
    return (E0 - Emin) * (denom - p * rho) / (denom ** 2)


def _E_ramp(rho, E0, Emin, p):
    denom = 1.0 + p * (1.0 - rho)
    return Emin + (E0 - Emin) * (rho / denom)


def dC_drho_ramp(rho, strain_energy, E0, Emin, p):
    dE_drho = dE_drho_ramp(rho, E0, Emin, p)
    E_elem = _E_ramp(rho, E0, Emin, p)
    return -2.0 * strain_energy * dE_drho / np.maximum(E_elem, 1e-12)


def dE_drho_ramp_inplace(rho, out, E0, Emin, p):
    """
    In-place version of dE_drho_ramp.
    Computes the derivative of E(rho) and stores in `out`.
    """
    np.copyto(out, rho)
    denom = 1.0 + p * (1.0 - rho)
    np.copyto(out, (E0 - Emin) * (denom - p * rho) / (denom ** 2))


def dC_drho_ramp_inplace(rho, strain_energy, out, E0, Emin, p):
    """
    In-place version of dC_drho_ramp.
    Computes the derivative of compliance and stores in `out`.
    """
    dE_drho_ramp_inplace(rho, out, E0, Emin, p)
    E_elem = _E_ramp(rho, E0, Emin, p)
    np.divide(out, np.maximum(E_elem, 1e-12), out=out)
    out *= -2.0 * strain_energy
