import numpy as np


def dE_drho_simp(rho, E0, Emin, p):
    """
    Derivative of SIMP-style E(rho)
    E(rho) = Emin + (E0 - Emin) * rho^p
    """
    return p * (E0 - Emin) * np.maximum(rho, 1e-6) ** (p - 1)


def dC_drho_simp(rho, strain_energy, E0, Emin, p):
    """
    dC/drho = -p * (E0 - Emin) * rho^(p - 1) * strain_energy
    """
    # dE_drho = p * (E0 - Emin) * np.maximum(rho, 1e-6) ** (p - 1)
    dE_drho = dE_drho_simp(rho, E0, Emin, p)
    return - dE_drho * strain_energy


# def dE_drho_rationalSIMP(rho, E0, Emin, p):
def dE_drho_ramp(rho, E0, Emin, p):
    """
    Derivative of Rational SIMP-style E(rho)
    E(rho) = Emin + (E0 - Emin) * rho / (1 + p * (1 - rho))
    """
    denom = 1.0 + p * (1.0 - rho)
    return (E0 - Emin) * (denom - p * rho) / (denom ** 2)


def dC_drho_ramp(rho, strain_energy, E0, Emin, p):
    dE_drho = dE_drho_ramp(rho, E0, Emin, p)
    return - dE_drho * strain_energy


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
    out *= -strain_energy
