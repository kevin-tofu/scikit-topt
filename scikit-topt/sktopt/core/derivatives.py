import os

import numpy as np


def _sensitivity_mode() -> str:
    raw_mode = os.environ.get("SCITOPT_SENSITIVITY_MODE", "current")
    mode = raw_mode.strip().lower()
    aliases = {
        "current": "current",
        "default": "current",
        "physical": "current",
        # backward-compatible alias: previous name for the current/default mode
        "scaled": "current",
    }
    if mode not in aliases:
        raise ValueError(
            "SCITOPT_SENSITIVITY_MODE must be one of "
            "'current', 'default', 'physical', or 'scaled', "
            f"got: {raw_mode}"
        )
    return aliases[mode]


def _E_simp(rho, E0, Emin, p):
    return Emin + (E0 - Emin) * np.maximum(rho, 1e-6) ** p


def _E_ramp(rho, E0, Emin, p):
    denom = 1.0 + p * (1.0 - rho)
    return Emin + (E0 - Emin) * (rho / denom)


def dE_drho_simp(rho, E0, Emin, p):
    """
    Derivative of SIMP-style E(rho)
    E(rho) = Emin + (E0 - Emin) * rho^p
    """
    return p * (E0 - Emin) * np.maximum(rho, 1e-6) ** (p - 1)


def dC_drho_simp(rho, strain_energy, E0, Emin, p):
    """
    Compliance sensitivity for SIMP.

    ``strain_energy`` is interpreted as the
    physical element strain energy ``U_e = 0.5 * u^T K(rho) u`` and the
    returned sensitivity is ``-(dE/drho) * (2 U_e / E)``.
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
