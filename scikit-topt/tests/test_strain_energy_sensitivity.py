import numpy as np

from sktopt.core import derivatives as d


def _expected_simp(rho, strain_energy, E0, Emin, p):
    E = Emin + (E0 - Emin) * np.maximum(rho, 1e-6) ** p
    dE = d.dE_drho_simp(rho, E0, Emin, p)
    return -2.0 * strain_energy * dE / np.maximum(E, 1e-12)


def _expected_ramp(rho, strain_energy, E0, Emin, p):
    denom = 1.0 + p * (1.0 - rho)
    E = Emin + (E0 - Emin) * (rho / denom)
    dE = (E0 - Emin) * (denom - p * rho) / (denom ** 2)
    return -2.0 * strain_energy * dE / np.maximum(E, 1e-12)


def test_simp_compliance_gradient_scales_physical_strain_energy():
    rho = np.array([0.25, 0.5, 0.9])
    strain_energy = np.array([0.1, 1.2, 3.4])
    E0, Emin, p = 1.0, 1e-3, 3.0

    expected = _expected_simp(rho, strain_energy, E0, Emin, p)
    got = d.dC_drho_simp(rho, strain_energy, E0, Emin, p)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_ramp_inplace_matches_scaled_strain_energy():
    rho = np.array([0.3, 0.6, 0.95])
    strain_energy = np.array([0.2, 0.8, 2.5])
    E0, Emin, p = 1.0, 1e-4, 4.0

    expected = _expected_ramp(rho, strain_energy, E0, Emin, p)
    out = np.empty_like(rho)

    d.dC_drho_ramp_inplace(rho, strain_energy, out, E0, Emin, p)

    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_heat_compliance_uses_half_k_gradT_sq_energy():
    rho = np.array([0.2, 0.5, 0.8])
    gradT_sq = np.array([1.0, 4.0, 9.0])  # |∇T|^2 per element
    k0, kmin, p = 5.0, 0.01, 3.0

    k_elem = kmin + (k0 - kmin) * np.maximum(rho, 1e-6) ** p
    energy = 0.5 * k_elem * gradT_sq  # matches solver_heat._heat_energy_density_

    expected = -d.dE_drho_simp(rho, k0, kmin, p) * gradT_sq
    got = d.dC_drho_simp(rho, energy, k0, kmin, p)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
