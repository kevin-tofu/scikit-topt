import numpy as np

import sktopt
from sktopt.filters.helmholtz_filter_nodal import (
    HelmholtzFilterNodal,
    element_to_node_density_averaging,
    node_to_element_density,
    solve_helmholtz,
)


def _uncached_forward(mesh, elements_volume, design_mask, radius, rho):
    rho_node = element_to_node_density_averaging(
        mesh,
        elements_volume,
        rho,
        design_mask=design_mask,
        fixed_value_for_design=1.0,
    )
    filtered_node = solve_helmholtz(
        "forward", mesh, rho_node, radius, design_mask=design_mask
    )
    return node_to_element_density(mesh, filtered_node)


def test_nodal_helmholtz_filter_cache_matches_uncached_forward_and_adjoint():
    task = sktopt.mesh.toy_problem.toy_test()
    mesh = task.mesh
    elements_volume = task.elements_volume
    design_mask = task.design_mask
    radius = 0.1
    filter_ = HelmholtzFilterNodal.from_defaults(
        mesh,
        elements_volume,
        radius=radius,
        design_mask=design_mask,
    )
    rng = np.random.default_rng(317)
    rho = rng.uniform(0.2, 0.9, mesh.nelements)
    direction = rng.normal(size=mesh.nelements)
    direction[~design_mask] = 0.0
    probe = rng.normal(size=mesh.nelements)

    actual = filter_.forward(rho)
    expected = _uncached_forward(
        mesh, elements_volume, design_mask, radius, rho
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)

    epsilon = 1e-6
    finite_difference = (
        np.dot(
            probe,
            filter_.forward(rho + epsilon * direction)
            - filter_.forward(rho - epsilon * direction),
        )
        / (2.0 * epsilon)
    )
    adjoint_product = np.dot(filter_.adjoint(probe), direction)
    np.testing.assert_allclose(
        adjoint_product, finite_difference, rtol=2e-5, atol=2e-7
    )

    new_radius = 0.07
    filter_.update_radius(new_radius)
    actual_after_update = filter_.forward(rho)
    expected_after_update = _uncached_forward(
        mesh, elements_volume, design_mask, new_radius, rho
    )
    np.testing.assert_allclose(
        actual_after_update,
        expected_after_update,
        rtol=1e-10,
        atol=1e-11,
    )
