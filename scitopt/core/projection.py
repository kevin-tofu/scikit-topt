import numpy as np


def heaviside_projection(rho, beta, eta=0.5):
    """
    Smooth Heaviside projection.
    Returns projected density in [0,1].
    """
    numerator = np.tanh(beta * eta) + np.tanh(beta * (rho - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return numerator / (denominator + 1e-12)


def heaviside_projection_derivative(rho, beta, eta=0.5):
    """
    Derivative of the smooth Heaviside projection w.r.t. rho.
    d/d(rho)[heaviside_projection(rho)].
    """
    # y = tanh( beta*(rho-eta) )
    # dy/d(rho) = beta*sech^2( ... )
    # factor from normalization
    numerator = beta * (1.0 - np.tanh(beta*(rho - eta))**2)
    denominator = np.tanh(beta*eta) + np.tanh(beta*(1.0 - eta))
    return numerator / (denominator + 1e-12)
