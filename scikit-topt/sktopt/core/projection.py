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


def heaviside_projection_inplace(rho, beta, eta=0.5, out=None):
    r"""
    In-place Heaviside projection.

    Applies the smoothed Heaviside projection to ``rho``:

    .. math::

       \tilde{\rho} = \frac{\tanh(\beta \eta) + \tanh(\beta (\rho - \eta))}
                           {\tanh(\beta \eta) + \tanh(\beta (1 - \eta))}

    Parameters
    ----------
    rho : ndarray
        Input density array in-place (modified) unless ``out`` is given.
    beta : float
        Sharpness parameter of the Heaviside.
    eta : float
        Threshold of the Heaviside (0..1).
    out : ndarray or None, optional
        If provided, the projection is written to ``out`` instead of ``rho``.

    Returns
    -------
    ndarray
        The projected array (``rho`` itself if ``out`` is ``None``).

    Notes
    -----
    - Ensure ``rho`` is clipped to ``[0, 1]`` before projection.
    - This implementation avoids temporary arrays when ``out`` is ``None``.
    """
    if out is None:
        out = np.empty_like(rho)

    # temp buffer for tanh(beta * (rho - eta))
    tanh_term = np.empty_like(rho)

    # numerator = tanh(beta * eta) + tanh(beta * (rho - eta))
    np.subtract(rho, eta, out=tanh_term)
    tanh_term *= beta
    np.tanh(tanh_term, out=tanh_term)

    numerator = np.tanh(beta * eta)
    np.add(numerator, tanh_term, out=out)

    # denominator = tanh(beta * eta) + tanh(beta * (1 - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta)) + 1e-12

    out /= denominator
    return out


def heaviside_projection_derivative_inplace(rho, beta, eta=0.5, out=None):
    """
    In-place computation of the derivative of the smooth Heaviside projection.

    Parameters
    ----------
    rho : ndarray
        Input density array.
    beta : float
        Sharpness parameter of the projection.
    eta : float
        Threshold parameter.
    out : ndarray or None
        If provided, result will be written here.
        Otherwise, a new array is allocated.

    Returns
    -------
    out : ndarray
        Derivative of the projection at each point in rho.
    """
    if out is None:
        out = np.empty_like(rho)

    # temp = beta * (rho - eta)
    np.subtract(rho, eta, out=out)
    out *= beta

    # sech²(x) = 1 / cosh²(x)
    np.cosh(out, out=out)
    np.square(out, out=out)
    np.reciprocal(out, out=out)  # now out = sech²

    out *= beta

    denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta)) + 1e-12
    out /= denom

    return out
