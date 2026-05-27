"""Utility functions used across :mod:`gpyrn`."""

from functools import wraps
from random import shuffle
from typing import Any, Callable, TypeVar, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from jax.numpy import ndarray as jnp_ndarray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import invgamma

Array = Union[np.ndarray, jnp_ndarray]
F = TypeVar("F", bound=Callable[..., Any])


def _array_input(f: F) -> F:
    """Convert the decorated function's first data argument to an array."""

    @wraps(f)
    def wrapped(self: Any, t: Array | float) -> Any:
        t = np.atleast_1d(t)
        r = f(self, t)
        return r

    return cast(F, wrapped)


def semi_amplitude(
    period: float,
    Mplanet: float,
    Mstar: float,
    ecc: float,
) -> float:
    """Calculate the radial-velocity semi-amplitude caused by a planet.

    Args:
        period: Period in years.
        Mplanet: Planet mass in Jupiter masses, technically M sin i.
        Mstar: Star mass in Solar masses.
        ecc: Eccentricity between 0 and 1.

    Returns:
        Semi-amplitude K in m/s.
    """
    per = float(np.power(1 / period, 1 / 3))
    Pmass = Mplanet / 1
    Smass = float(np.power(1 / Mstar, 2 / 3))
    Ecc = 1 / np.sqrt(1 - ecc**2)
    return 28.435 * per * Pmass * Smass * Ecc


def keplerian(
    P: float = 365,
    K: float = 0.1,
    e: float = 0,
    w: float = np.pi,
    T: float = 0,
    phi: float | None = None,
    gamma: float = 0,
    t: Array | None = None,
) -> tuple[Array, list[float]]:
    """Simulate a radial-velocity signal for one Keplerian orbit.

    Args:
        P: Period in days.
        K: Radial-velocity semi-amplitude.
        e: Eccentricity.
        w: Longitude of periastron.
        T: Zero phase.
        phi: Orbital phase. When provided, it is used to derive ``T``.
        gamma: Constant systemic radial velocity.
        t: Times at which to evaluate the signal.

    Returns:
        The input times and the generated radial-velocity signal.
    """
    if t is None:
        raise ValueError("t must be provided")
    # mean anomaly
    if phi is None:
        mean_anom = [2 * np.pi * (x1 - T) / P for x1 in t]
    else:
        T = t[0] - (P * phi) / (2.0 * np.pi)
        mean_anom = [2 * np.pi * (x1 - T) / P for x1 in t]
    # eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
    E0 = [x + e * np.sin(x) + 0.5 * (e**2) * np.sin(2 * x) for x in mean_anom]
    # mean anomaly -> M0=E0 - e*sin(E0)
    M0 = [x - e * np.sin(x) for x in E0]
    i = 0
    while i < 1000:
        # [x + y for x, y in zip(first, second)]
        calc_aux = [x2 - y for x2, y in zip(mean_anom, M0)]
        E1 = [x3 + y / (1 - e * np.cos(x3)) for x3, y in zip(E0, calc_aux)]
        M1 = [x4 - e * np.sin(x4) for x4 in E0]
        i += 1
        E0 = E1
        M0 = M1
    nu = [2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(x5 / 2)) for x5 in E0]
    RV = [gamma + K * (e * np.cos(w) + np.cos(w + x6)) for x6 in nu]  # m/s
    return t, RV


def phase_fold(t: Array, y: Array, yerr: Array | None, period: float):
    """Fold observations by a trial period.

    Args:
        t: Observation times.
        y: Measurements.
        yerr: Measurement uncertainties. If ``None``, zero uncertainties are
            returned.
        period: Period used to fold the observations.

    Returns:
        A tuple containing the sorted phase, measurements, and uncertainties.
    """
    # divide the time by the period to convert to phase
    foldtimes = t / period
    # remove the whole number part of the phase
    foldtimes = foldtimes % 1
    if yerr is None:
        yerr = 0 * y
    # sort everything
    phase, folded_y, folded_yerr = zip(*sorted(zip(foldtimes, y, yerr)))
    return phase, folded_y, folded_yerr


def truncated_cauchy_rvs(loc=0, scale=1, a=-1, b=1, size=None):
    """Draw random samples from a truncated Cauchy distribution.

    Args:
        loc: Location parameter.
        scale: Scale parameter.
        a: Lower truncation bound.
        b: Upper truncation bound.
        size: Output shape passed to ``numpy.random.uniform``.

    Returns:
        Random variates from the truncated distribution.
    """
    ua = np.arctan((a - loc) / scale) / np.pi + 0.5
    ub = np.arctan((b - loc) / scale) / np.pi + 0.5
    U = np.random.uniform(ua, ub, size=size)
    rvs = loc + scale * np.tan(np.pi * (U - 0.5))
    return rvs


f = lambda x, lims: (
    np.array(
        [
            invgamma(a=x[0], scale=x[1]).cdf(lims[0]) - 0.01,
            invgamma(a=x[0], scale=x[1]).sf(lims[1]) - 0.01,
        ]
    )
    ** 2
).sum()


def inverse_gamma_from_bounds(lower, upper, x0=(1, 5), showit=False):
    """Fit an inverse-gamma distribution from percentile bounds.

    Args:
        lower: Lower bound containing 1% of the probability below it.
        upper: Upper bound containing 1% of the probability above it.
        x0: Initial guesses for the inverse-gamma shape and scale.
        showit: Whether to plot the fitted distribution.

    Returns:
        A frozen SciPy inverse-gamma distribution.
    """
    limits = [lower, upper]
    result = minimize(
        f,
        x0=x0,
        args=limits,
        method="L-BFGS-B",
        bounds=[(0, None), (0, None)],
        tol=1e-10,
    )
    a, b = result.x
    if showit:
        _, ax = plt.subplots(1, 1, constrained_layout=True)
        d = invgamma(a=a, scale=b)
        x = np.linspace(0.2 * limits[0], 2 * limits[1], 1000)
        ax.plot(x, d.pdf(x))
        ax.vlines(limits, 0, d.pdf(x).max())
        plt.show()
    return invgamma(a=a, scale=b)


def logsumexp(log_summands):
    """Compute the logarithm of a sum of exponentials.

    Args:
        log_summands: Values already expressed in log space.

    Returns:
        The log-space sum.
    """
    a = np.inf
    x = log_summands.copy()
    while a == np.inf or a == -np.inf or np.isnan(a):
        a = x[0] + np.log(1 + np.sum(np.exp(x[1:] - x[0])))
        shuffle(x)
    return a


def multivariate_normal_logpdf(r, c, method="cholesky"):
    """Compute a multivariate-normal log density.

    Args:
        r: One-dimensional residual vector.
        c: Covariance matrix.
        method: Linear algebra method. Use ``"cholesky"`` for Cholesky
            decomposition or ``"solve"`` for ``numpy.linalg.solve``.

    Returns:
        The multivariate-normal log density at ``r``.
    """
    # Compute normalization factor used for all methods.
    kk = len(r) * np.log(2 * np.pi)
    if method == "cholesky":
        # Use Cholesky decomposition of covariance.
        cho, lower = cho_factor(c)
        alpha = cho_solve((cho, lower), r)
        return -0.5 * (kk + np.dot(r, alpha) + 2 * np.sum(np.log(np.diag(cho))))
    if method == "solve":
        # Use slogdet and solve
        (_, d) = np.linalg.slogdet(c)
        alpha = np.linalg.solve(c, r)
        return -0.5 * (kk + np.dot(r, alpha) + d)


def rms(array):
    """Compute the root mean square around the array mean.

    Args:
        array: Measurements.

    Returns:
        Root mean square.
    """
    mu = np.average(array)
    rms = np.sqrt(np.sum((array - mu) ** 2) / array.size)
    return rms


def weighted_rms(array, weights):
    """Compute the weighted root mean square around the weighted mean.

    Args:
        array: Measurements.
        weights: Measurement weights, commonly ``1 / errors**2``.

    Returns:
        Weighted root mean square.
    """
    mu = np.average(array, weights=weights)
    rms = np.sqrt(np.sum(weights * (array - mu) ** 2) / np.sum(weights))
    return rms


def anderson_darling_test(r):
    """Run the Anderson-Darling normality test.

    Args:
        r: Residuals to test.

    Returns:
        The SciPy test result and a compact label with the largest exceeded
        significance level.
    """
    from scipy.stats import anderson

    result = anderson(r)
    s = result.significance_level[result.statistic > result.critical_values]
    if s.size == 0:
        return result, f"A-D: {result.significance_level[-1]:.0f}%"
    else:
        return result, f"A-D: {s.max():.0f}%"


phase_folding = phase_fold
truncCauchy_rvs = truncated_cauchy_rvs
invGamma = inverse_gamma_from_bounds
log_sum = logsumexp
multivariate_normal = multivariate_normal_logpdf
wrms = weighted_rms
