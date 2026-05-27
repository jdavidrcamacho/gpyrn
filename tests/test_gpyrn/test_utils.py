import numpy as np
import pytest
from scipy.stats import multivariate_normal

from gpyrn import utils


def test_semi_amplitude_supports_current_numpy():
    assert np.isclose(utils.semi_amplitude(1.0, 1.0, 1.0, 0.0), 28.435)


def test_keplerian_requires_times_and_returns_expected_circular_signal():
    time = np.array([0.0, 0.25, 0.5])

    returned_time, rv = utils.keplerian(P=1.0, K=2.0, e=0.0, w=0.0, T=0.0, t=time)

    assert returned_time is time
    assert np.allclose(rv, 2.0 * np.cos(2.0 * np.pi * time), atol=1e-12)

    with pytest.raises(ValueError, match="t must be provided"):
        utils.keplerian()


def test_phase_fold_sorts_phase_measurements_and_errors():
    t = np.array([2.5, 0.5, 1.5])
    y = np.array([25.0, 5.0, 15.0])
    yerr = np.array([2.5, 0.5, 1.5])

    phase, folded_y, folded_yerr = utils.phase_fold(t, y, yerr, period=2.0)

    assert np.allclose(phase, [0.25, 0.25, 0.75])
    assert folded_y == (5.0, 25.0, 15.0)
    assert folded_yerr == (0.5, 2.5, 1.5)


def test_logsumexp_and_multivariate_normal_logpdf_match_scipy():
    values = np.array([-1000.0, -1001.0, -1002.0])
    residuals = np.array([0.5, -0.25])
    covariance = np.array([[2.0, 0.25], [0.25, 1.0]])

    assert np.isclose(utils.logsumexp(values), np.logaddexp.reduce(values))
    assert np.isclose(
        utils.multivariate_normal_logpdf(residuals, covariance),
        multivariate_normal(mean=np.zeros(2), cov=covariance).logpdf(residuals),
    )
    assert np.isclose(
        utils.multivariate_normal_logpdf(residuals, covariance, method="solve"),
        multivariate_normal(mean=np.zeros(2), cov=covariance).logpdf(residuals),
    )


def test_rms_and_weighted_rms_match_manual_calculation():
    values = np.array([1.0, 2.0, 4.0])
    weights = np.array([1.0, 2.0, 1.0])

    expected_rms = np.sqrt(np.sum((values - values.mean()) ** 2) / values.size)
    weighted_mean = np.average(values, weights=weights)
    expected_weighted_rms = np.sqrt(
        np.sum(weights * (values - weighted_mean) ** 2) / np.sum(weights)
    )

    assert np.isclose(utils.rms(values), expected_rms)
    assert np.isclose(utils.weighted_rms(values, weights), expected_weighted_rms)


def test_truncated_cauchy_samples_stay_within_bounds():
    samples = utils.truncated_cauchy_rvs(loc=0.0, scale=1.0, a=-2.0, b=3.0, size=1000)

    assert samples.shape == (1000,)
    assert np.all(samples >= -2.0)
    assert np.all(samples <= 3.0)
