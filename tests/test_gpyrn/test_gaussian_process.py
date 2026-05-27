import numpy as np

from gpyrn import kernels
from gpyrn.gaussian_process import GaussianProcess


def test_gaussian_process_initializes_default_and_explicit_errors():
    time = np.array([0.0, 1.0])
    y = np.array([1.0, 2.0])

    default = GaussianProcess(time, y)
    explicit = GaussianProcess(time, y, yerr=np.array([0.1, 0.2]))

    assert np.allclose(default.yerr, 1e-12 * np.eye(2))
    assert np.allclose(default.yerr2, (1e-12 * np.eye(2)) ** 2)
    assert np.allclose(explicit.yerr, [0.1, 0.2])
    assert np.allclose(explicit.yerr2, [0.01, 0.04])


def test_kernel_matrix_helpers_cover_stationary_and_nonstationary_kernels():
    time = np.array([0.2, 0.4, 0.6])
    gp = GaussianProcess(time, np.zeros_like(time))

    stationary = gp._kernel_matrix(kernels.SquaredExponential(1.0, 2.0), time)
    predictive = gp._predict_kernel_matrix(
        kernels.SquaredExponential(1.0, 2.0), np.array([0.3, 0.5])
    )
    polynomial = gp._kernel_matrix(kernels.Polynomial(1.0, 0.5, 2.0, 2), time)
    harmonic = gp._predict_kernel_matrix(
        kernels.HarmonicPeriodic(2, 1.0, 5.0, 1.5), np.array([0.3, 0.5])
    )

    assert stationary.shape == (3, 3)
    assert predictive.shape == (2, 3)
    assert polynomial.shape == (3, 3)
    assert harmonic.shape == (2, 3)


def test_new_kernel_rebuilds_single_sum_and_product_kernels():
    gp = GaussianProcess(np.array([0.0, 1.0]), np.zeros(2))

    single = gp.new_kernel(kernels.SquaredExponential(1.0, 2.0), [3.0, 4.0])
    summed = gp.new_kernel(
        kernels.SquaredExponential(1.0, 2.0) + kernels.Constant(3.0),
        [4.0, 5.0, 6.0],
    )
    product = gp.new_kernel(
        kernels.SquaredExponential(1.0, 2.0) * kernels.SquaredExponential(3.0, 4.0),
        [5.0, 6.0, 7.0, 8.0],
    )

    assert isinstance(single, kernels.SquaredExponential)
    assert np.allclose(single.pars, [3.0, 4.0])
    assert np.allclose(summed.k1.pars, [4.0, 5.0])
    assert np.allclose(summed.k2.pars, [6.0])
    assert np.allclose(product.k1.pars, [5.0, 6.0])
    assert np.allclose(product.k2.pars, [7.0, 8.0])


def test_prediction_returns_training_values_for_low_noise_identity_case():
    time = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, -0.5, 0.25])
    gp = GaussianProcess(time, y)

    mean, variance = gp.prediction(
        kernels.SquaredExponential(1.0, 0.2),
        time,
        m=y,
        v=np.full_like(time, 1e-8),
    )

    assert mean.shape == time.shape
    assert variance.shape == time.shape
    assert np.allclose(mean, y, atol=1e-4)
    assert np.all(variance >= 0.0)
