import numpy as np
import pytest

from gpyrn import kernels


def test_squared_exponential_kernel_matches_closed_form():
    kernel = kernels.SquaredExponential(theta=2.0, ell=4.0)
    r = np.array([0.0, 2.0, 4.0])

    expected = 4.0 * np.exp(-0.5 * r**2 / 16.0)

    assert np.allclose(kernel(r), expected)
    assert repr(kernel) == "SquaredExponential(theta=2.0, ell=4.0)"


def test_periodic_and_quasi_periodic_kernels_match_closed_forms():
    r = np.array([0.0, 0.5, 1.0])
    periodic = kernels.Periodic(theta=3.0, P=2.0, ell=0.5)
    quasi_periodic = kernels.QuasiPeriodic(theta=3.0, elle=10.0, P=2.0, ellp=0.5)

    expected_periodic = 9.0 * np.exp(-2.0 * np.sin(np.pi * np.abs(r) / 2.0) ** 2 / 0.25)
    expected_quasi_periodic = expected_periodic * np.exp(-(r**2) / 200.0)

    assert np.allclose(periodic(r), expected_periodic)
    assert np.allclose(quasi_periodic(r), expected_quasi_periodic)


def test_quasi_periodic_matches_squared_exponential_times_periodic():
    r = np.linspace(-5.0, 5.0, 21)

    product_kernel = kernels.SquaredExponential(2.0, 10.0) * kernels.Periodic(
        1.0, 3.0, 0.5
    )
    quasi_periodic = kernels.QuasiPeriodic(2.0, 10.0, 3.0, 0.5)

    assert np.allclose(product_kernel(r), quasi_periodic(r))


def test_white_noise_is_diagonal_for_square_matrices_and_dense_for_vectors():
    kernel = kernels.WhiteNoise(2.0)

    assert np.array_equal(kernel(np.zeros((3, 3))), 4.0 * np.eye(3))
    assert np.array_equal(kernel(np.zeros(3)), np.full(3, 4.0))


def test_kernel_sum_product_and_parameter_splitting():
    kernel = kernels.SquaredExponential(1.0, 2.0) + kernels.Constant(3.0)
    r = np.array([0.0, 1.0])

    assert np.allclose(
        kernel(r),
        kernels.SquaredExponential(1.0, 2.0)(r) + kernels.Constant(3.0)(r),
    )

    remaining = kernel.k1.set_parameters([4.0, 5.0, 6.0])
    assert np.allclose(kernel.k1.pars, [4.0, 5.0])
    assert np.allclose(remaining, [6.0])


def test_kernel_product_repr_and_exact_parameter_update():
    kernel = kernels.SquaredExponential(1.0, 2.0) * kernels.Periodic(1.0, 3.0, 0.5)
    r = np.array([0.0, 0.25])

    assert np.allclose(
        kernel(r),
        kernels.SquaredExponential(1.0, 2.0)(r) * kernels.Periodic(1.0, 3.0, 0.5)(r),
    )
    assert "*" in repr(kernel)

    assert kernel.k2.set_parameters([2.0, 4.0, 0.75]) is None
    assert np.allclose(kernel.k2.pars, [2.0, 4.0, 0.75])

    with pytest.raises(AssertionError, match="too few parameters"):
        kernel.k2.set_parameters([1.0])


def test_derivative_requires_twice_differentiable_kernel():
    with pytest.raises(ValueError, match="not twice differentiable"):
        kernels.Derivative(kernels.WhiteNoise(1.0))


def test_derivative_kernel_uses_second_derivative():
    r = np.array([0.0, 0.5])
    base = kernels.SquaredExponential(2.0, 3.0)
    derivative = kernels.Derivative(base)

    assert np.allclose(derivative(r), base._dkdxidj(r))
    assert repr(derivative).startswith("d SquaredExponential")


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (
            kernels.RationalQuadratic(2.0, 1.5, 3.0),
            lambda r: 4.0 * (1.0 + 0.5 * r**2 / (1.5 * 9.0)) ** -1.5,
        ),
        (kernels.RQP(2.0, 1.5, 3.0, 5.0, 0.75), None),
        (
            kernels.Cosine(2.0, 5.0),
            lambda r: 4.0 * np.cos(2.0 * np.pi * np.abs(r) / 5.0),
        ),
        (kernels.Exponential(2.0, 3.0), lambda r: 4.0 * np.exp(-np.abs(r) / 3.0)),
        (
            kernels.Matern32(2.0, 3.0),
            lambda r: 4.0
            * (1.0 + np.sqrt(3.0) * np.abs(r) / 3.0)
            * np.exp(-np.sqrt(3.0) * np.abs(r) / 3.0),
        ),
        (
            kernels.Matern52(2.0, 3.0),
            lambda r: 4.0
            * (
                1.0
                + (3.0 * np.sqrt(5.0) * 3.0 * np.abs(r) + 5.0 * np.abs(r) ** 2) / 27.0
            )
            * np.exp(-np.sqrt(5.0) * np.abs(r) / 3.0),
        ),
        (
            kernels.GammaExp(2.0, 1.5, 3.0),
            lambda r: 4.0 * np.exp(-((np.abs(r) / 3.0) ** 1.5)),
        ),
        (kernels.Piecewise(2.0), None),
        (kernels.Paciorek(2.0, 1.0, 3.0), None),
        (kernels.NewPeriodic(2.0, 1.5, 5.0, 0.75), None),
        (kernels.QuasiNewPeriodic(2.0, 1.5, 3.0, 5.0, 0.75), None),
        (kernels.NewRQP(2.0, 1.5, 2.0, 3.0, 5.0, 0.75), None),
        (kernels.CosPeriodic(2.0, 5.0, 0.75), None),
        (kernels.QuasiCosPeriodic(2.0, 3.0, 5.0, 0.75), None),
    ],
)
def test_stationary_kernel_families_return_finite_values(kernel, expected):
    r = np.array([0.0, 0.5, 1.0])
    values = kernel(r)

    assert values.shape == r.shape
    assert np.all(np.isfinite(values))
    if expected is not None:
        assert np.allclose(values, expected(r))


def test_nonstationary_kernel_families_return_finite_matrices():
    t1 = np.array([[0.2], [0.4]])
    t2 = np.array([[0.3, 0.6]])

    values = kernels.Linear(0.1)(None, t1, t2)
    assert values.shape == (2, 2)
    assert np.all(np.isfinite(values))

    for kernel in [
        kernels.Polynomial(1.0, 0.5, 2.0, 3),
        kernels.HarmonicPeriodic(2, 1.0, 5.0, 1.5),
        kernels.QuasiHarmonicPeriodic(2, 1.0, 10.0, 5.0, 1.5),
    ]:
        values = kernel(t1, t2)
        assert values.shape == (2, 2)
        assert np.all(np.isfinite(values))


def test_base_covariance_function_methods_raise_not_implemented():
    kernel = kernels.CovarianceFunction(1.0)

    with pytest.raises(NotImplementedError):
        kernel(np.array([0.0]))
    with pytest.raises(NotImplementedError):
        kernel._dkdxidj(np.array([0.0]))
