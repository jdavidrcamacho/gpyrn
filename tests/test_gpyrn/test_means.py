import numpy as np
import pytest

from gpyrn import means


def test_constant_mean_function_returns_scalar_for_scalar_input_and_array_for_array():
    mean = means.Constant(3.5)

    assert np.array_equal(mean(1.0), np.array([3.5]))
    assert np.array_equal(mean(np.array([0.0, 1.0, 2.0])), np.full(3, 3.5))

    with pytest.raises(TypeError):
        means.Constant()


def test_linear_mean_function_uses_sample_mean_as_reference():
    mean = means.Linear(slope=2.0, intercept=-1.0)
    t = np.array([0.0, 1.0, 3.0])

    assert np.allclose(mean(t), 2.0 * (t - t.mean()) - 1.0)


def test_polynomial_and_sine_mean_functions_match_numpy_formulas():
    t = np.array([0.0, 1.0, 2.0])

    assert np.allclose(
        means.Parabola(1.0, -2.0, 3.0)(t), np.polyval([1.0, -2.0, 3.0], t)
    )
    assert np.allclose(
        means.Cubic(1.0, 0.0, -1.0, 2.0)(t),
        np.polyval([1.0, 0.0, -1.0, 2.0], t),
    )
    assert np.allclose(
        means.Sine(2.0, 4.0, 0.0)(t), 2.0 * np.sin(2.0 * np.pi * t / 4.0)
    )


def test_composed_mean_functions_evaluate_and_split_parameters():
    first = means.Constant(1.0)
    second = means.Linear(2.0, 3.0)
    summed = first + second
    multiplied = first * second
    t = np.array([0.0, 1.0])

    assert np.allclose(summed(t), first(t) + second(t))
    assert np.allclose(multiplied(t), first(t) * second(t))

    remaining = summed.set_parameters([4.0, 5.0, 6.0, 7.0])
    assert np.allclose(first.pars, [4.0])
    assert np.allclose(second.pars, [5.0, 6.0])
    assert np.allclose(remaining, [7.0])


def test_multi_constant_applies_offsets_by_observation_id_and_prediction_bins():
    time = np.array([0.0, 1.0, 10.0, 11.0])
    obsid = np.array([1, 1, 2, 2])
    mean = means.MultiConstant(offsets=np.array([5.0, 10.0]), obsid=obsid, time=time)

    assert np.allclose(mean(time), [15.0, 15.0, 10.0, 10.0])
    assert np.allclose(mean(np.array([0.5, 10.5])), [15.0, 10.0])

    with pytest.raises(AssertionError, match="wrong number of parameters"):
        means.MultiConstant(offsets=np.array([1.0]), obsid=obsid, time=time)
