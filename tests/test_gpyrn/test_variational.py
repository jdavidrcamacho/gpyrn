import numpy as np
import pytest

from gpyrn import kernels
from gpyrn import means as mean_functions
from gpyrn.variational import MeanFieldInference


@pytest.fixture
def single_output_data():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 1.5, 0.5, 1.25])
    yerr = np.full_like(time, 0.1)
    return time, y, yerr


@pytest.fixture
def configured_single_output_model(single_output_data):
    time, y, yerr = single_output_data
    model = MeanFieldInference(1, time, y, yerr)
    model.set_components(
        kernels.SquaredExponential(1.0, 2.0),
        kernels.SquaredExponential(0.5, 1.5),
        mean_functions.Constant(0.1),
        0.05,
    )
    return model


def test_create_mean_field_inference_shapes_single_and_multiple_outputs():
    time = np.array([0.0, 1.0, 2.0])
    y1 = np.array([1.0, 2.0, 3.0])
    e1 = np.full(3, 0.1)
    y2 = np.array([3.0, 2.0, 1.0])
    e2 = np.full(3, 0.2)

    single = MeanFieldInference(1, time, y1, e1)
    multiple = MeanFieldInference(2, time, y1, e1, y2, e2)

    assert single.N == 3
    assert single.q == 1
    assert single.p == 1
    assert single.qp == 1
    assert np.array_equal(single.y, y1.reshape(1, 3))
    assert np.array_equal(single.yerr, e1.reshape(1, 3))

    assert multiple.q == 2
    assert multiple.p == 2
    assert multiple.qp == 4
    assert multiple.d == 18
    assert np.array_equal(multiple.y, np.vstack([y1, y2]))
    assert np.array_equal(multiple.yerr, np.vstack([e1, e2]))


def test_create_mean_field_inference_rejects_missing_odd_or_mismatched_outputs():
    time = np.array([0.0, 1.0, 2.0])
    y = np.ones(3)
    yerr = np.ones(3)

    with pytest.raises(TypeError):
        MeanFieldInference(1)

    with pytest.raises(AssertionError, match="should be even"):
        MeanFieldInference(1, time, y)

    with pytest.raises(AssertionError, match="same dimensions"):
        MeanFieldInference(1, time, y, yerr, np.ones(4), np.ones(4))


def test_set_components_accepts_singletons_and_normalizes_jitters(single_output_data):
    time, y, yerr = single_output_data
    model = MeanFieldInference(1, time, y, yerr)
    node = kernels.SquaredExponential(1.0, 2.0)
    weight = kernels.SquaredExponential(0.5, 1.5)
    mean = mean_functions.Constant(0.1)

    model.set_components(node, weight, mean, 0.05)

    assert model.nodes == [node]
    assert model.weights == [weight]
    assert model.means == [mean]
    assert np.allclose(model.jitters, [0.05])
    assert model.n_parameters == 6


@pytest.mark.parametrize(
    ("nodes", "weights", "means", "jitters", "message"),
    [
        (
            [kernels.SquaredExponential(1.0, 1.0)] * 2,
            [kernels.SquaredExponential(1.0, 1.0)],
            [mean_functions.Constant(0.0)],
            [0.1],
            "nodes",
        ),
        (
            [kernels.SquaredExponential(1.0, 1.0)],
            [kernels.SquaredExponential(1.0, 1.0)] * 2,
            [mean_functions.Constant(0.0)],
            [0.1],
            "weights",
        ),
        (
            [kernels.SquaredExponential(1.0, 1.0)],
            [kernels.SquaredExponential(1.0, 1.0)],
            [mean_functions.Constant(0.0)] * 2,
            [0.1],
            "means",
        ),
        (
            [kernels.SquaredExponential(1.0, 1.0)],
            [kernels.SquaredExponential(1.0, 1.0)],
            [mean_functions.Constant(0.0)],
            [0.1, 0.2],
            "jitters",
        ),
    ],
)
def test_set_components_rejects_wrong_component_counts(
    single_output_data, nodes, weights, means, jitters, message
):
    time, y, yerr = single_output_data
    model = MeanFieldInference(1, time, y, yerr)

    with pytest.raises(ValueError, match=message):
        model.set_components(nodes, weights, means, jitters)


def test_parameters_dict_and_get_parameters_order(configured_single_output_model):
    model = configured_single_output_model

    assert list(model.parameters_dict) == [
        "node1.theta",
        "node1.ell",
        "weight1.theta",
        "weight1.ell",
        "mean1.c",
        "jitter1",
    ]
    assert np.allclose(
        model.get_parameters(include_frozen=True), [1.0, 2.0, 0.5, 1.5, 0.1, 0.05]
    )


def test_freezing_by_name_and_index_controls_free_parameter_vector(
    configured_single_output_model,
):
    model = configured_single_output_model

    model.freeze_parameter(name="node1.theta")
    model.freeze_parameter(index=4)

    assert model.frozen_mask.tolist() == [True, False, False, False, True, False]
    assert np.allclose(model.get_parameters(), [2.0, 0.5, 1.5, 0.05])

    model.thaw_parameter(name="node1.theta")
    model.thaw_parameter(index=4)
    assert not model.frozen_mask.any()


def test_set_parameters_accepts_free_parameters_and_preserves_frozen_values(
    configured_single_output_model,
):
    model = configured_single_output_model
    model.freeze_parameter(name="node1.theta")

    model.set_parameters(np.array([20.0, 30.0, 40.0, 50.0, 60.0]))

    assert np.allclose(model.nodes[0].pars, [1.0, 20.0])
    assert np.allclose(model.weights[0].pars, [30.0, 40.0])
    assert np.allclose(model.means[0].pars, [50.0])
    assert np.allclose(model.jitters, [60.0])


def test_set_parameters_rejects_wrong_number_of_parameters(
    configured_single_output_model,
):
    with pytest.raises(ValueError, match="Wrong number of parameters"):
        configured_single_output_model.set_parameters(np.array([1.0, 2.0]))


def test_mean_and_kernel_matrix_helpers_are_deterministic(
    configured_single_output_model,
):
    model = configured_single_output_model
    time = model.time

    mean = model._mean(model.means)
    kernel_matrix = model._kernel_matrix(model.nodes[0], time)
    cross_kernel = model._predict_kernel_matrix(model.nodes[0], np.array([0.5, 1.5]))

    assert np.allclose(mean, np.full(time.size, 0.1))
    assert kernel_matrix.shape == (time.size, time.size)
    assert np.all(np.diag(kernel_matrix) > 1.0)
    assert cross_kernel.shape == (2, time.size)


def test_calculate_elbo_and_prediction_return_finite_values(
    configured_single_output_model,
):
    model = configured_single_output_model

    elbo, mu, var, iterations = model.calculate_elbo(max_iter=1)
    prediction, prediction_variance = model.predict_from_variational_parameters(
        tstar=np.array([0.5, 1.5]),
        mu=mu,
        var=var,
    )

    assert np.isfinite(elbo)
    assert mu.shape == (model.q * (model.p + 1), 1, model.N)
    assert var.shape == (model.q * (model.p + 1), 1, model.N)
    assert iterations == 1
    assert prediction.shape == (2, 1)
    assert prediction_variance.shape == (2, 1)
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction_variance > 0.0)
