import importlib.metadata

import gpyrn


def test_package_exports_current_public_api():
    assert gpyrn.__version__ == importlib.metadata.version("gpyrn")
    assert gpyrn.inference is gpyrn.MeanFieldInference
    assert gpyrn.Constant.__module__ == "gpyrn.means"
    assert gpyrn.SquaredExponential.__module__ == "gpyrn.kernels"


def test_current_modules_import_without_compatibility_wrappers():
    from gpyrn import (
        evidence,
        gaussian_process,
        kernels,
        means,
        plotting,
        utils,
        variational,
    )

    assert kernels.SquaredExponential
    assert means.Constant
    assert variational.MeanFieldInference
    assert gaussian_process.GaussianProcess
    assert plotting.plot_prediction
    assert utils.semi_amplitude
    assert evidence.compute_perrakis_estimate
