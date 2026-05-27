"""Gaussian process regression networks for exoplanet detection."""

__version__ = "2.0.0"

from .kernels import QuasiPeriodic, SquaredExponential
from .means import Constant, Linear
from .variational import MeanFieldInference

inference = MeanFieldInference
