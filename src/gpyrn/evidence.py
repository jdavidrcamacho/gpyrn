"""Bayesian evidence estimators."""

import random
from math import log, sqrt

import numpy as np
import scipy.stats


# Taken from https://github.com/exord/bayev
def compute_perrakis_estimate(
    marginal_sample,
    lnlikefunc,
    lnpriorfunc,
    nsamples=1000,
    lnlikeargs=(),
    lnpriorargs=(),
    densityestimation="histogram",
    errorestimation=False,
    **kwargs
):
    """Compute the Perrakis estimate of the Bayesian evidence.

    Args:
        marginal_sample: Marginal posterior sample with shape ``(n, k)``.
        lnlikefunc: Callable that evaluates log likelihoods.
        lnpriorfunc: Callable that evaluates log prior densities.
        nsamples: Number of reshuffled marginal samples to use.
        lnlikeargs: Extra positional arguments for ``lnlikefunc``.
        lnpriorargs: Extra positional arguments for ``lnpriorfunc``.
        densityestimation: Density estimator, one of ``"normal"``, ``"kde"``,
            or ``"histogram"``.
        errorestimation: Whether to estimate the evidence uncertainty.
        **kwargs: Extra options forwarded to :func:`estimate_density`.

    Returns:
        The Perrakis evidence estimate, or ``(estimate, error)`` when
        ``errorestimation`` is true.

    References:
        Perrakis et al. (2014; arXiv:1311.0674).
    """
    print("Estimating evidence...")
    if errorestimation:
        initial_sample = marginal_sample
    marginal_sample = make_marginal_samples(marginal_sample, nsamples)
    if not isinstance(marginal_sample, np.ndarray):
        marginal_sample = np.array(marginal_sample)
    number_parameters = marginal_sample.shape[1]
    marginal_posterior_density = np.zeros(marginal_sample.shape)
    for parameter_index in range(number_parameters):
        x = marginal_sample[:, parameter_index]
        marginal_posterior_density[:, parameter_index] = estimate_density(
            x, method=densityestimation, **kwargs
        )
    prod_marginal_densities = marginal_posterior_density.prod(axis=1)
    log_prior = lnpriorfunc(marginal_sample, *lnpriorargs)
    log_likelihood = lnlikefunc(marginal_sample, *lnlikeargs)
    cond = log_likelihood != 0
    log_summands = (
        log_likelihood[cond] + log_prior[cond] - np.log(prod_marginal_densities[cond])
    )
    perr = logsumexp(log_summands) - log(len(log_summands))
    # error estimation
    K = 10
    if errorestimation:
        batchSize = initial_sample.shape[0] // K
        meanErr = [
            _estimate_perrakis_error(
                initial_sample[0:batchSize, :],
                lnlikefunc,
                lnpriorfunc,
                nsamples=nsamples,
                densityestimation=densityestimation,
            )
        ]
        for i in range(K):
            meanErr.append(
                _estimate_perrakis_error(
                    initial_sample[i * batchSize : (i + 1) * batchSize, :],
                    lnlikefunc,
                    lnpriorfunc,
                    nsamples=nsamples,
                    densityestimation=densityestimation,
                )
            )
        stdErr = np.std(meanErr)
        meanErr = np.mean(meanErr)
        print(perr, stdErr)
        return perr, stdErr
    return perr


def _estimate_perrakis_error(
    marginal_samples,
    lnlikefunc,
    lnpriorfunc,
    nsamples=1000,
    densityestimation="histogram",
    errorestimation=False,
):
    """Estimate the uncertainty of the Perrakis evidence estimator."""
    return compute_perrakis_estimate(
        marginal_samples,
        lnlikefunc,
        lnpriorfunc,
        nsamples=nsamples,
        densityestimation=densityestimation,
        errorestimation=errorestimation,
    )


def _estimate_error(
    marginal_sample,
    lnlikefunc,
    lnpriorfunc,
    nsamples=300,
    densityestimation="histogram",
    **kwargs
):
    """Estimate evidence uncertainty from reshuffled marginal samples."""
    print("Estimating evidence error...")
    marginal_sample = make_marginal_samples(marginal_sample, nsamples)
    if not isinstance(marginal_sample, np.ndarray):
        marginal_sample = np.array(marginal_sample)
    number_parameters = marginal_sample.shape[1]
    # Estimate marginal posterior density for each parameter.
    marginal_posterior_density = np.zeros(marginal_sample.shape)
    for parameter_index in range(number_parameters):
        # Extract samples for this parameter.
        x = marginal_sample[:, parameter_index]
        # Estimate density with method "densityestimation".
        marginal_posterior_density[:, parameter_index] = estimate_density(
            x, method=densityestimation, **kwargs
        )
    # Compute produt of marginal posterior densities for all parameters
    prod_marginal_densities = marginal_posterior_density.prod(axis=1)
    # Compute lnprior and likelihood in marginal sample.
    log_prior = lnpriorfunc(marginal_sample)
    log_likelihood = lnlikefunc(marginal_sample)
    # Mask values with zero likelihood (a problem in lnlike)
    cond = log_likelihood != 0
    log_summands = (
        log_likelihood[cond] + log_prior[cond] - np.log(prod_marginal_densities[cond])
    )
    perr = logsumexp(log_summands) - log(len(log_summands))
    return perr


def estimate_density(x, method="histogram", **kwargs):
    """Estimate density at sample points.

    Args:
        x: One-dimensional sample.
        method: Estimator to use: ``"histogram"``, ``"kde"``, or ``"normal"``.
        **kwargs: Additional estimator options. ``nbins`` controls the number
            of bins for histogram estimation.

    Returns:
        Density estimates at the sample points.
    """
    nbins = kwargs.pop("nbins", 100)
    if method == "normal":
        # Approximate each parameter distribution by a normal.
        return scipy.stats.norm.pdf(x, loc=x.mean(), scale=sqrt(x.var()))
    if method == "kde":
        # Approximate each parameter distribution using a gaussian kernel estimation
        return scipy.stats.gaussian_kde(x)(x)
    if method == "histogram":
        # Approximate each parameter distribution based on the histogram
        density, bin_edges = np.histogram(x, nbins, density=True)
        # Find to which bin each element corresponds
        density_indexes = np.searchsorted(bin_edges, x, side="left")
        # Correct to avoid index zero from being assiged to last element
        density_indexes = np.where(
            density_indexes > 0, density_indexes, density_indexes + 1
        )
        return density[density_indexes - 1]


def make_marginal_samples(joint_samples, nsamples=None):
    """Reshuffle joint posterior samples into marginal samples.

    Args:
        joint_samples: Joint posterior samples with shape ``(n, k)``.
        nsamples: Number of samples to return. If ``None`` or larger than the
            available sample, all samples are used.

    Returns:
        Marginal samples with each parameter independently shuffled.
    """
    if nsamples > len(joint_samples) or nsamples is None:
        nsamples = len(joint_samples)
    marginal_samples = joint_samples[-nsamples:, :].copy()
    number_parameters = marginal_samples.shape[-1]
    # Reshuffle joint posterior samples to obtain _marginal_ posterior samples
    for parameter_index in range(number_parameters):
        random.shuffle(marginal_samples[:, parameter_index])
    return marginal_samples


def logsumexp(log_summands):
    """Compute the logarithm of a sum of exponentials."""
    a = np.inf
    x = log_summands.copy()
    while a == np.inf or a == -np.inf or np.isnan(a):
        a = x[0] + np.log(1 + np.sum(np.exp(x[1:] - x[0])))
        random.shuffle(x)
    return a


def compute_harmonic_mean(
    lnlike_post, posterior_sample=None, lnlikefunc=None, lnlikeargs=(), **kwargs
):
    """Compute the harmonic-mean estimate of the marginal likelihood.

    Args:
        lnlike_post: Log likelihoods evaluated on posterior samples. Pass an
            empty sequence to compute them from ``posterior_sample``.
        posterior_sample: Posterior sample used when ``lnlike_post`` is empty.
        lnlikefunc: Function used to compute log likelihoods.
        lnlikeargs: Extra positional arguments for ``lnlikefunc``.
        **kwargs: Optional ``size`` value controlling the subsample size.

    Returns:
        Harmonic-mean evidence estimate.

    References:
        Kass & Raftery (1995), JASA 90(430), 773-795.
    """
    if len(lnlike_post) == 0 and posterior_sample is not None:
        samplesize = kwargs.pop("size", len(posterior_sample))
        if samplesize < len(posterior_sample):
            posterior_subsample = np.random.choice(
                posterior_sample, size=samplesize, replace=False
            )
        else:
            posterior_subsample = posterior_sample.copy()
        # Compute log likelihood in posterior sample.
        log_likelihood = lnlikefunc(posterior_subsample, *lnlikeargs)
    elif len(lnlike_post) > 0:
        samplesize = kwargs.pop("size", len(lnlike_post))
        log_likelihood = np.random.choice(lnlike_post, size=samplesize, replace=False)
    hme = -logsumexp(-log_likelihood) + log(len(log_likelihood))
    return hme


def run_harmonic_mean_mc(log_likelihood, nmc, samplesize):
    """Run repeated harmonic-mean estimates on random subsamples."""
    hme = np.zeros(nmc)
    for i in range(nmc):
        hme[i] = compute_harmonic_mean(log_likelihood, size=samplesize)
    return hme


def compute_chib_jeliazkov_estimate(
    posterior_sample,
    lnlikefunc,
    lnpriorfunc,
    param_post,
    nsamples,
    qprob=None,
    lnlikeargs=(),
    lnpriorargs=(),
    lnlike_post=None,
    lnprior_post=None,
):
    """Compute the Chib-Jeliazkov estimate of Bayesian evidence.

    Args:
        posterior_sample: Posterior sample with shape ``(n, k)``.
        lnlikefunc: Function that evaluates log likelihoods.
        lnpriorfunc: Function that evaluates log prior densities.
        param_post: One-dimensional posterior parameter sample used to select
            the fixed point.
        nsamples: Number of proposal samples to draw.
        qprob: Proposal distribution with callable ``pdf`` and ``rvs`` methods.
            When omitted, a multivariate Gaussian proposal is fitted.
        lnlikeargs: Extra positional arguments for ``lnlikefunc``.
        lnpriorargs: Extra positional arguments for ``lnpriorfunc``.
        lnlike_post: Optional precomputed posterior log likelihoods.
        lnprior_post: Optional precomputed posterior log priors.

    Returns:
        Natural logarithm of the estimated Bayesian evidence.

    Raises:
        AttributeError: If ``qprob`` lacks ``pdf`` or ``rvs``.
        TypeError: If ``qprob.pdf`` or ``qprob.rvs`` is not callable.

    References:
        Chib & Jeliazkov (2001), Journal of the American Statistical
        Association, 96(453).
    """
    # Find fixed point on which to estimate posterior ordinate.
    if lnlike_post is not None:
        # Pass values of log(likelihood) in posterior sample.
        arg_fp = [
            lnlike_post,
        ]
    else:
        # Pass function that computes log(likelihood).
        arg_fp = [
            lnlikefunc,
        ]
    if lnlike_post is not None:
        # Pass values of log(prior) in posterior sample.
        arg_fp.append(lnprior_post)
    else:
        # Pass function that computes log(prior).
        arg_fp.append(lnpriorfunc)
    fp, lnpost0 = get_fixed_point(
        posterior_sample,
        param_post,
        lnlikefunc,
        lnpriorfunc,
        lnlikeargs=lnlikeargs,
        lnpriorargs=lnpriorargs,
    )
    # If proposal distribution is not given, define as multivariate Gaussian.
    if qprob is None:
        # Get covariance from posterior sample
        k = np.cov(posterior_sample.T)
        qprob = scipy.stats.multivariate_normal(mean=fp, cov=k, allow_singular=True)
    else:
        # Check that qprob has the necessary attributes
        for method in ("pdf", "rvs"):
            try:
                att = getattr(qprob, method)
            except AttributeError:
                raise AttributeError(
                    "qprob does not have method " "'{}'".format(method)
                )
            if not callable(att):
                raise TypeError("{} method of qprob is not " "callable".format(method))
    # Compute proposal density in posterior sample
    q_post = qprob.pdf(posterior_sample)
    # If likelihood over posterior sample is not given, compute it
    if lnlike_post is None:
        lnlike_post = lnlikefunc(posterior_sample, *lnlikeargs)
    # Idem for prior
    if lnprior_post is None:
        lnprior_post = lnpriorfunc(posterior_sample, *lnpriorargs)
    # Compute Metropolis ratio with respect to fixed point over posterior sample
    lnalpha_post = metropolis_ratio(lnprior_post + lnlike_post, lnpost0)
    # Sample from the proposal distribution with respect to fixed point
    proposal_sample = qprob.rvs(nsamples)
    # Compute likelihood and prior on proposal_sample
    lnprior_prop = lnpriorfunc(proposal_sample, *lnpriorargs)
    if np.all(lnprior_prop == -np.inf):
        raise ValueError(
            "All samples from proposal density have zero prior"
            "probability. Increase nsample."
        )
    # Now compute likelihood only on the samples where prior != 0.
    lnlike_prop = np.full_like(lnprior_prop, -np.inf)
    ind = lnprior_prop != -np.inf
    lnlike_prop[ind] = lnlikefunc(proposal_sample[ind, :], *lnlikeargs)
    # Get Metropolis ratio with respect to fixed point over proposal sample
    lnalpha_prop = metropolis_ratio(lnpost0, lnprior_prop + lnlike_prop)
    # Compute estimate of posterior ordinate (see Eq. 9 from reference)
    num = logsumexp(lnalpha_post + q_post) - log(len(posterior_sample))
    den = logsumexp(lnalpha_prop) - log(len(proposal_sample))
    lnpostord = num - den
    # Return log(Evidence) estimation
    return lnpost0 - lnpostord


def metropolis_ratio(lnpost0, lnpost1):
    """Compute the log Metropolis ratio for two states.

    Args:
        lnpost0: Log posterior for the initial state.
        lnpost1: Log posterior for the proposal state.

    Returns:
        The log Metropolis ratio.

    Raises:
        ValueError: If iterable inputs have different lengths.
    """
    if (
        hasattr(lnpost0, "__iter__")
        and hasattr(lnpost1, "__iter__")
        and len(lnpost0) != len(lnpost1)
    ):
        raise ValueError("lnpost0 and lnpost1 have different lenghts.")
    return np.minimum(lnpost1 - lnpost0, 0.0)


def get_fixed_point(
    posterior_samples, param_post, lnlike, lnprior, lnlikeargs=(), lnpriorargs=()
):
    """Find the posterior point used as the fixed point for evidence estimates.

    Args:
        posterior_samples: Posterior samples with shape ``(n, k)``.
        param_post: One-dimensional parameter sample used to choose the median
            fixed point.
        lnlike: Callable or precomputed log-likelihood array.
        lnprior: Callable or precomputed log-prior array.
        lnlikeargs: Extra positional arguments for callable ``lnlike``.
        lnpriorargs: Extra positional arguments for callable ``lnprior``.

    Returns:
        The fixed point and its log posterior value.

    Raises:
        IndexError: If precomputed arrays do not match ``posterior_samples``.
    """
    if param_post is not None:
        # Use median of param_post as fixed point.
        param0 = np.median(param_post)
        # Find argument closest to median.
        ind0 = np.argmin(np.abs(param_post - param0))
        fixed_point = posterior_samples[ind0, :]
        # Compute log(likelihood) at fixed_point
        if hasattr(lnlike, "__iter__"):
            if len(lnlike) != len(posterior_samples):
                raise IndexError(
                    "Number of elements in lnlike array and in "
                    "posterior sample must match."
                )
            lnlike0 = lnlike[ind0]
        else:
            # Evaluate lnlike function at fixed point.
            lnlike0 = lnlike(fixed_point, *lnlikeargs)
        # Compute log(prior) at fixed_point
        if hasattr(lnprior, "__iter__"):
            if len(lnprior) != len(posterior_samples):
                raise IndexError(
                    "Number of elements in lnprior array and in "
                    "posterior sample must match."
                )
            lnprior0 = lnprior[ind0]
        else:
            # Evaluate lnlike function at fixed point.
            lnprior0 = lnprior(fixed_point, *lnpriorargs)
        return fixed_point, lnlike0 + lnprior0
    raise NotImplementedError


_perrakis_error = _estimate_perrakis_error
_errorCalc = _estimate_error
log_sum = logsumexp
compute_harmonicmean = compute_harmonic_mean
run_hme_mc = run_harmonic_mean_mc
compute_cj_estimate = compute_chib_jeliazkov_estimate
