import numpy as np
from scipy.linalg import LinAlgError, cholesky, inv
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


class NonparametricInference:
    """Experimental nonparametric variational inference for GPRNs.

    Args:
        num_nodes: Number of latent node functions.
        time: Time coordinates.
        k: Number of isotropic Gaussian mixture components.
        *args: Observed data arrays in the order
            ``data1, data1error, data2, data2error, ...``.
    """

    def __init__(self, num_nodes, time, k, *args):
        # number of node functions; f(x) in Wilson et al. (2012)
        self.num_nodes = num_nodes
        self.q = num_nodes
        # array of the time
        self.time = time
        # number of observations, N in Wilson et al. (2012)
        self.N = self.time.size
        # mixture of k isotropic gaussian distributions
        self.k = k
        # the data, it should be given as data1, data1error, data2, ...
        self.args = args
        # number of outputs y(x); p in Wilson et al. (2012)
        self.p = int(len(self.args) / 2)
        # total number of weights, we will have q*p weights in total
        self.qp = self.q * self.p
        self.d = self.time.size * self.q * (self.p + 1)
        # to organize the data we now join everything
        self.tt = np.tile(time, self.p)  # "extended" time
        ys = []
        ystd = []
        yerrs = []
        for i, j in enumerate(args):
            if i % 2 == 0:
                ys.append(j)
                ystd.append(np.std(j))
            else:
                yerrs.append(j)
        self.ystd = np.array(ystd).reshape(self.p, 1)
        self.y = np.array(ys).reshape(self.p, self.N)  # matrix p*N of outputs
        self.yerr = np.array(yerrs).reshape(self.p, self.N)  # matrix p*N of errors
        self.yerr2 = self.yerr**2
        # check if the input was correct
        assert (
            int((i + 1) / 2) == self.p
        ), "Given data and number of components dont match"

    def _mean(self, means, time=None):
        """Evaluate output mean functions.

        Args:
            means: Mean functions for each output.
            time: Times at which to evaluate the means.

        Returns:
            Concatenated mean-function values.
        """
        if time is None:
            N = self.time.size
            m = np.zeros_like(self.tt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    m[i * N : (i + 1) * N] = meanfun(self.time)
        else:
            N = time.size
            tt = np.tile(time, self.p)
            m = np.zeros_like(tt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    m[i * N : (i + 1) * N] = meanfun(time)
        return m

    def _kernel_matrix(self, kernel, time=None):
        """Build a covariance matrix from a kernel and input times.

        Args:
            kernel: Covariance function.
            time: Input times.

        Returns:
            Kernel covariance matrix.
        """
        r = time[:, None] - time[None, :]
        K = kernel(r) + 1e-6 * np.diag(np.diag(np.ones_like(r)))
        K[np.abs(K) < 1e-12] = 0.0
        return K

    def _predict_kernel_matrix(self, kernel, time):
        """Build the cross-covariance matrix used for prediction.

        Args:
            kernel: Covariance function.
            time: Prediction times.

        Returns:
            Cross-covariance matrix.
        """
        if time.size == 1:
            r = time - self.time[None, :]
        else:
            r = time[:, None] - self.time[None, :]
        K = kernel(r)
        return K

    def _u_to_fhatW(self, u):
        """Split a flattened latent vector into node and weight blocks.

        Args:
            u: Flattened latent vector.

        Returns:
            Node and weight arrays.
        """
        f = u[: self.q * self.N].reshape((1, self.q, self.N))
        w = u[self.q * self.N :].reshape((self.p, self.q, self.N))
        return f, w

    def _cholesky_with_nugget(self, matrix, maximum=10):
        """Compute a Cholesky factor, increasing diagonal jitter if needed.

        Args:
            matrix: Matrix to decompose.
            maximum: Maximum number of jitter increases.

        Returns:
            The Cholesky factor and the final nugget.
        """
        nugget = 0  # our nugget starts as zero
        try:
            nugget += 1e-15
            L = cholesky(matrix, lower=True, overwrite_a=True)
            return L, nugget
        except LinAlgError:
            n = 0  # number of tries
            while n < maximum:
                try:
                    L = cholesky(
                        matrix + nugget * np.identity(matrix.shape[0]),
                        lower=True,
                        overwrite_a=True,
                    )
                    return L, nugget
                except LinAlgError:
                    nugget *= 10.0
                finally:
                    n += 1
            raise LinAlgError("Not positive definite, even with nugget.")

    def sample_from_kernel(self, latentFunc, time=None):
        """Draw one sample from a latent kernel.

        Args:
            latentFunc: Covariance function to sample from.
            time: Time coordinates. Uses the training times when omitted.

        Returns:
            A sample from the Gaussian process prior.
        """
        # print(latentFunc)
        if time is None:
            time = self.time
        mean = np.zeros_like(time)
        K = self._kernel_matrix(latentFunc, time)
        normal = multivariate_normal(mean, K, allow_singular=True).rvs()
        return normal

    def calculate_elbo(self, nodes, weights, meanf, jitters, iterations=10000):
        """Calculate the evidence lower bound.

        Args:
            nodes: Node kernels.
            weights: Weight kernels.
            meanf: Mean functions.
            jitters: Jitter terms.
            iterations: Maximum number of update iterations.

        Returns:
            The ELBO, variational means, and variational variances.
        """
        # initial variational parameters
        mu = np.random.rand(self.d, self.k).T
        var = np.ones_like(np.random.rand(1, self.k).T)  # why?
        muF, muW = [], []
        for k in range(self.k):
            m1, m2 = self._u_to_fhatW(mu[k, :])
            muF.append(m1)
            muW.append(m2)
        muF = np.array(muF)
        muW = np.array(muW)
        ELBO = self._calculate_elbo_terms(nodes, weights, meanf, jitters, mu, var)
        elboArray = np.array([ELBO])  # To add new elbo values inside
        iterNumber = 1
        while iterNumber < iterations:
            # ELBO = self._calculate_elbo_terms(nodes, weights, meanf, jitters, mu, var)
            ELBO, mu, var = self.update_mu_and_var(
                nodes, weights, meanf, jitters, mu, var
            )
            elboArray = np.append(elboArray, ELBO)
            iterNumber += 1
            # Stoping criteria:
            if iterNumber > 5:
                means = np.mean(elboArray[-5:])
                criteria = np.abs(np.std(elboArray[-5:]) / means)
                if criteria < 1e-3 and criteria != 0:
                    return ELBO, mu, var
        print("Max iterations reached")
        return ELBO, mu, var

    def _calculate_elbo_terms(self, nodes, weights, meanf, jitters, mu, var):
        """Compute ELBO terms for the nonparametric approximation.

        Args:
            nodes: Node kernels.
            weights: Weight kernels.
            meanf: Mean functions.
            jitters: Jitter terms.
            mu: Variational means.
            var: Variational variances.

        Returns:
            Evidence lower bound value.
        """
        muF, muW = [], []
        for k in range(self.k):
            m1, m2 = self._u_to_fhatW(mu[k, :])
            muF.append(m1)
            muW.append(m2)
        muF = np.array(muF)
        muW = np.array(muW)
        # nodes and means
        Kf = np.array([self._kernel_matrix(i, self.time) for i in nodes])
        invKf = np.array([inv(i) for i in Kf])
        Lf = np.array([self._cholesky_with_nugget(i)[0] for i in Kf])
        Kw = np.array([self._kernel_matrix(j, self.time) for j in weights])
        invKw = np.array([inv(j) for j in Kw])
        Lw = np.array([self._cholesky_with_nugget(j)[0] for j in Kw])
        # Entropy
        Entropy = self._entropy(mu, var)
        # print('entropy:', Entropy)
        # Expected log-likelihood
        ExpLoglike = self._expectedLogLike(
            nodes, weights, meanf, jitters, muF, muW, var
        )
        # print('expLL:', ExpLoglike)
        # Expected log prior
        ExpLogprior = self._expectedLogPrior(
            Kf, invKf, Lf, Kw, invKw, Lw, muF, muW, var, jitters
        )
        # print('expLP:', ExpLogprior)
        # print('expLL+expLP:', np.sum(ExpLoglike + ExpLogprior)/self.k)
        ELBO = np.sum(ExpLoglike + ExpLogprior) / self.k - Entropy
        return ELBO

    def _entropy(self, mu, var):
        varmin = 1e-7
        beta = np.ones((self.k, 1)) / self.k
        S0 = np.array(mu - np.mean(mu, axis=0)).T
        S = np.sum(S0 * S0, axis=0) - 2 * (S0.T @ S0)
        S = np.sum(S0 * S0, axis=0) + S.T
        S[S < 0] = 0  # numerical noise can cause it to negative
        var = var**2 + varmin
        s = var[:, None] + var[None, :]
        logP = -0.5 * S / s - 0.5 * self.d * np.log(s)
        logP[logP < 0] = 0  # numerical noise can cause it to negative
        a = np.zeros((self.k, 1))
        for i in range(self.k):
            a[i] = -np.log(self.k) + np.log(np.sum(np.exp(logP[0, i])))
        entropy = (a.T @ beta).item()
        return entropy

    def _expectedLogLike(self, nodes, weights, means, jitters, muF, muW, var):
        new_y = np.concatenate(self.y) - self._mean(means, self.time)
        new_y = np.array(np.array_split(new_y, self.p)).T
        jitt2 = np.array(jitters) ** 2
        errs = 0
        for i in range(self.p):
            errs += jitt2[i] + self.yerr2[i]
        ### first term of equation 3.22
        Wblk = []
        for k in range(self.k):
            Wblk.append(np.squeeze(muW[k]))
        Wblk = np.array(Wblk)
        Fblk = []
        for k in range(self.k):
            Fblk.append(np.squeeze(muF[k]))
        Fblk = np.array(Fblk)
        Ymean = Wblk * Fblk
        Ydiff = (new_y.T - Ymean) ** 2 / errs
        logl = -0.5 * np.sum(Ydiff, axis=1)
        ### second term of equation 3.22
        kvals = []
        for k in range(self.k):
            value = 0
            for q in range(self.q):
                for p in range(self.p):
                    value = (muF[k, :, q, :] * muF[k, :, q, :]) / (
                        jitt2[p] + self.yerr2[p, :]
                    )
                    value += (muW[k, p, :, :] * muW[k, p, :, :]) / (
                        jitt2[p] + self.yerr2[p, :]
                    )
                    value += (
                        var[k] ** 4 * self.q / (jitt2[p] + np.sum(self.yerr2[p, :]))
                    )
                    kvals.append(self.p * var[k] ** 2 * np.sum(value))
        kvals = np.array(np.squeeze(kvals))
        logl += -0.5 * kvals
        ### third term of equation 3.22
        value = 0
        for p in range(self.p):
            for n in range(self.N):
                value += np.log(2 * np.pi * (jitt2[p] + self.yerr2[p, n]))
        logl += -0.5 * value
        return logl

    def _expectedLogPrior(self, Kf, invKf, Lf, Kw, invKw, Lw, muF, muW, var, jitters):
        ### first term
        logKf = [2 * np.sum(np.log(np.diag(i))) for i in Lf]
        logKw = [2 * np.sum(np.log(np.diag(i))) for i in Lw]
        logprior = -0.5 * np.sum(logKf) - 0.5 * np.sum(logKw)
        ### second term
        sum_kj, sum_kw = [], []
        for k in range(self.k):
            for q in range(self.q):
                mKfm = muF[k, :, q, :] @ invKf[q] @ muF[k, :, q, :].T
                vartracef = var[k] ** 2 * np.trace(invKf[q])
                for p in range(self.p):
                    # almost certain this will fail for more than 1 node
                    mKwm = muW[k, p, q, :] @ invKw[p] @ muW[k, p, q, :].T
                    vartracew = var[k] ** 2 * np.trace(invKw[p])
            sum_kj.append(np.asarray(mKfm + vartracef).item())
            sum_kw.append(np.asarray(mKwm + vartracew).item())
        logprior += -0.5 * np.array(sum_kj) - 0.5 * np.array(sum_kw)
        return logprior

    def update_mu_and_var(self, nodes, weights, meanf, jitters, mu, var):
        res1 = minimize(
            self._update_mu,
            x0=mu,
            args=(nodes, weights, meanf, jitters, var),
            method="Nelder-Mead",
            options={"disp": False, "maxiter": 200},
        )
        mu = res1.x
        res2 = minimize(
            self._update_var,
            x0=var,
            args=(nodes, weights, meanf, jitters, mu),
            method="Nelder-Mead",
            options={"disp": False, "maxiter": 200},
        )
        var = res2.x
        mu = mu.reshape(self.k, self.d)
        ELBO = self._calculate_elbo_terms(nodes, weights, meanf, jitters, mu, var)
        return ELBO, mu, var

    def _update_mu(self, mu, nodes, weights, meanf, jitters, var):
        mu = mu.reshape(self.k, self.d)
        e = -self._calculate_elbo_terms(nodes, weights, meanf, jitters, mu, var)
        return e

    def _update_var(self, var, nodes, weights, meanf, jitters, mu):
        mu = mu.reshape(self.k, self.d)
        e = -self._calculate_elbo_terms(nodes, weights, meanf, jitters, mu, var)
        return e

    def _squaredDistance(self, X):
        m, n = X.shape
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = np.linalg.norm(X[:, i] - X[:, j]) ** 2
                D[j, i] = D[i, j]
        return D

    ELBOcalc = calculate_elbo
    ELBOaux = _calculate_elbo_terms
    sampleIt = sample_from_kernel
    _kernelMatrix = _kernel_matrix
    _predictKMatrix = _predict_kernel_matrix
    _cholNugget = _cholesky_with_nugget
    updateMUandVAR = update_mu_and_var
    _updateMU = _update_mu
    _updateVAR = _update_var


inference = NonparametricInference
