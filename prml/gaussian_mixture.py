import numpy as np
from scipy.stats import multivariate_normal
from typing import Optional


class GaussianMixture:
    """
    Gaussian mixture

    N: num of data
    D: dimension of each data
    K: num of cluster

    X.shape = (N, D)
    mu.shape = (K, D)
    sigma.shape = (K, D, D)
    pi.shape = (K,)
    prob.shape = (N, K)
    resp.shape = (N, K)
    """

    def __init__(self, n_components: int, random_state: int = 42,
                 max_iter: int = 100, verbose: int = 0,
                 init_params: str = "random",
                 means_init: Optional[np.array] = None) -> None:
        np.random.seed(random_state)
        if n_components <= 0:
            raise ValueError("n_components must be larger than 0")
        self.n_components = n_components
        self.max_iter = max_iter
        self.verbose = verbose
        self.init_params = init_params
        self.mu = means_init

    def _initialize(self, X) -> None:
        """
        initialization methods
        mu -> uniform from each dim
        sigma -> identity matrix
        pi -> uniform
        """
        if self.mu is None:
            idx = np.random.choice(
                np.arange(X.shape[0]), size=self.n_components, replace=False)
            self.mu = X[idx]
        self.sigma = np.array([np.eye(X.shape[1])
                               for _ in range(self.n_components)])
        self.pi = np.ones(self.n_components) / self.n_components
        self.prob = []
        for i in range(self.n_components):
            self.prob.append(multivariate_normal.pdf(
                X, mean=self.mu[i], cov=self.sigma[i]))
        self.prob = np.asarray(self.prob).T

    def fit(self, X, y=None) -> None:
        num_data, _ = X.shape
        if num_data < self.n_components:
            raise ValueError("size of X must be smaller than n_components")

        self._initialize(X)
        for i in range(self.max_iter):
            # E-step
            self.resp = self.pi * self.prob
            self.resp /= self.resp.sum(axis=1, keepdims=True)
            # M-step
            Nk = self.resp.sum(axis=0)
            self.mu = np.dot(self.resp.T, X) / Nk[:, np.newaxis]
            self.sigma = np.empty(self.sigma.shape)
            for k in range(self.n_components):
                diff = X - self.mu[k]
                self.sigma[k] = np.dot(self.resp[:, k] * diff.T, diff) / Nk[k]
            self.pi = Nk / num_data
            if self.verbose:
                print(f"[step {i+1}] loglike =", self.loglike())

    def loglike(self):
        return np.log((self.pi * self.prob).sum(axis=1)).sum()

    def get_hard_cluster(self):
        return self.resp.argmax(axis=1)
