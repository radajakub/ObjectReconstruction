from __future__ import annotations

import os

import numpy as np
from models import Model
from inout import Logger, RANSACLogEntry


class Estimate:
    @staticmethod
    def load(folder: str) -> Estimate:
        if not (os.path.exists(folder) and os.path.isdir(folder)):
            raise FileNotFoundError(f'folder {folder} does not exist or is not a directory')
        M = np.loadtxt(os.path.join(folder, 'M.txt'))
        inliers = np.loadtxt(os.path.join(folder, 'inliers.txt'), dtype=int)
        return Estimate(M, inliers)

    def __init__(self, M: np.ndarray, inlier_indices: np.ndarray) -> None:
        self.M = M
        self.inlier_indices = inlier_indices

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        np.savetxt(os.path.join(folder, 'M.txt'), self.M)
        np.savetxt(os.path.join(folder, 'inliers.txt'), self.inlier_indices)


class RANSAC:
    def __init__(self, model: Model, threshold: float = 3, p=0.999, max_iterations=1000, rng: np.random.Generator = None, logger: Logger = None) -> None:
        self.model = model
        self.threshold = threshold
        self.p = p
        self.max_iterations = max_iterations
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self.logger = logger

    def _criterion(self, w: float) -> int:
        return np.log(1 - self.p) / np.log(1 - np.power(w, self.model.min_samples))

    def fit(self, X: np.ndarray) -> None:
        self.it = 0
        Nmax = 1

        best_support = 0
        best_M = None
        N = X.shape[1]

        while self.it <= Nmax and self.it < self.max_iterations:
            self.it += 1

            # sample
            sample = X[:, self.rng.choice(N, self.model.min_samples, replace=False)]
            # construct hypothesis
            Mk = self.model.hypothesis(sample)
            # evaluate error
            eps = self.model.error(Mk, X)
            # compute inliers
            inlier_indices = eps < self.threshold
            inliers = X[:, inlier_indices]
            Ni = inliers.shape[1]
            # compute support
            supp = self.model.support(eps[inlier_indices])

            self.logger.log(RANSACLogEntry(self.it, Ni, supp, Nmax))

            if supp > best_support:
                best_support = supp
                best_M = Mk
                Nmax = self._criterion(Ni / N)

        # fit to best inliers
        best_M = self.model.hypothesis(X[:, self.model.error(best_M, X) < self.threshold])

        return Estimate(best_M, X[:, self.model.error(best_M, X) < self.threshold])
