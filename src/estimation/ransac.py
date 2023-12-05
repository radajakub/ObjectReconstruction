from __future__ import annotations

import os

import numpy as np
from models import Model, Linear
from data import Plotter

DATAFOLDER = 'data'


class Estimate:
    @staticmethod
    def load(folder: str) -> Estimate:
        if not (os.path.exists(os.path.join(folder, DATAFOLDER)) and os.path.isdir(os.path.join(folder, DATAFOLDER))):
            raise FileNotFoundError(f'folder {folder} does not exist or is not a directory')
        M = np.loadtxt(os.path.join(folder, DATAFOLDER, 'M.txt'))
        inliers = np.loadtxt(os.path.join(folder, DATAFOLDER, 'inliers.txt'), dtype=int)
        return Estimate(M, inliers)

    def __init__(self, M: np.ndarray, inlier_indices: np.ndarray) -> None:
        self.M = M
        self.inlier_indices = inlier_indices

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        np.savetxt(os.path.join(folder, DATAFOLDER, 'M.txt'), self.M)
        np.savetxt(os.path.join(folder, DATAFOLDER, 'inliers.txt'), self.inlier_indices)


class RANSAC:
    def __init__(self, model: Model, threshold: float = 3, p=0.999, max_iterations=1000, rng: np.random.Generator = None) -> None:
        self.model = model
        self.threshold = threshold
        self.p = p
        self.max_iterations = max_iterations
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

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
            sample = X[:, self.rng.choice(
                N, self.model.min_samples, replace=False)]
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

            if supp > best_support:
                best_support = supp
                best_M = Mk
                Nmax = self._criterion(Ni / N)

        # fit to best inliers
        best_M = self.model.hypothesis(X[:, self.model.error(best_M, X) < self.threshold])

        return Estimate(best_M, X[:, self.model.error(best_M, X) < self.threshold])


if __name__ == "__main__":
    print('Testing RANSAC on line fitting...')
    X = np.loadtxt('data/ransac/ransac.txt').T
    orig_line = np.array([-10, 3, 1200])[:, np.newaxis]

    plotter = Plotter(rows=1, cols=2, hide_axes=True, invert_yaxis=False, aspect_equal=True)
    plotter.set_title('Only points', row=1, col=1)
    plotter.set_title('Points with fitted line', row=1, col=2)
    plotter.add_points(X, col=1)
    plotter.add_line(orig_line, col=1, color='red')
    plotter.add_points(X, col=2)
    plotter.add_line(orig_line, col=2, color='red')

    ransac = RANSAC(model=Linear(), max_iterations=1000, threshold=3, p=0.999, rng=np.random.default_rng(0))

    estimate = ransac.fit(X)
    plotter.add_line(estimate.M, col=2, color='blue')

    print(f'RANSAC in {ransac.it} iterations found line with parameters: {estimate.M.flatten()}')

    plotter.show()
