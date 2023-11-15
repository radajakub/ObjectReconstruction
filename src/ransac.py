from typing import Callable
import numpy as np
from models import Model, Linear
from plotter import Plotter


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

        self.estimate = None
        self.inliers = None

    def _criterion(self, w: float) -> int:
        return np.log(
            1 - self.p) / np.log(1 - np.power(w, self.model.min_samples))

    def fit(self, X: np.ndarray) -> None:
        self.result = None
        self.inliers = None

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
        M = self.model.hypothesis(
            X[:, self.model.error(best_M, X) < self.threshold])

        self.estimate = M
        self.inliers = X[:, self.model.error(
            self.estimate, X) < self.threshold]


if __name__ == "__main__":
    print('Testing RANSAC on line fitting...')
    X = np.loadtxt('test_data/ransac.txt').T
    orig_line = np.array([-10, 3, 1200])[:, np.newaxis]

    plotter = Plotter(rows=1, cols=2, hide_axes=True,
                      invert_yaxis=False, aspect_equal=True, labels=[['Only points', 'Points with fitted line']])
    plotter.add_points(X, col=1)
    plotter.add_line(orig_line, col=1, color='red')
    plotter.add_points(X, col=2)
    plotter.add_line(orig_line, col=2, color='red')

    ransac = RANSAC(model=Linear(), max_iterations=1000,
                    threshold=3, p=0.999, rng=np.random.default_rng(0))

    ransac.fit(X)
    plotter.add_line(ransac.estimate, col=2, color='blue')

    print(
        f'RANSAC in {ransac.it} iterations found line with parameters: {ransac.estimate.flatten()}')

    plotter.show()
