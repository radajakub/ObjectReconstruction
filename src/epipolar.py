import numpy as np

from ransac import RANSAC
from models import EssentialMatrix
import toolbox as tb


class EpipolarEstimator(RANSAC):
    def __init__(self, K: np.ndarray, threshold: float = 3, p: float = 0.9999, max_iterations: int = 200, rng: np.random.Generator = None) -> None:
        super().__init__(model=EssentialMatrix(K), threshold=threshold,
                         p=p, max_iterations=max_iterations, rng=rng)

    def compute_epipolar_lines(self, E: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        F = self.model.get_fundamental(E)
        l1 = F.T @ tb.e2p(x2[:, np.newaxis])
        l2 = F @ tb.e2p(x1[:, np.newaxis])
        return l1, l2

    def fit(self, X1: np.ndarray, X2: np.ndarray) -> None:
        assert (X1.shape == X2.shape)
        X1 = tb.e2p(X1)
        X2 = tb.e2p(X2)

        self.estimate = None
        self.inliers = None

        self.it = 0
        Nmax = np.inf

        best_supp = 0
        best_E = None
        best_inlier_indices = None
        N = X1.shape[1]

        while self.it <= Nmax and self.it < self.max_iterations:
            self.it += 1

            # sample
            sample_indices = self.rng.choice(N, self.model.min_samples, replace=False)
            X1_sample = self.model.unapply_K(X1[:, sample_indices])
            X2_sample = self.model.unapply_K(X2[:, sample_indices])

            # construct hypothesis
            Es = self.model.hypothesis(X1_sample, X2_sample)
            # verify every essential matrix
            Eidx = 0
            for E in Es:
                Eidx += 1

                # verify visibility of samples and get R,t
                R, t = self.model.decompose(E, X1_sample, X2_sample)
                if R.size == 0 or t.size == 0:
                    continue

                # compute support with sampson error (without visibility yet)
                eps = self.model.error(E, X1, X2)
                supp = self.model.support(eps[eps < self.threshold], threshold=self.threshold)
                if supp <= best_supp:
                    continue

                # compute inliers from visible points
                eps = self.model.error(E, X1, X2)
                inliers = eps < self.threshold
                inlier_indices = np.arange(X1.shape[1])[inliers]
                supp = self.model.support(eps[inliers], threshold=self.threshold)
                Ni = inlier_indices.shape[0]

                if supp > best_supp:
                    best_supp = supp
                    best_E = E
                    best_inlier_indices = inlier_indices
                    Nmax = self._criterion(Ni / N)

        self.estimate = best_E
        self.inliers = best_inlier_indices
