import numpy as np

from ransac import RANSAC
from models import EssentialMatrix
import toolbox as tb
from logger import Logger, LogEntry
from config import Config


class EpipolarEstimator(RANSAC):
    def __init__(self, K: np.ndarray, threshold: float = Config.default_threshold, p: float = Config.default_p, max_iterations: int = Config.default_max_iter, rng: np.random.Generator = None, logger: Logger = None) -> None:
        super().__init__(model=EssentialMatrix(K), threshold=threshold,
                         p=p, max_iterations=max_iterations, rng=rng)
        self.logger = logger

    def compute_epipolar_lines(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.estimate is None:
            raise AttributeError("estimate not computed yet")
        F = self.model.get_fundamental(self.estimate)
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
                inliers = eps < self.threshold
                inlier_eps = eps[inliers]
                supp = self.model.support(inlier_eps, threshold=self.threshold)
                if supp <= best_supp:
                    self.logger.log(LogEntry(iteration=self.it, inliers=inliers.sum(), support=supp, visible=-1, Nmax=Nmax))
                    continue

                # compute inlier indices
                inlier_indices = np.arange(X1.shape[1])[inliers]

                # check visibility of inliers
                P1 = np.eye(3, 4)
                P2 = np.hstack((R, t))  # rotated and moved camera
                X1_ = self.model.unapply_K(X1[:, inliers])
                X2_ = self.model.unapply_K(X2[:, inliers])
                X = tb.Pu2X(P1, P2, X1_, X2_)
                u1p = P1 @ X
                u2p = P2 @ X
                visible = np.logical_and(u1p[2] > 0, u2p[2] > 0)
                visible_indices = inlier_indices[visible]

                supp = self.model.support(inlier_eps[visible], threshold=self.threshold)
                Ni = visible_indices.shape[0]

                if supp > best_supp:
                    best_supp = supp
                    best_E = E
                    best_inlier_indices = visible_indices
                    Nmax = self._criterion(Ni / N)
                    self.logger.log_improve(LogEntry(iteration=self.it, inliers=inlier_indices.shape[0], support=supp, visible=Ni, Nmax=Nmax))

                self.logger.log(LogEntry(iteration=self.it, inliers=inlier_indices.shape[0], support=supp, visible=Ni, Nmax=Nmax))

        self.estimate = best_E
        self.inliers = best_inlier_indices
