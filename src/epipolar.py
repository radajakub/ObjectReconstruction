from __future__ import annotations

import numpy as np
import os

from ransac import RANSAC, Estimate
from models import EssentialMatrix
import toolbox as tb
from logger import Logger, EpipolarEstimateLogEntry
from config import Config

DATAFOLDER = 'data'


class EpipolarEstimate(Estimate):
    @staticmethod
    def load(folder: str) -> EpipolarEstimate:
        if not (os.path.exists(os.path.join(folder, DATAFOLDER)) and os.path.isdir(os.path.join(folder, DATAFOLDER))):
            raise FileNotFoundError(f'folder {folder} does not exist or is not a directory')
        E = np.loadtxt(os.path.join(folder, DATAFOLDER, 'E.txt'))
        R = np.loadtxt(os.path.join(folder, DATAFOLDER, 'R.txt'))
        t = np.loadtxt(os.path.join(folder, DATAFOLDER, 't.txt'))
        inliers = np.loadtxt(os.path.join(folder, DATAFOLDER, 'inliers.txt'), dtype=int)
        return EpipolarEstimate(E, R, t, inliers)

    def __init__(self, E: np.ndarray, R: np.ndarray, t: np.ndarray, inlier_indices: np.ndarray) -> None:
        self.E = E
        self.R = R
        self.t = t
        self.inlier_indices = inlier_indices

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        np.savetxt(os.path.join(folder, DATAFOLDER, 'E.txt'), self.E)
        np.savetxt(os.path.join(folder, DATAFOLDER, 'R.txt'), self.R)
        np.savetxt(os.path.join(folder, DATAFOLDER, 't.txt'), self.t)
        np.savetxt(os.path.join(folder, DATAFOLDER, 'inliers.txt'), self.inlier_indices)


class EpipolarEstimator(RANSAC):
    def __init__(self, K: np.ndarray, threshold: float = Config.default_threshold, p: float = Config.default_p, max_iterations: int = Config.default_max_iter, rng: np.random.Generator = None, logger: Logger = None) -> None:
        super().__init__(model=EssentialMatrix(K), threshold=threshold,
                         p=p, max_iterations=max_iterations, rng=rng)
        self.logger = logger

    def compute_epipolar_lines(self, estimate: EpipolarEstimate, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        F = self.model.get_fundamental(estimate.E)
        l1 = F.T @ tb.e2p(x2[:, np.newaxis])
        l2 = F @ tb.e2p(x1[:, np.newaxis])
        return l1, l2

    def fit(self, X1: np.ndarray, X2: np.ndarray) -> EpipolarEstimate:
        assert (X1.shape == X2.shape)
        X1 = tb.e2p(X1)
        X2 = tb.e2p(X2)

        self.it = 0
        Nmax = np.inf

        best_supp = 0
        best_estimate = None
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
                    self.logger.log(EpipolarEstimateLogEntry(iteration=self.it, inliers=inliers.sum(), support=supp, visible=-1, Nmax=Nmax))
                    continue

                # compute inlier indices
                inlier_indices = np.arange(X1.shape[1])[inliers]

                # check visibility of inliers
                P1 = np.eye(3, 4)
                P2 = np.hstack((R, t))  # rotated and moved camera
                X1_ = self.model.unapply_K(X1[:, inliers])
                X2_ = self.model.unapply_K(X2[:, inliers])
                X = tb.Pu2X(P1, P2, X1_, X2_)
                X = tb.e2p(tb.p2e(X))
                u1p = P1 @ X
                u2p = P2 @ X
                visible = np.logical_and(u1p[2] > 0, u2p[2] > 0)
                visible_indices = inlier_indices[visible]

                supp = self.model.support(inlier_eps[visible], threshold=self.threshold)
                Ni = visible_indices.shape[0]

                if supp > best_supp:
                    best_supp = supp
                    best_estimate = EpipolarEstimate(E, R, t, visible_indices)
                    Nmax = self._criterion(Ni / N)
                    self.logger.log_improve(EpipolarEstimateLogEntry(iteration=self.it, inliers=inlier_indices.shape[0], support=supp, visible=Ni, Nmax=Nmax))

                self.logger.log(EpipolarEstimateLogEntry(iteration=self.it, inliers=inlier_indices.shape[0], support=supp, visible=Ni, Nmax=Nmax))

        return best_estimate
