import numpy as np

import utils.toolbox as tb
from .epipolar import EpipolarEstimate


class PointCloud:
    def __init__(self) -> None:
        self.points = None
        self.num_points = 0

    def add(self, epipolar: EpipolarEstimate, corr1: np.ndarray, corr2: np.ndarray) -> None:
        if corr1.shape[0] == 2:
            corr1 = tb.e2p(corr1)
        if corr2.shape[0] == 2:
            corr2 = tb.e2p(corr2)

        assert (corr1.shape[0] == 3)
        assert (corr2.shape[0] == 3)

        corr1, corr2 = tb.u_correct_sampson(epipolar.model.get_fundamental(epipolar.E), corr1, corr2)

        P1, P2 = epipolar.get_cameras()
        points_3d = tb.Pu2X(P1, P2, corr1, corr2)

        if self.points is None:
            self.points = points_3d
        else:
            self.points = np.hstack((self.points, points_3d))

    def get_points(self) -> np.ndarray:
        return tb.p2e(self.points)
