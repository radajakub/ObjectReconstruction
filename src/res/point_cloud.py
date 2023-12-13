import numpy as np
import os

import utils.toolbox as tb
from .camera import Camera
from packages import ge as ge


class PointCloud:
    def __init__(self, K: np.ndarray) -> None:
        self.points = None
        self.K = K
        self.K_ = np.linalg.inv(K)

    def get_size(self) -> int:
        if self.points is None:
            return 0
        return self.points.shape[1]

    def add(self, P1: Camera, P2: Camera, corr1: np.ndarray, corr2: np.ndarray) -> np.ndarray:
        if corr1.shape[0] == 2:
            corr1 = tb.e2p(corr1)
        if corr2.shape[0] == 2:
            corr2 = tb.e2p(corr2)

        assert (corr1.shape[0] == 3)
        assert (corr2.shape[0] == 3)

        corr1, corr2 = tb.u_correct_sampson(Camera.get_fundamental(P1, P2), corr1, corr2)
        points_3d = tb.Pu2X(P1.P, P2.P, corr1, corr2)

        prev_l = 0 if self.points is None else self.points.shape[1]

        if self.points is None:
            self.points = points_3d
        else:
            self.points = np.hstack((self.points, points_3d))

        curr_l = self.points.shape[1]
        return np.arange(prev_l, curr_l)

    def add_F(self, F: np.ndarray, P1: Camera, P2: Camera, corr1: np.ndarray, corr2: np.ndarray) -> np.ndarray:
        assert (corr1.shape[0] == 3)
        assert (corr2.shape[0] == 3)

        corr1, corr2 = tb.u_correct_sampson(F, corr1, corr2)

        points_3d = tb.Pu2X(P1.P, P2.P, corr1, corr2)

        prev_l = 0 if self.points is None else self.points.shape[1]

        if self.points is None:
            self.points = points_3d
        else:
            self.points = np.hstack((self.points, points_3d))

        curr_l = self.points.shape[1]
        return np.arange(prev_l, curr_l)

    def get_points(self, indices: np.ndarray) -> np.ndarray:
        return tb.p2e(self.points[:, indices])

    def get_all(self) -> np.ndarray:
        return tb.p2e(self.points)

    def save(self, outpath: str, name: str = 'points') -> str:
        outpath = os.path.join(outpath, 'point_cloud')
        os.makedirs(outpath, exist_ok=True)
        # export to ply
        g = ge.GePly(os.path.join(outpath, f'{name}.ply'))
        g.points(self.get_all())
        g.close()
        # save numpy points
        np.savetxt(os.path.join(outpath, f'{name}.txt'), self.get_all())
