from __future__ import annotations
import os

import numpy as np

from .camera import Camera
import utils.toolbox as tb
from packages import ge as ge
from inout import Logger, ActionLogEntry


class PointCloud:
    FOLDER_NAME = 'point_cloud'
    SPARSE_NAME = 'sparse'
    DENSE_NAME = 'dense'

    def __init__(self, K: np.ndarray, logger: Logger) -> None:
        self.sparse = None
        self.dense = None
        self.K = K
        self.K_ = np.linalg.inv(K)
        self.logger = logger

    def load(self, outpath: str) -> None:
        outpath = os.path.join(outpath, PointCloud.FOLDER_NAME)
        name = os.path.join(outpath, f'{PointCloud.SPARSE_NAME}.txt')
        if os.path.exists(name):
            self.sparse = tb.e2p(np.loadtxt(name))
        name = os.path.join(outpath, f'{PointCloud.DENSE_NAME}.txt')
        if os.path.exists(name):
            self.dense = tb.e2p(np.loadtxt(name))
        self.logger.log(ActionLogEntry(f'Loaded point cloud from {name}'))

    def get_size(self) -> int:
        if self.sparse is None:
            return 0
        return self.sparse.shape[1]

    def add_dense(self, P1: Camera, P2: Camera, corr1: np.ndarray, corr2: np.ndarray) -> int:
        assert (corr1.shape[0] == 3)
        assert (corr2.shape[0] == 3)

        corr1, corr2 = tb.u_correct_sampson(Camera.get_fundamental(P1, P2), corr1, corr2)
        points_3d = tb.Pu2X(P1.P, P2.P, corr1, corr2)
        if self.dense is None:
            self.dense = points_3d
        else:
            self.dense = np.hstack((self.dense, points_3d))
        return points_3d.shape[1]

    def add(self, P1: Camera, P2: Camera, corr1: np.ndarray, corr2: np.ndarray) -> np.ndarray:
        if corr1.shape[0] == 2:
            corr1 = tb.e2p(corr1)
        if corr2.shape[0] == 2:
            corr2 = tb.e2p(corr2)

        assert (corr1.shape[0] == 3)
        assert (corr2.shape[0] == 3)

        corr1, corr2 = tb.u_correct_sampson(Camera.get_fundamental(P1, P2), corr1, corr2)
        points_3d = tb.Pu2X(P1.P, P2.P, corr1, corr2)

        prev_l = 0 if self.sparse is None else self.sparse.shape[1]

        if self.sparse is None:
            self.sparse = points_3d
        else:
            self.sparse = np.hstack((self.sparse, points_3d))

        curr_l = self.sparse.shape[1]
        return np.arange(prev_l, curr_l)

    def add_F(self, F: np.ndarray, P1: Camera, P2: Camera, corr1: np.ndarray, corr2: np.ndarray) -> np.ndarray:
        assert (corr1.shape[0] == 3)
        assert (corr2.shape[0] == 3)

        corr1, corr2 = tb.u_correct_sampson(F, corr1, corr2)

        points_3d = tb.Pu2X(P1.P, P2.P, corr1, corr2)

        prev_l = 0 if self.sparse is None else self.sparse.shape[1]

        if self.sparse is None:
            self.sparse = points_3d
        else:
            self.sparse = np.hstack((self.sparse, points_3d))

        curr_l = self.sparse.shape[1]
        return np.arange(prev_l, curr_l)

    def sparse_get_points(self, indices: np.ndarray) -> np.ndarray:
        return tb.p2e(self.sparse[:, indices])

    def sparse_get_all(self) -> np.ndarray:
        return tb.p2e(self.sparse)

    def dense_get_points(self, indices: np.ndarray) -> np.ndarray:
        return tb.p2e(self.dense[:, indices])

    def dense_get_all(self) -> np.ndarray:
        return tb.p2e(self.dense)

    def get_all(self) -> np.ndarray:
        if self.sparse is None:
            return self.dense_get_all()
        if self.dense is None:
            return self.sparse_get_all()
        return tb.p2e(np.hstack((self.sparse, self.dense)))

    def save(self, outpath: str, export: bool = True) -> str:
        outpath = os.path.join(outpath, PointCloud.FOLDER_NAME)
        os.makedirs(outpath, exist_ok=True)
        # save numpy points
        if self.sparse is not None:
            name = os.path.join(outpath, f'{PointCloud.SPARSE_NAME}.txt')
            np.savetxt(name, self.sparse_get_all())
        if self.dense is not None:
            name = os.path.join(outpath, f'{PointCloud.DENSE_NAME}.txt')
            np.savetxt(name, self.dense_get_all())
        self.logger.log(ActionLogEntry(f'Saved point cloud to {name}'))
        if export:
            # export to ply
            if self.sparse is not None:
                name = os.path.join(outpath, f'{PointCloud.SPARSE_NAME}.ply')
                g = ge.GePly(name)
                g.points(self.sparse_get_all())
                g.close()
                self.logger.log(ActionLogEntry(f'Exported sparse point cloud to {name}'))
            if self.dense is not None:
                name = os.path.join(outpath, f'{PointCloud.DENSE_NAME}.ply')
                g = ge.GePly(name)
                g.points(self.dense_get_all())
                g.close()
                self.logger.log(ActionLogEntry(f'Exported dense point cloud to {name}'))
