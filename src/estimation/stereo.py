from __future__ import annotations

import os

import numpy as np
import scipy

from res import CameraSet, PointCloud
from inout import DataLoader, Logger, ActionLogEntry
from utils import toolbox as tb, Config


class StereoMatcher:
    pairs = [
        # horizontal pairs
        (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        # vertical pairs
        (1, 5), (2, 6), (3, 7), (4, 8),
        (5, 9), (6, 10), (7, 11), (8, 12),
    ]

    @staticmethod
    def load(config: Config, loader: DataLoader, logger: Logger) -> StereoMatcher:
        if config.inpath is None:
            raise ValueError('Cannot load StereoMatcher without inpath')
        outpath = config.inpath
        camera_set = CameraSet(loader, logger)
        camera_set.load(outpath)
        point_cloud = PointCloud(loader.K, logger)
        point_cloud.load(outpath)
        return StereoMatcher(config, loader, camera_set, point_cloud, logger)

    def __init__(self, config: Config, loader: DataLoader, camera_set: CameraSet, point_cloud: PointCloud, logger: Logger) -> None:
        self.config = config
        self.loader = loader
        self.camera_set = camera_set
        self.point_cloud = point_cloud
        self.logger = logger
        self.tasks = None
        self.disparities = None

    def prepare_disparities(self) -> None:
        self.disparities = None
        tasks = []
        for i1, i2 in StereoMatcher.pairs:
            # rectify images with homographies and fundamental
            F, H1, H2, img1_r, img2_r = self.camera_set.rectify(i1, i2)

            # load corresponding points
            u1, u2 = self.loader.get_corresp(i1, i2)

            # keep only inliers wrt F
            u1 = tb.e2p(u1)
            u2 = tb.e2p(u2)
            # TODO do sampson or abs
            vals = np.array([x.T @ F @ y for x, y in zip(u1.T, u2.T)])
            inlier_indices = vals < self.config.fundamental_threshold
            u1 = u1[:, inlier_indices]
            u2 = u2[:, inlier_indices]

            # rectify corresponding points
            u1_r = tb.p2e(H1 @ u1)
            u2_r = tb.p2e(H2 @ u2)

            seeds = np.vstack((u1_r[0, :], u2_r[0, :], (u1_r[1, :] + u1_r[1, :]) / 2)).T

            task_i = np.array([img1_r, img2_r, seeds], dtype=object)
            tasks += [task_i]

            self.logger.log(ActionLogEntry(f'Rectified images and points for cameras {i1} and {i2}'))

        self.tasks = np.vstack(tasks)

    def save(self) -> None:
        if self.config.outpath is None:
            raise ValueError('Cannot save StereoMatcher without outpath')
        outpath = self.config.outpath
        os.makedirs(outpath, exist_ok=True)
        self.camera_set.save(outpath)
        self.point_cloud.save(outpath)
        self.logger.log(ActionLogEntry(f'Saved StereoMatcher to {outpath}'))
        if self.tasks is not None:
            scipy.io.savemat(os.path.join(outpath, 'stereo_in.mat'), {'task': self.tasks})
            self.logger.log(ActionLogEntry(f'Saved tasks to {outpath}'))
