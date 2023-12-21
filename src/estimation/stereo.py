from __future__ import annotations

import os

import numpy as np
import scipy
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

from res import CameraSet, PointCloud
from inout import DataLoader, Logger, ActionLogEntry, Plotter
from utils import toolbox as tb, Config


class RectifiedData:
    @staticmethod
    def load(folder: str) -> RectifiedData:
        H1 = np.loadtxt(os.path.join(folder, 'H1.txt'))
        H2 = np.loadtxt(os.path.join(folder, 'H2.txt'))
        F = np.loadtxt(os.path.join(folder, 'F.txt'))
        u1 = np.loadtxt(os.path.join(folder, 'u1.txt'))
        u2 = np.loadtxt(os.path.join(folder, 'u2.txt'))
        img1 = np.array(Image.open(os.path.join(folder, 'img1.png')))
        img2 = np.array(Image.open(os.path.join(folder, 'img2.png')))
        return RectifiedData(img1, img2, H1, H2, F, u1, u2)

    def __init__(self, img1: Image, img2: Image, H1: np.ndarray, H2: np.ndarray, F: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> None:
        self.img1 = img1
        self.img2 = img2
        self.H1 = H1
        self.H2 = H2
        self.F = F
        self.u1 = u1
        self.u2 = u2

    def save(self, outpath: str) -> None:
        os.makedirs(outpath, exist_ok=True)
        np.savetxt(os.path.join(outpath, 'H1.txt'), self.H1)
        np.savetxt(os.path.join(outpath, 'H2.txt'), self.H2)
        np.savetxt(os.path.join(outpath, 'F.txt'), self.F)
        np.savetxt(os.path.join(outpath, 'u1.txt'), self.u1)
        np.savetxt(os.path.join(outpath, 'u2.txt'), self.u2)
        Image.fromarray(self.img1).save(os.path.join(outpath, 'img1.png'))
        Image.fromarray(self.img2).save(os.path.join(outpath, 'img2.png'))


class StereoMatcher:
    RECTIFIED_FOLDER = 'rectified'

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
    def get_file_name(i1: int, i2: int) -> str:
        return f'{i1}_{i2}'

    @staticmethod
    def extract_pair_from_name(name: str) -> tuple[int, int]:
        return tuple(int(x) for x in name.split('_'))

    @staticmethod
    def load(config: Config, loader: DataLoader, logger: Logger) -> StereoMatcher:
        if config.inpath is None:
            raise ValueError('Cannot load StereoMatcher without inpath')
        outpath = config.inpath
        camera_set = CameraSet(loader, logger)
        camera_set.load(outpath)
        logger.log(ActionLogEntry(f'Loaded camera set from {outpath}'))
        point_cloud = PointCloud(loader.K, logger)
        point_cloud.load(outpath)
        logger.log(ActionLogEntry(f'Loaded point cloud set from {outpath}'))

        stereo = StereoMatcher(config, loader, camera_set, point_cloud, logger)

        rectified_path = os.path.join(outpath, StereoMatcher.RECTIFIED_FOLDER)
        for dirname in os.listdir(rectified_path):
            if os.path.isdir(os.path.join(rectified_path, dirname)):
                logger.log(ActionLogEntry(f'Loading rectified data from {dirname}'))
                key = StereoMatcher.extract_pair_from_name(dirname)
                stereo.rectified[key] = RectifiedData.load(os.path.join(rectified_path, dirname))

        return stereo

    def __init__(self, config: Config, loader: DataLoader, camera_set: CameraSet, point_cloud: PointCloud, logger: Logger) -> None:
        self.config = config
        self.loader = loader
        self.camera_set = camera_set
        self.point_cloud = point_cloud
        self.logger = logger
        self.reset()

    def reset(self) -> None:
        self.tasks = None
        self.disparities = None
        self.rectified = {}

    def start_disparities(self) -> None:
        self.reset()
        tasks = []
        for i1, i2 in StereoMatcher.pairs:
            # rectify images with homographies and fundamental
            F, H1, H2, img1_r, img2_r = self.camera_set.rectify(i1, i2)

            # load corresponding points
            u1, u2 = self.loader.get_corresp(i1, i2)

            # keep only inliers wrt F
            u1 = tb.e2p(u1)
            u2 = tb.e2p(u2)
            vals = tb.err_F_sampson(F, u1, u2)
            inlier_indices = vals < self.config.fundamental_threshold
            u1 = u1[:, inlier_indices]
            u2 = u2[:, inlier_indices]

            self.rectified[(i1, i2)] = RectifiedData(img1_r, img2_r, H1, H2, F, u1, u2)

            self.logger.log(ActionLogEntry(f'Saved rectified data and transforms for cameras {i1} and {i2}'))

            # rectify corresponding points
            u1_r = tb.p2e(H1 @ u1)
            u2_r = tb.p2e(H2 @ u2)

            seeds = np.vstack((u1_r[0, :], u2_r[0, :], (u1_r[1, :] + u1_r[1, :]) / 2)).T

            task_i = np.array([img1_r, img2_r, seeds], dtype=object)
            tasks += [task_i]

            self.logger.log(ActionLogEntry(f'Prepared task for gsc for cameras {i1} and {i2}'))

        self.tasks = np.vstack(tasks)
        self.logger.log(ActionLogEntry(f'All gsc tasks prepared'))

    def load_disparities(self) -> None:
        self.disparities = {}
        for pair, disparity in zip(StereoMatcher.pairs, scipy.io.loadmat(os.path.join(self.config.inpath, StereoMatcher.RECTIFIED_FOLDER, 'stereo_out.mat'))['D'][:, 0]):
            self.disparities[pair] = disparity
            self.logger.log(ActionLogEntry(f'Load disparity for cameras {pair[0]} and {pair[1]}'))

    def fill_point_cloud(self) -> None:
        for i1, i2 in StereoMatcher.pairs:
            disparity = self.disparities[(i1, i2)]
            corr1 = []
            corr2 = []
            # compute correspondences as (x, y) and (x + D[y, x], y)
            for y, x in np.ndindex(disparity.shape):
                if not np.isnan(disparity[y, x]):
                    corr1.append([x, y])
                    corr2.append([x + disparity[y, x], y])
            corr1 = tb.e2p(np.array(corr1).T)
            corr2 = tb.e2p(np.array(corr2).T)
            # transform points back by inverse of rectifying homographies
            rectified_data = self.rectified[(i1, i2)]
            corr1 = np.linalg.inv(rectified_data.H1) @ corr1
            corr2 = np.linalg.inv(rectified_data.H2) @ corr2
            # add them to point cloud
            points = self.point_cloud.add_dense(self.camera_set.get_camera(
                i1), self.camera_set.get_camera(i2), corr1, corr2)
            self.logger.log(ActionLogEntry(f'Added {points} dense points for cameras {i1} and {i2}'))

    def get_horizontal_disparities(self) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
        return StereoMatcher.pairs[:9], [self.disparities[pair] for pair in StereoMatcher.pairs[:9]]

    def get_vertical_disparities(self) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
        return StereoMatcher.pairs[9:], [self.disparities[pair] for pair in StereoMatcher.pairs[9:]]

    def get_disparity_to_plot(self, i1: int, i2: int) -> Image.Image:
        return self.disparities[(i1, i2)]

    def save(self) -> None:
        if self.config.outpath is None:
            raise ValueError('Cannot save StereoMatcher without outpath')
        outpath = self.config.outpath
        os.makedirs(outpath, exist_ok=True)
        self.camera_set.save(outpath)
        self.point_cloud.save(outpath)

        if len(self.rectified):
            folder = os.path.join(outpath, StereoMatcher.RECTIFIED_FOLDER)
            for key, val in self.rectified.items():
                val.save(os.path.join(folder, StereoMatcher.get_file_name(key[0], key[1])))

        if self.tasks is not None:
            scipy.io.savemat(os.path.join(outpath, StereoMatcher.RECTIFIED_FOLDER,
                             'stereo_in.mat'), {'task': self.tasks})
            self.logger.log(ActionLogEntry(f'Saved tasks to {outpath}'))
        self.logger.log(ActionLogEntry(f'Saved StereoMatcher to {outpath}'))
