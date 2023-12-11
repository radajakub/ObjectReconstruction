import numpy as np
import scipy.optimize as opt

from packages.corresp import Corresp

from .point_cloud import PointCloud
from .ransac import RANSAC
from .epipolar import EpipolarEstimator
from utils import Config, toolbox as tb
from inout import Logger, DataLoader, CameraGluerLogEntry
from models import GlobalPose, Camera


class GlobalPoseEstimate:
    def __init__(self, R: np.ndarray, t: np.ndarray, inlier_indices: np.ndarray) -> None:
        self.R = R
        self.t = t
        self.inlier_indices = inlier_indices

    def __str__(self) -> str:
        res = 'GlobalPoseEstimate\n'
        res += 'R:\n'
        res += self.R.__str__() + '\n'
        res += 't:\n'
        res += self.t.__str__() + '\n'
        return res


class CameraGluer(RANSAC):
    def __init__(self, loader: DataLoader, point_cloud: PointCloud, config: Config, rng: np.random.Generator = None, logger: Logger = None) -> None:
        super().__init__(model=GlobalPose(loader.K), config=config, rng=rng)
        self.logger = logger
        self.loader = loader
        self.point_cloud = point_cloud
        self.epipolar_estimator = EpipolarEstimator(self.loader.K, config, rng, logger)
        self.pose_threshold = config.pose_threshold
        self.reprojection_threshold = config.reprojection_threshold
        self.p = config.p
        self.max_iterations = config.max_iter
        self.refine_cameras = config.refine_cameras
        self.reset_cameras()
        self.initialize_manipulator()

    def get_cameras(self) -> list[Camera]:
        return [camera for camera in self.cameras.values() if camera is not None]

    def get_camera_count(self) -> int:
        return sum([1 for camera in self.cameras.values() if camera is not None])

    def reset_cameras(self) -> None:
        self.cameras = {key: None for key in self.loader.image_ids}
        self.count = 0

    def initialize_manipulator(self) -> None:
        # initialize cameras
        self.manipulator = Corresp(self.loader.image_num)
        for i in self.loader.image_ids:
            for j in self.loader.image_ids:
                if i < j:
                    ci, cj = self.loader.get_corres(i, j)
                    mij = np.hstack((ci[:, np.newaxis], cj[:, np.newaxis]))
                    self.manipulator.add_pair(i - 1, j - 1, mij)

    def initial(self, img1: int, img2: int) -> None:
        # (1) The set of selected cameras is empty.
        self.reset_cameras()

        # (2) Choose a pair of images I1 a I2 -> done from Config

        # (3) Find the relative pose and orientation (R,t), choose (nonzero) scale, e.g., choose length of base equal to 1.
        m1, m2 = self.manipulator.get_m(img1 - 1, img2 - 1)
        corr1 = self.loader.get_points(img1, m1)
        corr2 = self.loader.get_points(img2, m2)
        estimate = self.epipolar_estimator.fit(corr1, corr2)  # in estimate there is R, t (with unit length)

        # (4) Choose the global coordinate system such that it is equal to the coordinate system of the first camera.
        # Construct the cameras P1 and P2.
        # Put these cameras into the set of selected cameras.
        P1, P2 = estimate.get_cameras()
        self.count += 1
        P1.set_order(self.count)
        self.count += 1
        P2.set_order(self.count)
        self.cameras[img1] = P1
        self.cameras[img2] = P2

        # (5) Reconstruct the 3D point cloud using inlier correspondences between the images I1 and I2 using the cameras P1 and P2 (the points must be in front of both cameras)
        corr_in_1, corr_in_2 = estimate.get_inliers(corr1, corr2)
        scene_indices = self.point_cloud.add_epipolar(estimate, corr_in_1, corr_in_2)

        self.manipulator.start(img1 - 1, img2 - 1, estimate.inlier_indices, scene_indices)

        # (6) Refine the camera set {P1, P2} together with the point cloud using bundle adjustment.
        # TODO

        self.logger.log(CameraGluerLogEntry([img1, img2]))

    def append_camera(self) -> None:
        # (1) Select an image (Ij) that has not a camera estimated yet
        Is, counts = self.manipulator.get_green_cameras()
        # add 1 to get it into my indexing for consistency
        Ij = Is[np.argmax(counts)] + 1

        # (2) Find image points in Ij, that correspond to some allready reconstructed 3D points in the cloud
        point_indices, corr_indices, _ = self.manipulator.get_Xu(Ij - 1)
        scene_points = tb.e2p(self.point_cloud.get_points(point_indices))
        image_points = tb.e2p(self.loader.get_points(Ij, corr_indices))

        # (3) Estimate the global pose and orientation of the camera Pj using the P3P algorithm in RANSAC scheme
        estimate = self.fit(scene_points, image_points)

        # (4) Insert the camera Pj into the set of selected cameras
        Pj = Camera.from_Rt(self.loader.K, estimate.R, estimate.t)
        self.count += 1
        Pj.set_order(self.count)
        self.cameras[Ij] = Pj
        self.manipulator.join_camera(Ij - 1, estimate.inlier_indices)

        # (5) Refine the camera Pj using numeric minimisation of reprojection errors in Ij (updates Pj only)
        if self.refine_cameras:
            scene_inliers = scene_points[:, estimate.inlier_indices]
            image_inliers = image_points[:, estimate.inlier_indices]

            def opt_P(P: np.ndarray):
                P = P.reshape((3, 4))
                image_inliers_reprojected = P @ scene_inliers
                return np.sum(self.model.error(image_inliers, image_inliers_reprojected))
            res = opt.minimize(opt_P, Pj.P.flatten())

            # replace the old camera
            new_P = res['x'].reshape((3, 4))
            Pj = Camera.from_P(Pj.K, new_P)
            self.cameras[Ij] = Pj

        # (6) Find correspondences from between Ij and the images of selected cameras, that have not 3D point yet and reconstruct new 3D points and add them to the point cloud
        for Ii in self.manipulator.get_cneighbours(Ij - 1) + 1:
            mj, mi = self.manipulator.get_m(Ij - 1, Ii - 1)
            Pi = self.cameras[Ii]

            # Reconstruct new scene points using the cameras i and ic and image-to-image correspondences m
            corrj = tb.e2p(self.loader.get_points(Ij, mj))
            corri = tb.e2p(self.loader.get_points(Ii, mi))

            # reproject and threshold reprojection error
            X = tb.Pu2X(Pj.P, Pi.P, corrj, corri)
            ej = self.model.camera_error(X, corrj, Pj)
            ei = self.model.camera_error(X, corri, Pi)
            correct = np.logical_and(ej < self.reprojection_threshold, ei < self.reprojection_threshold)
            corrj = corrj[:, correct]
            corri = corri[:, correct]

            # TODO: Check visibility ??
            visible = Camera.check_visibility(Pj, Pi, corrj, corri)
            corrj = corrj[:, visible]
            corri = corri[:, visible]

            scene_indices = self.point_cloud.add(Pj, Pi, corrj, corri)
            # self.manipulator.new_x(Ij - 1, Ii - 1, np.arange(X.shape[1])[correct], scene_indices)
            self.manipulator.new_x(Ij - 1, Ii - 1, np.arange(X.shape[1])[correct][visible], scene_indices)

        for Ii in self.manipulator.get_selected_cameras() + 1:
            Pi = self.cameras[Ii]
            Xid, uid, Xu_verified = self.manipulator.get_Xu(Ii - 1)
            X = tb.e2p(self.point_cloud.get_points(Xid))
            u = tb.e2p(self.loader.get_points(Ii, uid))

            # Verify (by reprojection error) scene-to-image correspondences in Xu_tentative. A subset of good points is obtained
            e = self.model.camera_error(X, u, Pi)
            indices = np.arange(X.shape[1])
            inl = np.logical_and(~Xu_verified, e < self.reprojection_threshold)
            curr_ok = indices[inl]

            self.manipulator.verify_x(Ii - 1, curr_ok)

        self.manipulator.finalize_camera()

    def fit(self, scene_points: np.ndarray, image_points: np.ndarray) -> GlobalPoseEstimate:
        assert (scene_points.shape[1] == image_points.shape[1])

        # (1) Check, if there are enough correspondences. If not, terminate with no solution found.
        N = scene_points.shape[1]
        if N < self.model.min_samples:
            raise Exception("P3P: Not enough correspondences")

        # (2) Initialise number of iterations and its allowed maximum
        self.it = 0
        Nmax = np.inf

        # (3) Initialise best hypothesis and support
        best_support = 0
        best_estimate = None

        # (4) Loop while number of iterations is less than maximum allowed
        while self.it <= Nmax and self.it < self.max_iterations:
            self.it += 1
            # (5) Generate hypothesis
            # (5a) Sample 3 correspondences
            sample_idx = self.rng.choice(N, self.model.min_samples, replace=False)
            sample_scene_points = scene_points[:, sample_idx]
            sample_image_points = image_points[:, sample_idx]

            Rts = self.model.hypothesis(sample_scene_points, sample_image_points)

            for Rt in Rts:
                R, t = Rt
                # (6) Verify hypothesis
                # (6a) Compute camera
                P = Camera.from_Rt(self.model.K, R, t)

                # (6b) Select 3D points that are in front of the camera
                visible_indices = P.visible_indices(scene_points)
                visible_scene_points = scene_points[:, visible_indices]
                visible_image_points = image_points[:, visible_indices]
                if visible_indices.shape[0] == 0:
                    continue

                # (6c) Compute reprojection error
                err = self.model.camera_error(visible_scene_points, visible_image_points, P)
                inliers = err < np.power(self.pose_threshold, 2)
                inlier_indices = visible_indices[inliers]
                support = self.model.support(err[inliers], self.pose_threshold)
                if support > best_support:
                    best_support = support
                    best_estimate = GlobalPoseEstimate(R, t, inlier_indices)
                    eps = 1 - inlier_indices.shape[0] / N
                    Nmax = 0 if eps == 0 else (np.log(1 - self.p) / np.log(1 - np.power(1 - eps, 3)))

        return best_estimate
