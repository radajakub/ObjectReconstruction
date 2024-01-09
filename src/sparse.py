import sys
import os
import numpy as np

from inout import DataLoader, Logger, Plotter3D, ActionLogEntry
from utils import Config
from estimation import CameraGluer, StereoMatcher

if __name__ == "__main__":
    config = Config(sys.argv)

    rng = np.random.default_rng(seed=config.seed)

    logger = Logger(config=config)
    logger.intro()

    logger.log(ActionLogEntry('Loading data'))

    loader = DataLoader(config.scene)
    config.check_images_given(loader)
    config.check_images_correct(loader)

    camera_gluer = CameraGluer(loader, config=config, rng=rng, logger=logger)

    camera_gluer.initial(config.img1, config.img2)
    while camera_gluer.can_add():
        camera_gluer.append_camera()

    camera_set, point_cloud = camera_gluer.get_result()

    stereo = StereoMatcher(config, loader, camera_set, point_cloud, logger)
    stereo.start_disparities()

    plotter = Plotter3D(hide_axes=True, aspect_equal=True)
    plotter.add_points(point_cloud.sparse_get_all())
    plotter.add_cameras(camera_set.get_cameras())

    logger.log(ActionLogEntry('FINISHED: All cameras glued, sparse point computed and rectified images prepared'))

    if config.outpath is None:
        plotter.show()
    else:
        outpath = config.outpath
        os.makedirs(outpath, exist_ok=True)
        logger.dump(path=outpath)
        stereo.save()
