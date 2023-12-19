import sys
import os
import numpy as np

from inout import DataLoader, Logger, Plotter3D, ActionLogEntry
from utils import Config
from estimation import StereoMatcher

if __name__ == "__main__":
    config = Config(sys.argv)

    rng = np.random.default_rng(seed=config.seed)

    logger = Logger(config=config)
    logger.intro()

    logger.log(ActionLogEntry('Loading data'))

    loader = DataLoader(config.scene)
    config.check_valid(loader)

    stereo = StereoMatcher.load(config, loader, logger)

    plotter = Plotter3D(hide_axes=True, aspect_equal=True)
    plotter.add_points(stereo.point_cloud.sparse_get_all())
    plotter.add_cameras(stereo.camera_set.get_cameras())
    plotter.show()
