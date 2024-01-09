import sys
import os
import numpy as np

from inout import DataLoader, Logger, Plotter, Plotter3D, ActionLogEntry
from utils import Config
from estimation import StereoMatcher

if __name__ == "__main__":
    config = Config(sys.argv)

    rng = np.random.default_rng(seed=config.seed)

    logger = Logger(config=config)
    logger.intro()

    logger.log(ActionLogEntry('Loading data'))

    loader = DataLoader(config.scene)

    stereo = StereoMatcher.load(config, loader, logger)
    stereo.start_disparities()

    stereo.save()

    logger.log(ActionLogEntry('Save rectified images and prepared tasks for MATLAB'))
    logger.log(ActionLogEntry('FINISHED'))
