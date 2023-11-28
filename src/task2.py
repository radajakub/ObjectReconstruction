import sys
import numpy as np

from plotter import Plotter
from loader import DataLoader
from epipolar import EpipolarEstimator
from config import Config
from logger import Logger


if __name__ == "__main__":
    config = Config(sys.argv)

    rng = np.random.default_rng(seed=config.seed)

    loader = DataLoader(config.scene)
    config.check_valid(loader)

    logger = Logger(config=config, loader=loader)
    logger.intro()

    logger.dump()
    logger.summary()
    logger.outro()
