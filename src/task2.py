import sys
import numpy as np

from data import DataLoader, Logger
from utils import Config

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
