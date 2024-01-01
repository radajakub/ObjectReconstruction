import sys
import os

import numpy as np

from inout import Plotter, Logger, ActionLogEntry
from estimation import RANSAC
from models import Linear
from utils import Config

if __name__ == "__main__":
    config = Config(argv=sys.argv)
    X = np.loadtxt(os.path.join(config.scene, 'ransac.txt')).T
    orig_line = np.array([-10, 3, 1200])[:, np.newaxis]

    plotter = Plotter(rows=1, cols=2, hide_axes=True, invert_yaxis=False, aspect_equal=True)
    plotter.set_title('Only points', row=1, col=1)
    plotter.set_title('Points with fitted line', row=1, col=2)
    plotter.add_points(X, col=1)
    plotter.add_line(orig_line, col=1, color='red')
    plotter.add_points(X, col=2)
    plotter.add_line(orig_line, col=2, color='red')

    logger = Logger(config=config)
    logger.intro()
    logger.log(ActionLogEntry('Estimating line with RANSAC'))

    ransac = RANSAC(model=Linear(), config=config, rng=np.random.default_rng(0), logger=logger)
    estimate = ransac.fit(X)

    plotter.add_line(estimate.M, col=2, color='blue')

    if config.outpath is not None:
        print(config.outpath)
        os.makedirs(config.outpath, exist_ok=True)
        plotter.save(outfile=os.path.join(config.outpath, 'ransac.png'))
        logger.dump(path=config.outpath)
    else:
        plotter.show()
