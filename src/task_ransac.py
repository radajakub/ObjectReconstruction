import numpy as np

from inout import Plotter, Logger
from estimation import RANSAC
from models import Linear

if __name__ == "__main__":
    print('Testing RANSAC on line fitting...')
    X = np.loadtxt('data/ransac/ransac.txt').T
    orig_line = np.array([-10, 3, 1200])[:, np.newaxis]

    plotter = Plotter(rows=1, cols=2, hide_axes=True, invert_yaxis=False, aspect_equal=True)
    plotter.set_title('Only points', row=1, col=1)
    plotter.set_title('Points with fitted line', row=1, col=2)
    plotter.add_points(X, col=1)
    plotter.add_line(orig_line, col=1, color='red')
    plotter.add_points(X, col=2)
    plotter.add_line(orig_line, col=2, color='red')

    logger = Logger()
    logger.intro()

    ransac = RANSAC(model=Linear(), max_iterations=1000, threshold=3,
                    p=0.999, rng=np.random.default_rng(0), logger=logger)
    estimate = ransac.fit(X)

    plotter.add_line(estimate.M, col=2, color='blue')

    logger.dump()
    logger.summary()
    logger.outro()

    plotter.show()
