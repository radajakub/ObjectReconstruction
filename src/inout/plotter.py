import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import utils.toolbox as tb
from utils import is_in_range_approx


class Plotter:
    def __init__(self, rows: int = 1, cols: int = 1, hide_axes: bool = True, invert_yaxis: bool = True, aspect_equal: bool = False):
        self.fig, self.axes = plt.subplots(rows, cols, squeeze=False)
        self.cmap = plt.cm.get_cmap('tab10')
        for ax in self.axes.flatten():
            if hide_axes:
                ax.axis('off')
            if invert_yaxis:
                ax.invert_yaxis()
            if aspect_equal:
                ax.set_aspect("equal", adjustable="box")

    def set_title(self, title: str = '', row: int = 1, col: int = 1):
        self.get_ax(row, col).set_title(title)

    def get_ax(self, row: int = 1, col: int = 1):
        return self.axes[row-1, col-1]

    def add_image(self, image: Image, grayscale: bool = False, row: int = 1, col: int = 1):
        if grayscale:
            kwargs = {'cmap': plt.get_cmap('gray'), 'vmin': 0, 'vmax': 255}
        else:
            kwargs = {}
        self.get_ax(row, col).imshow(image, **kwargs)

    def get_color(self, num: int):
        return self.cmap(num)

    def add_points(self, X: np.ndarray, color: str = 'black', marker='o', size: float = 1.0, row: int = 1, col: int = 1):
        assert len(X.shape) == 2
        assert X.shape[0] == 2
        self.get_ax(row, col).scatter(X[0, :], X[1, :], s=size, facecolors=color, edgecolors=color, marker=marker)

    def add_point(self, x: np.ndarray, color: str = 'black', marker='o', size: float = 1.0, row: int = 1, col: int = 1):
        assert x.shape == (2,)
        self.get_ax(row, col).scatter(x[0], x[1], s=size, facecolors=color, edgecolors=color, marker=marker)

    def add_needles(
            self, starts: np.ndarray, ends: np.ndarray, every_nth: int = 1, color: str = 'black', size: float = 1.0, linewidth: float = 0.6,
            row: int = 1, col: int = 1):
        assert starts.shape[0] == 2
        assert ends.shape[0] == 2

        indices = np.array([x for x in range(0, starts.shape[1], every_nth)])
        starts = starts[:, indices]
        ends = ends[:, indices]

        self.add_points(starts, color=color, size=size, row=row, col=col)
        self.get_ax(row, col).plot([starts[0, :], ends[0, :]], [
            starts[1, :], ends[1, :]], color=color, linewidth=linewidth)

    def _get_line_intersections(self, line: np.ndarray, row: int = 1, col: int = 1):
        ax = self.get_ax(row, col)
        # compute corners
        xmin, xmax = np.sort(ax.get_xlim())
        ymin, ymax = np.sort(ax.get_ylim())
        corners = tb.e2p(np.array([[xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]]))
        # compute boundaries
        boundaries = np.array([np.cross(corners[:, i], corners[:, (i + 1) % 4]) for i in range(4)]).T
        # compute intersections of the line with the boundaries
        intersections = np.array([tb.p2e(np.cross(boundary, line)) for boundary in boundaries.T]).T
        # filter out intersections that are outside of the plot -> two should remain
        intersections = np.array([
            intersection for intersection in intersections.T
            if is_in_range_approx(intersection[0], xmin, xmax) and is_in_range_approx(intersection[1], ymin, ymax)
        ]).T
        return intersections

    def add_line(self, normal: np.ndarray, color: str = 'black', linewidth: float = 0.8,
                 row: int = 1, col: int = 1):
        ax = self.get_ax(row, col)
        intersections = self._get_line_intersections(normal.squeeze(), row, col)
        ax.plot(intersections[0, :], intersections[1, :], color=color, linewidth=linewidth)

    def show(self):
        plt.show()

    def save(self, outfile: str):
        self.fig.savefig(outfile, bbox_inches='tight')


class Plotter3D:
    def __init__(self, hide_axes: bool = True, invert_yaxis: bool = True, aspect_equal: bool = False):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.cmap = plt.cm.get_cmap('tab10')
        if hide_axes:
            self.ax.axis('off')
        if invert_yaxis:
            self.ax.invert_yaxis()
        if aspect_equal:
            self.ax.set_aspect("equal", adjustable="box")

    def add_points(self, X: np.ndarray, color: str = 'black', marker='o', size: float = 1.0):
        assert X.shape[0] == 3
        self.ax.scatter(X[0, :], X[1, :], X[2, :], s=size, facecolors=color, edgecolors=color, marker=marker)

    def show(self):
        plt.show()

    def save(self, outfile: str):
        self.fig.savefig(outfile, bbox_inches='tight')
