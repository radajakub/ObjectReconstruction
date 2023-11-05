import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import toolbox as tb


class Plotter:
    def __init__(self, rows: int = 1, cols: int = 1, hide_axes: bool = True, invert_yaxis: bool = True,
                 aspect_equal: bool = False):
        self.fig, self.axes = plt.subplots(rows, cols, squeeze=False)
        for ax in self.axes.flatten():
            if hide_axes:
                ax.axis('off')
            if invert_yaxis:
                ax.invert_yaxis()
            if aspect_equal:
                ax.set_aspect("equal", adjustable="box")

    def get_ax(self, row: int = 1, col: int = 1):
        return self.axes[row-1, col-1]

    def add_image(self, image: Image, grayscale: bool = False, row: int = 1, col: int = 1):
        if grayscale:
            kwargs = {
                'cmap': plt.get_cmap('gray'), 'vmin': 0, 'vmax': 255
            }
        else:
            kwargs = {
            }
        self.get_ax(row, col).imshow(image, **kwargs)

    def add_points(self, X: np.ndarray, color: str = 'black', marker='o', size: float = 1.0, row: int = 1,
                   col: int = 1,):
        assert X.shape[0] == 2
        self.get_ax(row, col).scatter(X[0, :], X[1, :], s=size, facecolors="none", edgecolors=color, marker=marker)

    def add_needles(
            self, starts: np.ndarray, ends: np.ndarray, color: str = 'black', size: float = 1.0, linewidth: float = 0.6,
            row: int = 1, col: int = 1):
        assert starts.shape[0] == 2
        assert ends.shape[0] == 2
        self.add_points(starts, color=color, size=size, row=row, col=col)
        self.get_ax(row, col).plot([starts[0, :], ends[0, :]], [
            starts[1, :], ends[1, :]], color=color, linewidth=linewidth)

    def _get_boundaries(self, row: int = 1, col: int = 1):
        ax = self.get_ax(row, col)
        corners = np.array([[x, y] for x in ax.get_xlim() for y in ax.get_ylim()]).T
        corners = tb.e2p(corners)
        corners = np.column_stack((corners, corners[:, 0]))
        return np.array([np.cross(corners[:, i], corners[:, i + 1]) for i in range(4)]).T

    def add_line(self, normal: np.ndarray, color: str = 'black', linewidth: float = 0.6,
                 row: int = 1, col: int = 1):
        _ = self._get_boundaries(row, col)
        # TODO

    def show(self):
        plt.show()
        # self.fig.show()
