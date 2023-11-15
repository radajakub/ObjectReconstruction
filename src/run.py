import sys

from plotter import Plotter
from loader import DataLoader
import toolbox as tb


class Config:
    def __init__(self, argv: list[str]) -> None:
        args = argv[1:]
        if len(args) % 2 != 0:
            print("wrong number of arguments")
            exit(1)

        self.scene = None
        self.img1 = None
        self.img2 = None
        for i in range(0, len(args), 2):
            key, val = args[i], args[i+1]
            if key == '--scene':
                self.scene = val
            elif key == '--img1':
                self.img1 = int(val)
            elif key == '--img2':
                self.img2 = int(val)
        if self.scene is None:
            raise ValueError('scene not specified')


if __name__ == "__main__":
    config = Config(sys.argv)

    loader = DataLoader(config.scene)
    print(loader)

    if config.img1 is None or config.img2 is None:
        print('one of the images images is not specified, nothing is shown')
        exit(1)

    if config.img1 > loader.image_num or config.img2 > loader.image_num or config.img1 < 1 or config.img2 < 1:
        print(f'invalid image id -> must be between 1 and {loader.image_num} (including)')
        exit(1)

    plotter = Plotter(rows=1, cols=3)
    plotter.add_image(loader.images[config.img1], row=1, col=1)
    plotter.add_image(loader.images[config.img2], row=1, col=2)
    corr1, corr2 = loader.get_corresp(config.img1, config.img2)
    plotter.add_image(loader.images[config.img1], row=1, col=3)
    plotter.add_needles(tb.p2e(corr1), tb.p2e(corr2), row=1, col=3)
    plotter.show()
