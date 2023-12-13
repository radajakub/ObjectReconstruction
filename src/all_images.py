import sys
import os

from inout import DataLoader, Plotter
from utils import Config

if __name__ == "__main__":
    config = Config(sys.argv)

    loader = DataLoader(config.scene)
    plotter = Plotter(rows=3, cols=4)

    for id, im in loader.images.items():
        r = ((id - 1) // 4) + 1
        c = ((id - 1) % 4) + 1
        plotter.add_image(im, row=r, col=c)
        plotter.set_title(f'Image {id}', row=r, col=c)

    if config.outpath is None:
        plotter.show()
    else:
        os.makedirs(config.outpath, exist_ok=True)
        plotter.save(outfile=os.path.join(config.outpath, 'all_images.png'))
