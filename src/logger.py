from config import Config
from loader import DataLoader


class LogEntry:
    def __init__(self, iteration: int, inliers: int, support: float, visible: int,  Nmax: float) -> None:
        self.iteration = iteration
        self.inliers = inliers
        self.support = support
        self.visible = visible
        self.Nmax = Nmax

    def __str__(self) -> None:
        return f'iteration: {self.iteration} -> inliers: {self.inliers}, support: {self.support}, visible: {"-" if self.visible == -1 else self.visible}, Nmax: {self.Nmax}'


class Logger:
    def __init__(self, config: Config, loader: DataLoader) -> None:
        self.config = config
        self.loader = loader
        self.logs = []
        self.improves = []

    def log_end(self, last_iteration: int) -> None:
        self.last_iteration = last_iteration

    def log(self, entry: LogEntry) -> None:
        self.logs.append(entry)

    def log_improve(self, entry: LogEntry) -> None:
        self.improves.append(entry)

    def intro(self) -> None:
        print('config:')
        print(f'-- scene {self.config.scene} with images {self.config.img1} and {self.config.img2}')
        print(f'-- seed {self.config.seed}')
        print(f'-- save output image to {self.config.outpath}')
        print(f'-- epipolar estimation params')
        print(f'---- threshold {self.config.threshold}')
        print(f'---- p {self.config.p}')
        print(f'---- max_iter {self.config.max_iter}')

    def outro(self) -> None:
        print(f'solved in {self.logs[-1].iteration} iterations with best estimate')
        print(self.improves[-1])

    def summary(self) -> None:
        print('BEST ESTIMATES')
        for log in self.improves:
            print(log)

    def dump(self) -> None:
        print('ESTIMATION')
        for log in self.logs:
            print(log)
