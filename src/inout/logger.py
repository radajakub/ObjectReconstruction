import time

from utils import Config


class LogEntry:
    def __init__(self, label: str) -> None:
        self.time = 0
        self.label = label

    def __str__(self) -> str:
        return self.label

    def add_timestamp(self, start_time: float) -> None:
        self.time = round(time.process_time() - start_time, 4)


class EpipolarEstimateLogEntry(LogEntry):
    def __init__(self, iteration: int, inliers: int, support: float, visible: int,  Nmax: float) -> None:
        super().__init__(label='EpipolarEstimate')
        self.iteration = iteration
        self.inliers = inliers
        self.support = support
        self.visible = visible
        self.Nmax = Nmax

    def __str__(self) -> None:
        return f'({self.label}) [{self.time} | it: {self.iteration}] inliers: {self.inliers}, support: {self.support}, visible: {"-" if self.visible == -1 else self.visible}, Nmax: {self.Nmax}'


class RANSACLogEntry(LogEntry):
    def __init__(self, iteration: int, inliers: int, support: float, Nmax: float) -> None:
        super().__init__(label='RANSAC')
        self.iteration = iteration
        self.inliers = inliers
        self.support = support
        self.Nmax = Nmax

    def __str__(self) -> None:
        return f'({self.label}) [{self.time} | it: {self.iteration}] inliers: {self.inliers}, support: {self.support}, Nmax: {self.Nmax}'


class Logger:
    def __init__(self, config: Config = None) -> None:
        self.start_time = time.process_time()
        self.config = config
        self.logs = []
        self.improves = []

    def log_clean(self) -> None:
        self.start_time = time.process_time()
        self.logs = []
        self.improves = []

    def log_end(self, last_iteration: int) -> None:
        self.last_iteration = last_iteration

    def log(self, entry: LogEntry) -> None:
        entry.add_timestamp(start_time=self.start_time)
        self.logs.append(entry)

    def log_improve(self, entry: LogEntry) -> None:
        entry.add_timestamp(start_time=self.start_time)
        self.improves.append(entry)

    def intro(self) -> None:
        if self.config is not None:
            print('config:')
            print(f'-- scene {self.config.scene} with images {self.config.img1} and {self.config.img2}')
            print(f'-- seed {self.config.seed}')
            print(f'-- save output image to {self.config.outpath}')
            print(f'-- epipolar estimation params')
            print(f'---- threshold {self.config.threshold}')
            print(f'---- p {self.config.p}')
            print(f'---- max_iter {self.config.max_iter}')
            print()

    def outro(self) -> None:
        if len(self.logs):
            print(f'solved in {self.logs[-1].iteration} iterations with best estimate')
            print(self.improves[-1])
            print()

    def summary(self) -> None:
        print('BEST ESTIMATES')
        for log in self.improves:
            print(log)
        print()

    def dump(self) -> None:
        print('ESTIMATION')
        for log in self.logs:
            print(log)
        print()
