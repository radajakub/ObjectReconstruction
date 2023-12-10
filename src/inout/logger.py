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


class CameraGluerLogEntry(LogEntry):
    def __init__(self, camera_ids: list[int]) -> None:
        super().__init__(label='CameraGluer')
        self.camera_ids = camera_ids

    def __str__(self) -> None:
        if len(self.camera_ids) == 2:
            return f'({self.label}) [{self.time}] initialized with cameras {self.camera_ids[0]} and {self.camera_ids[1]}'
        return f'({self.label}) [{self.time}] glued camera {self.camera_ids[0]}'


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
        if not self.config.silent:
            print(entry)

    def intro(self) -> None:
        if self.config is not None and not self.config.silent:
            print(self.config)

    def dump(self) -> None:
        print('ESTIMATION')
        for log in self.logs:
            print(log)
        print()
