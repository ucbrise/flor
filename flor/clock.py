import time
from datetime import datetime

start_time = time.perf_counter()


class Clock:
    current_time = datetime.now().isoformat(timespec="seconds")

    def __init__(self) -> None:
        self.s_time = None

    def set_start_time(self):
        self.s_time = time.perf_counter()

    def get_delta(self):
        s_time = start_time if self.s_time is None else self.s_time
        return time.perf_counter() - s_time

    @classmethod
    def get_time(cls):
        return cls.current_time

    @classmethod
    def set_new_time(cls):
        cls.current_time = datetime.now().isoformat(timespec="seconds")
