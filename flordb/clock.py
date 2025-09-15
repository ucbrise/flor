import time
from datetime import datetime
from . import cli


class Clock:
    start_time = time.perf_counter()
    current_datetime = datetime.now().isoformat(timespec="seconds")

    def __init__(self) -> None:
        self.s_time = None

    def set_start_time(self):
        self.s_time = time.perf_counter()

    def get_delta(self):
        s_time = self.start_time if self.s_time is None else self.s_time
        return time.perf_counter() - s_time

    @classmethod
    def get_datetime(cls):
        if cli.in_replay_mode():
            assert cli.flags.old_tstamp is not None
            return cli.flags.old_tstamp
        else:
            assert cls.current_datetime is not None
            return cls.current_datetime

    @classmethod
    def set_new_datetime(cls):
        cls.current_datetime = datetime.now().isoformat(timespec="seconds")
        cls.start_time = time.perf_counter()
