from datetime import datetime


class Clock:
    current_time = datetime.now().isoformat(timespec="seconds")

    @classmethod
    def get_time(cls):
        return cls.current_time

    @classmethod
    def set_new_time(cls):
        cls.current_time = datetime.now().isoformat(timespec="seconds")
