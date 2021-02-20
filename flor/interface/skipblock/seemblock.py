from abc import ABC, abstractmethod


class SeemBlock(ABC):
    @staticmethod
    @abstractmethod
    def step_into(block_name: str, probed=False):
        ...

    @staticmethod
    @abstractmethod
    def end(*args, values=None):
        ...
