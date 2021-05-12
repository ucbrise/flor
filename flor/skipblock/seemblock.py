from ..journal import Journal
from ..logger import Logger

from abc import ABC, abstractmethod


class SeemBlock(ABC):
    journal = Journal()
    logger = Logger()

    @staticmethod
    @abstractmethod
    def step_into(block_name: str, probed=None):
        ...

    @staticmethod
    @abstractmethod
    def end(*args, values=None):
        ...
