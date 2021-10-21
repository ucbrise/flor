from abc import ABC, abstractmethod


class Future(ABC):
    def __init__(self, v):
        self.value = v
        self.promised = None

    @abstractmethod
    def promise(self):
        """
        Synchronous move or copy for data protection
            Example:
            self.promised = deepcopy(self.value)
        """
        ...

    @abstractmethod
    def fulfill(self) -> str:
        """
        Finishes intended work on stored version of self.value
        """
        assert self.promised is not None
        return ""
