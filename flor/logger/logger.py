import os
from pathlib import PurePath
from typing import List, Tuple, Any

from .copy import deepcopy
from flor import shelf


class Logger:
    def __init__(self, path=None, buf_size=None):
        self.path = PurePath(path) if path is not None else None

        self.buffer = Buffer() if buf_size is None else Buffer(buf_size)
        self.promise = None

    def set_path(self, path):
        self.path = PurePath(path)

    def append(self, *args):
        assert self.path is not None, "Logger path not set."
        self.buffer.append(self.to_promises(args))
        if self.buffer.is_full():
            self.flush()

    def flush(self):
        pid = os.fork()
        if not pid:
            self._flush_buffer()
        else:
            self.buffer.clear()

    def _flush_buffer(self):
        if self.promise is not None:
            self.buffer = list(map(self.promise, self.buffer))
        with open(self.path, 'w') as f:
            for e in self.buffer:
                f.write(e)

    def force(self):
        self._flush_buffer()

    def register_promise(self, func):
        self.promise = func

    def to_promises(self, e: List[Any]):
        return list(map(self.promise, e))


class Buffer:
    def __init__(self, size=1024):
        self._b: List[Tuple[int, Any]] = []
        self.size = size

    def append(self, e: List[Any]):
        self._b.append((len(e), e))

    def is_full(self):
        return len(self._b) >= self.size

    def clear(self):
        self._b[:] = []

    def __len__(self):
        return len(self._b)

    
def default_promise(e: Any):
    deepcopy(e)


# def merge():
#     """
#     Stitch together parallel-written files
#     """
#     if shelf.get_latest().exists():
#         shelf.get_latest().unlink()
#     shelf.get_latest().symlink_to(shelf.get_index())


# def close(eof_entry: EOF):
#     feed(eof_entry)
#     write()
#     merge()


# Add feed and write
# there used to be a tree feed entry. Why?