from .copy import deepcopy
from .future import Future
from .. import shelf

import os
from pathlib import PurePath
from typing import List, Tuple, Any


class Logger:
    def __init__(self, path=None, buf_size=None):
        self.path = PurePath(path) if path is not None else None
        self.buffer = Buffer() if buf_size is None else Buffer(buf_size)
        self.flush_count = 0

    def set_path(self, path):
        self.path = PurePath(path)

    def append(self, o: Future):
        assert self.path is not None, "Logger path not set."
        self.buffer.append(o)
        if self.buffer.is_full():
            self.flush()

    def flush(self):
        self.flush_count += 1
        pid = os.fork()
        if not pid:
            self._flush_buffer()
        else:
            self.buffer.clear()

    def force(self):
        self._flush_buffer()
    
    def _flush_buffer(self):
        assert isinstance(self.path, PurePath)
        p = self.path.with_name(self.path.stem + f'_{self.flush_count}' + self.path.suffix)
        with open(p, 'w') as f:
            for o in self.buffer.flush():
                f.write(o + os.linesep)


class Buffer:
    def __init__(self, size=1024):
        self._b: List[Future] = []
        self.size = size

    def append(self, o: Future):
        assert isinstance(o, Future), "object passed to flor Logger must implement Future interface"
        o.promise()
        self._b.append((len(o), o))

    def is_full(self):
        return len(self._b) >= self.size

    def clear(self):
        self._b[:] = []

    def flush(self):
        for o in self._b:
            yield o.fulfill()
        self.clear()

    def __len__(self):
        return len(self._b)

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
