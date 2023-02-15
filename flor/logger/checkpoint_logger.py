import glob
import os
from pathlib import Path
from typing import List, Generator

from .. import shelf
from .future import Future
from .. import flags


class Logger:
    def __init__(self, path=None, buf_size=None):
        self.path = Path(path) if path is not None else None
        self.buffer = Buffer() if buf_size is None else Buffer(buf_size)
        self.flush_count = 0

    def set_path(self, path):
        self.path = Path(path)

    def append(self, o: Future):
        assert self.path is not None, "Logger path not set."
        self.buffer.append(o)
        if self.buffer.is_full():
            self.flush()

    def flush(self, is_final=False):
        assert self.path is not None
        self.flush_count += 1
        # pid = os.fork()
        # if not pid:
        self._flush_buffer()
        if is_final:
            p = self.path.with_name(self.path.stem + "_*" + self.path.suffix)
            with open(self.path, "wb") as out_f:
                for pi in glob.glob(p.as_posix()):
                    with open(pi, "rb") as in_f:
                        out_f.write(in_f.read())
                    os.remove(pi)
        # else:
        self.buffer.clear()

    def force(self):
        self._flush_buffer()

    def _flush_buffer(self):
        assert isinstance(self.path, Path)
        p = self.path.with_name(
            self.path.stem + f"_{self.flush_count}" + self.path.suffix
        )
        with open(p, "w") as f:
            for o in self.buffer.flush():
                assert o is not None
                f.write(o + os.linesep)


class Buffer:
    def __init__(self, size=1024):
        self._b: List[Future] = []
        self.size = size

    def append(self, o: Future):
        assert isinstance(
            o, Future
        ), "object passed to flor Logger must implement Future interface"
        o.promise()
        self._b.append(o)

    def is_full(self):
        return len(self._b) >= self.size

    def clear(self):
        self._b[:] = []

    def flush(self) -> Generator[str, None, None]:
        for o in self._b:
            yield o.fulfill()
        self.clear()

    def __len__(self):
        return len(self._b)
