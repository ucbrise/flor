from .copy import deepcopy
import os
from flor import shelf

class Logger:
    def __init__(self, path=None, buf_size=1024):
        self.path = path
        self.buf_size = buf_size
        self.buffer = []
        self.preprocess_f = None

    def set_path(self, path):
        self.path = path

    def append(self, *args):
        assert self.path is not None, "Logger path not set."
        for e in args:
            self.buffer.append(e)
        if len(self.buffer) >= self.buf_size:
            self.flush()

    def flush(self):
        pid = os.fork()
        if not pid:
            self._flush_buffer()
        else:
            self.buffer = []

    def _flush_buffer(self):
        if self.preprocess_f is not None:
            self.buffer = list(map(self.preprocess_f, self.buffer))
        with open(self.path, 'w') as f:
            for e in self.buffer:
                f.write(e)

    def force(self):
        self._flush_buffer()

    def register_pre(self, func):
        self.preprocess_f = func