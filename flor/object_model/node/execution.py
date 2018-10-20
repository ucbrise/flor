#!/usr/bin/env python3

class Execution:

    def __init__(self, func, funcName=None):
        self.func = func
        self.funcName = funcName

    @property
    def v(self):
        return self.func

    @property
    def name(self):
        return self.funcName