#!/usr/bin/env python3


class ActionLight:

    def __init__(self, funcName, func):
        self.funcName = funcName
        self.func = func

        self.resourceType = False
        self.ready = False
        self.max_depth = 0

    def set_ready(self):
        self.ready = True

    def equals(self, other):
        # TODO: should we compare func?
        return (type(self) == type(other)
                and self.funcName == other.funcName)

