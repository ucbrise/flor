#!/usr/bin/env python3


class ArtifactLight:

    def __init__(self, loc):
        self.loc = loc

        self.resourceType = True
        self.produced = False
        self.max_depth = 0

    def set_produced(self):
        self.produced = True

    def equals(self, other):
        return (type(self) == type(other)
                and self.loc == other.loc)