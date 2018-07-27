#!/usr/bin/env python3


class ArtifactLight:

    def __init__(self, loc, name):
        self.loc = loc
        self.name = name

        self.resourceType = True
        self.produced = False
        self.max_depth = 0

    def set_produced(self):
        self.produced = True

    def get_location(self):
        # TODO: S3 case, API call
        return id(self), self.loc

    def equals(self, other):
        return (type(self) == type(other)
                and self.loc == other.loc)
