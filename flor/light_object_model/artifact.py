#!/usr/bin/env python3


class ArtifactLight:

    def __init__(self, loc, name):
        self.loc = loc

        location_array = self.loc.split('.')
        self.isolated_loc = "{}_{}.{}".format('.'.join(location_array[0:-1]), id(self), location_array[-1])

        self.name = name

        self.resourceType = True
        self.parent = False
        self.max_depth = 0

    def set_produced(self):
        self.parent = True

    def get_location(self):
        # TODO: S3 case, API call
        return self.loc

    def get_isolated_location(self):
        return self.isolated_loc

    def equals(self, other):
        return (type(self) == type(other)
                and self.loc == other.loc)
