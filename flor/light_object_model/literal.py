#!/usr/bin/env python3


class LiteralLight:

    def __init__(self, v, name):
        self.v = v
        self.name = name

        self.resourceType = True
        self.produced = False
        self.max_depth = 0

    def set_produced(self):
        self.produced = True

    def equals(self, other):
        """
        Not using __eq__ because we want the type hashable
        :param other:
        :return:
        """
        return (
            type(self) == type(other) and
            self.v == other.v and
            self.name == other.name
        )
