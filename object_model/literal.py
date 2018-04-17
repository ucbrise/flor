#!/usr/bin/env python3

from .. import util

class Literal:

    def __init__(self, v, name, xp_state):
        """

        :param v: Literal value
        :param name: must be globally unique per experiment
        """
        self.v = v
        self.xp_state = xp_state
        self.loc = 'ghost_literal_' + str(self.xp_state.literalFilenamesAndIncr()) + '.pkl'
        self.__oneByOne__ = False
        self.i = 0
        self.n = 1

        # The name is used in the visualization and Ground versioning
        if name is None:
            temp = self.loc.split('.')[0]
            i = int(temp.split('_')[2])
            self.name = 'ghost' + str(i)
            while self.name in self.xp_state.literalNames:
                i += 1
                self.name = 'ghost' + str(i)
        else:
            self.name = name
            assert name not in self.xp_state.literalNames
        self.xp_state.literalNames |= {self.name}

        self.xp_state.literalNameToObj[self.name] = self
        self.xp_state.ghostFiles |= {self.loc, }

    def forEach(self):
        if not util.isIterable(self.v):
            raise TypeError("Cannot iterate over literal {}".format(self.v))
        self.__oneByOne__ = True
        self.n = len(self.v)
        return self

    def getLocation(self):
        return self.loc

    def __pop__(self):
        if self.i >= self.n:
            return False
        if self.__oneByOne__:
            util.pickleTo(self.v[self.i], self.loc)
        else:
            util.pickleTo(self.v, self.loc)
        self.i += 1
        return True

    def __enable__(self):
        self.xp_state.ghostFiles |= {self.loc}
        self.xp_state.literals.append(self)
        self.__reset__()

    def __reset__(self):
        self.i = 0
        self.__pop__()
