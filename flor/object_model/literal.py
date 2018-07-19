#!/usr/bin/env python3

from flor import util
from flor.shared_object_model.resource import Resource
from flor.engine.expander import Expander
from flor.engine.consolidator import Consolidator

from uuid import uuid4


class Literal(Resource):

    def __init__(self, v,
                 name: str,
                 parent,
                 xp_state,
                 default):
        """
        Initialize a Literal. Literals are versioned in Ground
        :param v: Any memory-resident object
        :param name: The name of the literal, give it one or we'll name it ourselves
        :param parent: The Action that generates this literal, if any
        :param xp_state: Global state of the experiment version
        :param default: param v may be a list, default value for peek purposes
        """

        super().__init__(parent, xp_state)

        self.v = v
        self.__oneByOne__ = False
        self.i = 0
        self.n = 1

        self.default = default

        # The name is used in the visualization and Ground versioning
        if name is None:
            candidate_name = uuid4().hex[0:7]
            while candidate_name in self.xp_state.literalNames:
                candidate_name = uuid4().hex[0:7]
            self.name = candidate_name
        else:
            self.name = name
            assert name not in self.xp_state.literalNames, "Literal name repeated, invalid"

        self.xp_state.literalNames |= {self.name}

        self.xp_state.literalNameToObj[self.name] = self
        self.max_depth = 0

    def __forEach__(self):
        """
        Makes self.v iterable, for multi-trial experiments
        :return:
        """
        if not util.isIterable(self.v):
            raise TypeError("Cannot iterate over literal {}".format(self.v))
        self.__oneByOne__ = True

        # Check if they did not set their own default.
        if self.default is None:
            self.default = self.v[0]
        self.n = len(self.v)
        return self

    def getLocation(self):
        """
        No real location, temporarily added so we don't break anything
        Remove when refactor
        :return:
        """
        return self.name

    def get_shape(self):
        return "underline"

    def getDefault(self):
        if self.default is None:
            return self.v
        return self.default

    def plot(self, rankdir=None):
        super().__plot__(self.name, "underline", rankdir)

    def pull(self, manifest=None):
        experiment_graphs = Expander.expand(self.xp_state.eg, self)
        consolidated_graph = Consolidator.consolidate(experiment_graphs)
        import cloudpickle
        with open('debugging_medium.pkl', 'wb') as f:
            cloudpickle.dump(consolidated_graph, f)
