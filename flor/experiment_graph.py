#!/usr/bin/env python3

import cloudpickle as dill
from flor.shared_object_model.resource import Resource


class ExperimentGraph:

    def __init__(self):
        # forward edges
        self.d = {}
        # backward edges
        self.b = {}
        # a start is a Resource which has no incoming edge
        self.starts = set([])
        # Name_map only contains resources
        self.name_map = {}
        # Given a Flor Object, returns the relevant starts subset
        self.connected_starts = {}

    def node(self, v):
        """
        Experiment facing method
        :param v: a Flor Object
        :return:
        """
        assert v not in self.d
        self.d[v] = set([])
        self.b[v] = set([])
        if issubclass(type(v), Resource):
            self.starts |= {v,}
            self.name_map[v.getLocation()] = v
            if v.parent is not None:
                self.connected_starts[v] = self.connected_starts[v.parent]
            else:
                self.connected_starts[v] = {v, }
        else:
            self.connected_starts[v] = set([])
            for each in v.in_artifacts:
                self.connected_starts[v] |= self.connected_starts[each]

    def light_node(self, v):
        """
        Engine facing method
        :param v: The execution-relevant aspects of a Flor Object
        :return:
        """
        assert v not in self.d
        self.d[v] = set([])
        self.b[v] = set([])
        self.starts |= {v,}


    def edge(self, u, v):
        assert u in self.d
        assert v in self.d
        self.d[u] |= {v, }
        self.b[v] |= {u, }
        self.starts -= {v, }

    def serialize(self):
        with open('experiment_graph.pkl', 'wb') as f:
            dill.dump(self, f)


def deserialize() -> ExperimentGraph:
    with open('experiment_graph.pkl', 'rb') as f:
        out = dill.load(f)
    return out
