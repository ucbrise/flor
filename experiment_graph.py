#!/usr/bin/env python3

import dill

class ExperimentGraph:

    def __init__(self):
        self.d = {}
        # a start is that which has no incoming edge
        self.starts = set([])

    def node(self, v):
        assert v not in self.d
        self.d[v] = set([])
        self.starts |= {v,}

    def edge(self, u, v):
        assert u in self.d
        assert v in self.d
        self.d[u] |= {v,}
        self.starts -= {v,}

    def serialize(self):
        with open('experiment_graph.pkl', 'wb') as f:
            dill.dump(self, f)