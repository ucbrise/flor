#!/usr/bin/env python3


from typing import Dict

from flor.shared_object_model.resource import Resource
from flor.engine.executor import  Executor
from flor.engine.expander import Expander
from flor.engine.consolidator import Consolidator

class Artifact(Resource):

    def __init__(self, loc: str,
                 parent, name,
                 manifest: Dict[str, "Artifact"],
                 xp_state):
        """
        Initialize an Artifact
        :param loc: the path of the artifact
        :param parent: Action that generates this parent
        :param manifest: May be getting deprecated soon
        :param xp_state: The global state of the experiment version
        """
        super().__init__(parent, xp_state)
        self.loc = loc
        self.manifest = manifest
        self.name = name

        self.xp_state.artifactNameToObj[self.loc] = self
        self.max_depth = 0

    def getLocation(self):
        return self.loc

    def get_name(self):
        return self.name

    def get_shape(self):
        return "box"

    def plot(self, rankdir=None):
        super().__plot__(self.loc, "box", rankdir)

    def pull(self, manifest=None):
        self.xp_state.eg.serialize()
        experiment_graphs = Expander.expand(self.xp_state.eg, self)
        consolidated_graph = Consolidator.consolidate(experiment_graphs)
        Executor.execute(consolidated_graph)
        self.xp_state.eg.clean()
