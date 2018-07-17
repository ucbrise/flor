#!/usr/bin/env python3


from typing import Dict

from flor.object_model.action import Action
from flor.object_model.resource import Resource


class Artifact(Resource):

    def __init__(self, loc: str,
                 parent: Action,
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

        self.xp_state.artifactNameToObj[self.loc] = self

    def __getLiteralsAttached__(self):
        """
        Front-facing interface
        :return:
        """
        # Reset visited for getLiteralsAttached graph traversal
        self.xp_state.visited = []
        literalsAttachedNames = []
        if not self.parent:
            return literalsAttachedNames
        self.parent.__getLiteralsAttached__(literalsAttachedNames)
        return literalsAttachedNames

    def getLocation(self):
        return self.loc

    def get_shape(self):
        return "box"

    def plot(self, rankdir=None):
        super().__plot__(self.loc, "box", rankdir)