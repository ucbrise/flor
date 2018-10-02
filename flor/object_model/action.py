#!/usr/bin/env python3

from typing import List, Union, Tuple
from flor import util

from flor.viz import VizNode, VizGraph


class Action:

    def __init__(self, func,
                 in_artifacts,
                 xp_state):
        """
        Initialize the action
        :param func: flor decoracted function
        :param in_artifacts: see type hints
        :param xp_state: see type hints
        """
        self.filenameWithFunc, self.funcName, self.func = func
        self.out_artifacts = []
        self.in_artifacts = in_artifacts
        self.xp_state = xp_state
        self.max_depth = 0

    def __plotWalk__(self, graph: VizGraph) -> List[VizNode]:
        # TODO: Code artifacts should be shared in Diagram.
        to_list = []

        for child in self.out_artifacts:
            node = graph.newNode(child.name, child.get_shape(), [])
            self.xp_state.nodes[child.getLocation()] = node
            to_list.append(node)

        this = graph.newNode(self.funcName, 'ellipse', [i for i in to_list])
        self.xp_state.nodes[self.funcName] = this

        # scriptNode = graph.newNode(self.filenameWithFunc, 'box', [this,])
        # graph.regis_orphan(scriptNode)

        if self.in_artifacts:
            for artifact in self.in_artifacts:
                if artifact.getLocation() in self.xp_state.nodes:
                    from_node = self.xp_state.nodes[artifact.getLocation()]
                    if artifact.getLocation() in [art.getLocation() for art in self.in_artifacts]: #TODO: CLEAN UP
                        if this not in from_node.next:
                            from_node.next.append(this)
                else:
                    if not util.isOrphan(artifact):
                        from_nodes: List[VizNode] = artifact.parent.__plotWalk__(graph)
                        for from_node in from_nodes:
                            interim = [art.name for art in self.in_artifacts]
                            if from_node.name in interim:
                                if this not in from_node.next:
                                    from_node.next.append(this)
                    else:
                        if type(artifact).__name__ == "Literal":
                            node = graph.newNode(artifact.name, artifact.get_shape(), [this,], util.plating([artifact]))
                        else:
                            node = graph.newNode(artifact.name, artifact.get_shape(), [this,])
                        graph.regis_orphan(node)
                        self.xp_state.nodes[artifact.getLocation()] = node
        return to_list