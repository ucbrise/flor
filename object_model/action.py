#!/usr/bin/env python3

from typing import List
from .. import util
from .literal import Literal
from .artifact import Artifact

from ..viz import VizNode, VizGraph

class Action:

    def __init__(self, func, in_artifacts, xp_state):
        self.filenameWithFunc, self.funcName, self.func = func
        self.out_artifacts : List[Artifact] = []
        self.in_artifacts = in_artifacts
        self.xp_state = xp_state

    def __run__(self, loclist, scriptNames, literalNames={}):
        outNames = ''
        for out_artifact in self.out_artifacts:
            outNames += out_artifact.loc
        if self.funcName + outNames in self.xp_state.visited:
            return
        scriptNames.append(self.filenameWithFunc)
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                loclist.append(artifact.loc)
                if type(artifact) == Literal:
                    literalNames[artifact.name] = artifact.loc
                if not util.isOrphan(artifact):
                    artifact.parent.__run__(loclist, scriptNames, literalNames)
        self.func(self.in_artifacts, self.out_artifacts)
        self.xp_state.visited.append(self.funcName + outNames)

    def __getLiteralsAttached__(self, literalsAttachedNames):
        outNames = ''
        for out_artifact in self.out_artifacts:
            outNames += out_artifact.loc
        if self.funcName + outNames in self.xp_state.visited:
            return
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                if not util.isOrphan(artifact):
                    artifact.parent.__getLiteralsAttached__(literalsAttachedNames)
                elif type(artifact) == Literal:
                    literalsAttachedNames.append(artifact.name)

    def __serialize__(self, lambdas, loclist, scriptNames):
        outNames = ''
        namedLiterals = []
        for out_artifact in self.out_artifacts:
            outNames += out_artifact.loc
        if self.funcName + outNames in self.xp_state.visited:
            return
        scriptNames.append(self.filenameWithFunc)
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                loclist.append(artifact.loc)
                if not util.isOrphan(artifact):
                    artifact.parent.__serialize__(lambdas, loclist, scriptNames)
                elif type(artifact) == Literal:
                    namedLiterals.append(artifact.name)

        def _lambda(literals=[]):
            i = 0
            args = []
            for in_art in self.in_artifacts:
                if type(in_art) == Literal:
                    args.append(literals[i])
                    i += 1
                else:
                    args.append(in_art)
            self.func(args, self.out_artifacts)

        lambdas.append((_lambda, namedLiterals))
        self.xp_state.visited.append(self.funcName + outNames)

    def __plotWalk__(self, graph: VizGraph) -> List[VizNode]:

        to_list = []

        for child in self.out_artifacts:
            node = graph.newNode(child.loc, 'box', [])
            self.xp_state.nodes[child.loc] = node
            to_list.append(node)

        this = graph.newNode(self.funcName, 'ellipse', [i for i in to_list])
        self.xp_state.nodes[self.funcName] = this

        scriptNode = graph.newNode(self.filenameWithFunc, 'box', [this,])
        graph.regis_orphan(scriptNode)

        if self.in_artifacts:
            for artifact in self.in_artifacts:
                if artifact.getLocation() in self.xp_state.nodes:
                    from_node = self.xp_state.nodes[artifact.getLocation()]
                    if from_node.name in [art.getLocation() if type(art) == Artifact else art.name
                                                  for art in self.in_artifacts]:
                        if this not in from_node.next:
                            from_node.next.append(this)
                else:
                    if not util.isOrphan(artifact):
                        from_nodes : List[VizNode] = artifact.parent.__plotWalk__(graph)
                        for from_node in from_nodes:
                            interim = [art.getLocation() if type(art) == Artifact else art.name
                                                  for art in self.in_artifacts]
                            if from_node.name in interim:
                                if this not in from_node.next:
                                    from_node.next.append(this)
                    else:
                        if type(artifact) == Literal and artifact.name:
                            node = graph.newNode(artifact.name, 'underline', [this,], util.plating([artifact]))
                            graph.regis_orphan(node)
                            self.xp_state.nodes[artifact.loc] = node
                        elif type(artifact) == Literal :
                            node = graph.newNode(artifact.loc, 'underline', [this,], util.plating([artifact]))
                            graph.regis_orphan(node)
                            self.xp_state.nodes[artifact.loc] = node
                        else:
                            node = graph.newNode(artifact.loc, 'box', [this,])
                            graph.regis_orphan(node)
                            self.xp_state.nodes[artifact.loc] = node
        return to_list


    def __plotWalkOld__(self, diagram):
        dot = diagram["dot"]

        # Create nodes for the children
        to_list = []


        clust_i = str(diagram['counter'])
        diagram['counter'] += 1
        with dot.subgraph(name='cluster' + clust_i) as cluster:
            # Prepare the children nodes
            for child in self.out_artifacts:
                node_diagram_id = str(diagram["counter"])
                cluster.node(node_diagram_id, child.loc, shape="box")
                self.xp_state.nodes[child.loc] = node_diagram_id
                to_list.append((node_diagram_id, child.loc))
                diagram["counter"] += 1

            # Prepare this node
            node_diagram_id = str(diagram["counter"])
            cluster.node(node_diagram_id, self.funcName, shape="ellipse")
            self.xp_state.nodes[self.funcName] = node_diagram_id
            diagram["counter"] += 1

            print(self.funcName, ' plating ', util.plating(self.in_artifacts))

            # Add the script artifact
            node_diagram_id_script = str(diagram["counter"])
            dot.node(node_diagram_id_script, self.filenameWithFunc, shape="box")
            diagram["counter"] += 1
            dot.edge(node_diagram_id_script, node_diagram_id)
            self.xp_state.edges.append((node_diagram_id_script, node_diagram_id))

            for to_node, loc in to_list:
                cluster.edge(node_diagram_id, to_node)
                self.xp_state.edges.append((node_diagram_id, to_node))

        if self.in_artifacts:
            for artifact in self.in_artifacts:
                if artifact.getLocation() in self.xp_state.nodes:
                    if (self.xp_state.nodes[artifact.getLocation()], node_diagram_id) not in self.xp_state.edges:
                        dot.edge(self.xp_state.nodes[artifact.getLocation()], node_diagram_id)
                        self.xp_state.edges.append((self.xp_state.nodes[artifact.getLocation()], node_diagram_id))
                else:
                    # Never seen this artifact before
                    if not util.isOrphan(artifact):
                        from_nodes = artifact.parent.__plotWalk__(diagram)
                        for from_node, loc in from_nodes:
                            if loc in [art.getLocation() for art in self.in_artifacts]:
                                if (from_node, node_diagram_id) not in self.xp_state.edges:
                                    dot.edge(from_node, node_diagram_id)
                                    self.xp_state.edges.append((from_node, node_diagram_id))
                    else:
                        node_diagram_id2 = str(diagram["counter"])
                        if type(artifact) == Literal and artifact.name:
                            dot.node(node_diagram_id2, artifact.name,
                                     shape="underline")
                        elif type(artifact) == Literal :
                            dot.node(node_diagram_id2, artifact.loc,
                                     shape="underline")
                        else:
                            dot.node(node_diagram_id2, artifact.loc,
                                     shape="box")
                        self.xp_state.nodes[artifact.loc] = node_diagram_id2
                        diagram["counter"] += 1
                        if (node_diagram_id2, node_diagram_id) not in self.xp_state.edges:
                            dot.edge(node_diagram_id2, node_diagram_id)
                            self.xp_state.edges.append((node_diagram_id2, node_diagram_id))

        return to_list
