#!/usr/bin/env python3

from typing import List, Union, Tuple
from flor import util
from flor.object_model import *

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

    def __get_output_names__(self):
        """
        Get Output Artifact Loc / Literal Names as concat string
        :return: see above
        """
        outNames = ''
        for out_artifact in self.out_artifacts:
            if type(out_artifact) == Artifact:
                outNames += out_artifact.loc
            elif type(out_artifact) == Literal:
                outNames += out_artifact.name
            else:
                raise TypeError("{} is invalid, must be Artifact or Literal".format(type(out_artifact)))

    def __getLiteralsAttached__(self, literalsAttachedNames):
        """
        Gets the names of all the literals attached (with leaf at this action) and puts them in literalsAttachedNames
        Not front-facing interface, see Artifact.
        :param literalsAttachedNames: An empty list to be populated with the names of (orphan) literals attached
        :return: None, output is written to literalsAttachedNames
        """
        outNames = self.__get_output_names__()
        if self.funcName + outNames in self.xp_state.visited:
            return
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                if not util.isOrphan(artifact):
                    artifact.parent.__getLiteralsAttached__(literalsAttachedNames)
                elif type(artifact) == Literal:
                    literalsAttachedNames.append(artifact.name)

    def __run__(self, loclist, scriptNames, literalNames={}):
        # TODO: update after Literal change semantics
        outNames = self.__get_output_names__()
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

    def __serialize__(self, lambdas, loclist, scriptNames):
        # TODO: Do we need to serialize for summer?
        outNames = self.__get_output_names__()
        namedLiterals = []
        if self.funcName + outNames in self.xp_state.visited:
            return
        scriptNames.append(self.filenameWithFunc)
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                if type(artifact) == Artifact:
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
        # TODO: Code artifacts should be shared in Diagram.
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