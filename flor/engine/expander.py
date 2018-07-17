#!/usr/bin/env python3

import itertools

from flor import util
from flor.experiment_graph import ExperimentGraph
from flor.light_object_model import *


class Expander:

    @staticmethod
    def __const_literal_col_store__(literals, literal_names):
        """
        Subroutine of cross_prod_literals

        :param literals: cross product of input literals [(lit1_1, lit2_1), (lit1_1, lit2_2), (lit1_2, lit2_1), (lit1_2, lit2_2)]
        :param literal_names: a list of names for the literals ["lit1name", "lit2name"]
        :return: {
            "lit1name" : [lit1_1, lit1_1, lit1_2, lit1_2],
            "lit2name" : [lit2_1, lit2_2, lit2_1, lit2_2]
        }
        """
        col_store = {}
        for idx, val in enumerate(literals[0]):
            col_store[literal_names[idx]] = [val, ]
        for each in literals[1:]:
            for idx, val in enumerate(each):
                col_store[literal_names[idx]].append(val)
        return col_store

    @staticmethod
    def __cross_prod_literals__(starts_subset):
        """
        Takes the cross product of all the root literals (those with statically defined values)
        The method handles cases where the literal value is iterated over or not.
        In addition to calculating the cross product,
            The name of the literal is preserved, so that we may assign the right values (configuration)
            to each literal.
        :param starts_subset: The starts set that is connected to pull artifact
        :return: see const_literal_col_store, (int) number of trials
        """
        literal_names = []
        literals = []
        for each in starts_subset:
            if type(each).__name__ == "Literal":
                if each.__oneByOne__:
                    literals.append(each.v)
                else:
                    if type(each.v) == tuple:
                        literals.append((each.v, ))
                    else:
                        literals.append([each.v, ])
                literal_names.append(each.name)

        literals = list(itertools.product(*literals))
        num_trials = len(literals)

        return Expander.__const_literal_col_store__(literals, literal_names), num_trials

    @staticmethod
    def __bfs__(starts, src_eg: ExperimentGraph,
                dest_eg: ExperimentGraph, col_store,
                trial_index: int):

        # Make the nodes
        explored = []
        queue = [i for i in starts]
        src_dest_map = {}

        while queue:
            node = queue.pop(0)
            if type(node).__name__ == "Action":
                light = ActionLight(node.funcName, node.func)
                dest_eg.light_node(light)
            elif type(node).__name__ == "Artifact":
                light = ArtifactLight(node.loc)
                dest_eg.light_node(light)
            elif type(node).__name__ == "Literal":
                if node.name in col_store:
                    light = LiteralLight(col_store[node.name][trial_index], node.name)
                    dest_eg.light_node(light)
                else:
                    light = LiteralLight(None, node.name)
                    dest_eg.light_node(light)
            else:
                raise TypeError("Invalid node type: {}".format(type(node)))

            src_dest_map[node] = light
            explored.append(node)

            for child in src_eg.d[node]:
                if child not in (explored + queue):
                    queue.append(child)

        # Make the edges
        explored = []
        queue = [i for i in starts]

        while queue:
            node = queue.pop(0)

            for each in src_eg.d[node]:
                dest_eg.edge(src_dest_map[node], src_dest_map[each])

            explored.append(node)

            for child in src_eg.d[node]:
                if child not in (explored + queue):
                    queue.append(child)


    @staticmethod
    def expand(eg: ExperimentGraph, pulled_resource):
        starts_subset = eg.connected_starts[pulled_resource]
        col_store, num_trials = Expander.__cross_prod_literals__(starts_subset)

        experiment_graphs = []

        for i in range(num_trials):
            new_eg = ExperimentGraph()
            Expander.__bfs__(starts_subset, eg, new_eg, col_store, i)
            experiment_graphs.append(new_eg)

        return experiment_graphs





