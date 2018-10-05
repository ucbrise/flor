#!/usr/bin/env python3

import itertools

from flor.experiment_graph import ExperimentGraph
from flor.light_object_model import *

import os

class Expander:
    """
    See expand
    """

    @staticmethod
    def expand(eg: 'ExperimentGraph', pulled_resource):
        """
        Expands an experiment graph into a set of independent experiment graphs: one per trial.
        The Flor Objects are converted into specialized "Light" Flor Objects more suitable for execution
            and further processing.
        :param eg: The experiment graph constructed and populated by the Flor Plan
        :param pulled_resource: The Artifact or Literal object that was pulled
        :return: LIST[ExperimentGraph], where the nodes of ExperimentGraph are in light_object_model rather
            than object_model
        """
        starts_subset = eg.connected_starts[pulled_resource]
        col_store, num_trials = Expander.__cross_prod_literals__(starts_subset)

        experiment_graphs = []

        for i in range(num_trials):
            new_eg = ExperimentGraph()
            Expander.__bfs__(starts_subset, eg, new_eg, col_store, i)
            experiment_graphs.append(new_eg)

        return experiment_graphs

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
    def __bfs__(starts, src_eg: 'ExperimentGraph',
                dest_eg: 'ExperimentGraph', col_store,
                trial_index: int):
        """
        By traversing every node of the source experiment graph twice, with graph-traversal,
        populates the new experiment graph with the corresponding Light Flor Objects
        And draws edges between the nodes, respecting the original structure of the source exp. graph
        :param starts: The starts set of the source experiment graph
        :param src_eg: The source experiment graph (containing Flor Objects)
        :param dest_eg: The destination experiment graph, corresponds to one trial of the experiment,
            contains Light Flor Objects.
        :param col_store: A dictionary mapping literal name to array of values. The length of the array
            corresponds to the number of trials, col_store is indexed to determine what value to initialize
            the light flor literal with.
        :param trial_index: The index of the trial to which this call of __bfs__ corresponds
        :return: None. But outcome is the populated dest_eg
        """
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
                if node.parent:
                    light = ArtifactLight(node.loc, node.name)
                    light.set_produced()
                else:
                    light = ArtifactLight(os.path.relpath(node.resolve_location()), node.name, node.version is not None)
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
