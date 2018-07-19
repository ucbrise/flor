#!/usr/bin/env python3

from flor.experiment_graph import ExperimentGraph
from typing import List


class Aligner:

    def __init__(self, seed_eg: ExperimentGraph):
        self.eg = seed_eg

    def put(self, other_eg: ExperimentGraph):
        self.eg.absorb(other_eg)
        self.__consolidate_starts__(other_eg.starts)
        depths = [i for i in self.eg.actions_at_depth.keys()]
        depths.sort()
        for depth in depths:
            self.__consolidate_actions_at_depth__(other_eg, depth)

    def __consolidate_starts__(self, other_eg_starts):
        true_start = self.eg.starts.copy()
        self.eg.starts |= other_eg_starts

        for seed_start in true_start:
            for other_start in other_eg_starts:
                if seed_start.equals(other_start):
                    self.eg.starts -= {other_start, }
                    for consuming_action in self.eg.d[other_start]:
                        # First clean up
                        self.eg.b[consuming_action] -= {other_start, }
                        # Now connect new to seed
                        self.eg.d[seed_start] |= {consuming_action, }
                        self.eg.b[consuming_action] |= {seed_start, }
                    del self.eg.d[other_start]

    def __has_same_outputs__(self, seed_action, other_action):
        expressions = []
        for seed_action_product in self.eg.d[seed_action]:
            exp = any(map(lambda x: x.equals(seed_action_product), self.eg.d[other_action]))
            expressions.append(exp)
        return all(expressions)

    def __collapse_actions__(self, seed_action, other_action):
        # For every action consuming an artifact other_action produces, fix the pointers
        for seed_action_product in self.eg.d[seed_action]:
            for other_action_product in self.eg.d[other_action]:
                if seed_action_product.equals(other_action_product):
                    # Every dependency that was pointing at other action no longer points to it
                    input_resources = self.eg.b[other_action]
                    debugme = [i for i in input_resources if type(i).__name__ == "LiteralLight" and i.v == 2]
                    print("debugme: {}".format([i.v if type(i).__name__ == "LiteralLight" else i.loc for i in input_resources]))
                    degubme = debugme
                    for input_resource in self.eg.b[other_action]:
                        self.eg.d[input_resource] -= {other_action, }
                    for down_consuming_other_action in self.eg.d[other_action_product]:
                        self.eg.b[down_consuming_other_action] -= {other_action_product, }
                        self.eg.b[down_consuming_other_action] |= {seed_action_product, }
                        self.eg.d[seed_action_product] |= {down_consuming_other_action, }
                    for up_producing_other_action in self.eg.b[other_action_product]:
                        if up_producing_other_action != other_action:
                            self.eg.b[seed_action_product] |= {up_producing_other_action, }
                            self.eg.d[up_producing_other_action] -= {other_action_product, }
                            self.eg.d[up_producing_other_action] |= {seed_action_product, }
                    # Clean up
                    if other_action_product not in self.eg.pre_absorb_d:
                        del self.eg.d[other_action_product]
                    if other_action_product not in self.eg.pre_absorb_b:
                        del self.eg.b[other_action_product]

        # Clean up
        if other_action not in self.eg.pre_absorb_d:
            del self.eg.d[other_action]
        if other_action not in self.eg.pre_absorb_b:
            del self.eg.b[other_action]

    def __consolidate_actions_at_depth__(self, other: ExperimentGraph, depth):
        for seed_action in self.eg.actions_at_depth[depth]:
            for other_action in other.actions_at_depth[depth]:
                if (seed_action.equals(other_action)
                        and self.eg.b[seed_action] == self.eg.b[other_action]
                        and self.__has_same_outputs__(seed_action, other_action)):
                    # Have established seed_action and other_action are identical

                    # begin collapse
                    self.__collapse_actions__(seed_action, other_action)


class Consolidator:

    @staticmethod
    def consolidate(experiment_graphs: List[ExperimentGraph]) -> ExperimentGraph:
        assert len(experiment_graphs) > 0
        if len(experiment_graphs) == 1:
            return experiment_graphs.pop()

        aligner = Aligner(experiment_graphs[0])

        for eg in experiment_graphs[1:]:
            print("starting iteration")
            aligner.put(eg)

        return aligner.eg
