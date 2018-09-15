#!/usr/bin/env python3


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from flor.experiment_graph import ExperimentGraph
from typing import List


class Consolidator:
    """
    See consolidate
    """

    @staticmethod
    def consolidate(experiment_graphs: List['ExperimentGraph']) -> 'ExperimentGraph':
        """
        Takes a list of independent experiment graphs and consolidates them into a single
        graph (in place). The consolidation removes redundancies and enables artifact sharing
        respecting the dependencies and identity semantics (see wiki).
        :param experiment_graphs: List of experiment graph, output of source experiment graph expansion
        :return: One consolidated experiment graph with "Light" versions of the object model
        """
        assert len(experiment_graphs) > 0, "Failed: Expansion of Experiment Graphs"
        if len(experiment_graphs) == 1:
            # No "lifting", return the only experiment graph
            return experiment_graphs.pop()

        aligner = Aligner(experiment_graphs[0])

        for eg in experiment_graphs[1:]:
            aligner.put(eg)

        return aligner.eg


class Aligner:
    """
    Helper class for Consolidator
    """

    def __init__(self, seed_eg: 'ExperimentGraph'):
        self.eg = seed_eg

    def put(self, other_eg: 'ExperimentGraph'):
        """
        Merges, by consolidation, other_eg with self.eg
        :param other_eg: Another experiment graph
        """
        # Copy the linked lists from other_eg into eg
        self.eg.absorb(other_eg)

        self.__consolidate_starts__(other_eg.starts)

        depths = [i for i in self.eg.actions_at_depth.keys()]
        depths.sort()
        for depth in depths:
            new_actions = self.__consolidate_actions_at_depth__(other_eg, depth)
            # Add the actions that are not collapsible (because they are distinct)
            self.eg.actions_at_depth[depth] |= new_actions

    def __consolidate_starts__(self, other_eg_starts):
        """
        Consolidates the Resources (ActionLight/ArtifactLight) in the starts set
        When this subroutine finishes, self.eg.starts will have all the relevant (and only the relevant)
            elements in other_eg_starts.
        Redundant starts are eliminated
        Pointers from redundants are deleted (back and forth)
        Pointers from actual relevants are set (back and forth)
        :param other_eg_starts: The starts set of other experiment graph
        """
        # Before the change, take a copy for iteration purposes
        true_start = self.eg.starts.copy()
        self.eg.starts |= other_eg_starts

        for seed_start in true_start:
            for other_start in other_eg_starts:
                if seed_start.equals(other_start):
                    # Remove the redundant start
                    self.eg.starts -= {other_start, }
                    for consuming_action in self.eg.d[other_start]:
                        # Consuming action no longer has backward edge to redundant start
                        self.eg.b[consuming_action] -= {other_start, }
                        # Now connect new to seed
                        # The relevant (non-redundant) start now has an edge to
                        #   The action that would have consumed the redundant start
                        self.eg.d[seed_start] |= {consuming_action, }
                        self.eg.b[consuming_action] |= {seed_start, }
                    del self.eg.d[other_start]

    def __consolidate_actions_at_depth__(self, other_eg: 'ExperimentGraph', depth: int):
        """
        Consolidates the ActionLights at the specified depth
        Consolidation entails removal of duplicates, preservation of distincts, and updating of pointers
        We consolidate depth-wise in ascending order to preserve topological order
            True candidates for consolidation would have had their ancestors consolidated (because they are equal)
            Localizing the check for equality (check parents and children rather than full subgraph of ancestry).
        :param other_eg: The other experiment graph
        :param depth: The (int) depth of actions to consolidate
        :return: a set of actions that were not consolidated because they were distinct and should be added to self.eg
        """
        # New actions are actions that do not consolidate because they have no equivalent action in self.eg
        new_actions = set([])
        for other_action in other_eg.actions_at_depth[depth]:
            no_match = True
            for seed_action in self.eg.actions_at_depth[depth]:
                if (seed_action.equals(other_action)
                        and self.eg.b[seed_action] == self.eg.b[other_action]
                        and self.__has_same_outputs__(seed_action, other_action)):
                    # Have established seed_action and other_action are identical
                    # begin collapse
                    self.__collapse_actions__(seed_action, other_action)
                    no_match = False
                    break
            if no_match:
                assert other_action in self.eg.d
                assert other_action in self.eg.b
                new_actions |= {other_action, }
        return new_actions

    def __has_same_outputs__(self, seed_action, other_action) -> bool:
        """
        Helper Method
        Checks whether the seed action and other action have the same output resources (ArtifactLight/LiteralLight)
        :param seed_action: an action from the source experiment graph
        :param other_action: an action from other experiment graph
        :return: Whether both actions have the same outputs
        """
        # In English:
        #     For all outputs in seed_action
        #        There is some output in other_action that is equal
        expressions = []
        for seed_action_product in self.eg.d[seed_action]:
            exp = any(map(lambda x: x.equals(seed_action_product), self.eg.d[other_action]))
            expressions.append(exp)
        return all(expressions)

    def __collapse_actions__(self, seed_action, other_action):
        """
        Helper Method
        Once equality is established, this method does all the work of:
            * Removing duplicated
            * Updating pointers
        :param seed_action: an action from the source experiment graph
        :param other_action: an action from other experiment graph
        """
        # Every dependency that was pointing to other action, delete the pointer
        for input_resource in self.eg.b[other_action]:
            self.eg.d[input_resource] -= {other_action, }

        # For every action consuming an artifact other_action produces, fix the pointers
        for seed_action_product in self.eg.d[seed_action]:
            for other_action_product in self.eg.d[other_action]:
                if seed_action_product.equals(other_action_product):
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
