#!/usr/bin/env python3

import cloudpickle as dill
import os

class ExperimentGraph:

    def __init__(self):
        # forward edges
        self.d = {}
        # backward edges
        self.b = {}
        # a start is a Resource which has no incoming edge
        self.starts = set([])
        # Loc_map only contains resources
        self.loc_map = {}
        # Maps string to the object itself
        self.name_map = {}
        # Given a Flor Object, returns the relevant starts subset
        self.connected_starts = {}
        self.actions_at_depth = {}

        self.pre_absorb_d = None
        self.pre_absorb_b = None

    def __graph_traverse__(self):

        response = {'Action': set([]),
                    'Artifact': set([]),
                    'Literal': set([])}

        explored = []
        queue = [i for i in self.starts]
        visited = lambda: explored + queue

        while queue:
            node = queue.pop(0)

            for type_prefix in response.keys():
                if type(node).__name__[0:len(type_prefix)] == type_prefix:
                    response[type_prefix] |= {node, }
                    break

            explored.append(node)

            for child in self.d[node]:
                if child not in visited():
                    queue.append(child)

        return response

    @property
    def sinks(self):
        d = self.__graph_traverse__()
        out = set([])
        for v in d['Artifact'] | d['Literal']:
            if not self.d[v]:
                out |= {v,}

        return out

    def copy(self):
        new_eg = ExperimentGraph()
        new_eg.starts.update(self.starts)
        new_eg.absorb(self)
        new_eg.loc_map.update(self.loc_map)
        new_eg.connected_starts.update(self.connected_starts)
        new_eg.actions_at_depth.update(self.actions_at_depth)

        return new_eg


    def summary_back_traverse(self, sink):

        response = {'Action': set([]),
                    'Artifact': set([]),
                    'Literal': set([])}

        explored = []
        queue = [sink, ]
        visited = lambda: explored + queue

        while queue:
            node = queue.pop(0)

            for type_prefix in response.keys():
                if type(node).__name__[0:len(type_prefix)] == type_prefix:
                    response[type_prefix] |= {node, }
                    break

            explored.append(node)

            for child in self.b[node]:
                if child not in visited():
                    queue.append(child)

        return response

    def node(self, v):
        """
        Experiment facing method
        :param v: a Flor Object
        :return:
        """
        assert v not in self.d
        self.d[v] = set([])
        self.b[v] = set([])
        if type(v).__name__ == "Artifact" or type(v).__name__ == "Literal":
            self.starts |= {v,}
            self.loc_map[v.getLocation()] = v
            self.name_map[v.name] = v
            if v.parent is not None:
                self.connected_starts[v] = v.xp_state.eg.connected_starts[v.parent]
            else:
                self.connected_starts[v] = {v, }
        else:
            if v.max_depth in self.actions_at_depth:
                self.actions_at_depth[v.max_depth] |= {v, }
            else:
                self.actions_at_depth[v.max_depth] = {v, }
            self.connected_starts[v] = set([])
            for each in v.in_artifacts:
                self.connected_starts[v] |= each.xp_state.eg.connected_starts[each]

    def light_node(self, v):
        """
        Engine facing method
        :param v: The execution-relevant aspects of a Flor Object
        :return:
        """
        assert v not in self.d
        self.d[v] = set([])
        self.b[v] = set([])
        self.starts |= {v, }
        if type(v).__name__ == "Action" or type(v).__name__ == "ActionLight":
            if v.max_depth in self.actions_at_depth:
                self.actions_at_depth[v.max_depth] |= {v, }
            else:
                self.actions_at_depth[v.max_depth] = {v, }
        else:
            self.name_map["{}{}".format(v.name, id(v))] = v

    def edge(self, u, v):
        # Composition code
        if type(u).__name__[-len('Light'):] != "Light":
            self.absorb(u.xp_state.eg)
        if type(v).__name__[-len('Light'):] != "Light":
            self.absorb(v.xp_state.eg)

        assert u in self.d
        assert v in self.d

        self.d[u] |= {v, }
        self.b[v] |= {u, }
        self.starts -= {v, }
        self.actions_at_depth[v.max_depth] -= {v, }
        v.max_depth = max(v.max_depth, u.max_depth + 1)
        if type(v).__name__ == "Action" or type(v).__name__ == "ActionLight":
            if v.max_depth in self.actions_at_depth:
                self.actions_at_depth[v.max_depth] |= {v, }
            else:
                self.actions_at_depth[v.max_depth] = {v, }

    def serialize(self):
        # Vikram's Patch
        foos = {}

        for depth in self.actions_at_depth:
            for action in self.actions_at_depth[depth]:
                if type(action).__name__ == "Action" or type(action).__name__ == "ActionLight":
                    foos[action] = action.func
                    action.func = None
                else:
                    raise ValueError("Invalid action type: {}".format(type(action)))

        with open('experiment_graph.pkl', 'wb') as f:
            dill.dump(self, f)

        for depth in self.actions_at_depth:
            for action in self.actions_at_depth[depth]:
                if type(action).__name__ == "Action" or type(action).__name__ == "ActionLight":
                    action.func = foos[action]
                else:
                    raise ValueError("Invalid action type: {}".format(type(action)))

    def clean(self):
        # Safe to remove only when experiment graph has been copied to flor.d
        os.remove('experiment_graph.pkl')


    def absorb(self, other_eg):
        self.pre_absorb_d = self.d.copy()
        self.pre_absorb_b = self.b.copy()

        self.name_map.update(other_eg.name_map)

        self.d.update(other_eg.d)
        self.b.update(other_eg.b)

    def is_none_pending(self):
        action_set = set([])
        for depth in self.actions_at_depth:
            action_set |= self.actions_at_depth[depth]
        return all(map(lambda x: not x.pending, action_set))

    def update_value(self, name, id_num, value):
        obj = self.name_map["{}{}".format(name, id_num)]
        if "Artifact" in type(obj).__name__:
            obj.loc = value
        elif "Literal" in type(obj).__name__:
            obj.v = value
        else:
            raise TypeError("Uknown type: {}".format(type(obj)))

    def get_artifacts_reachable_from_starts(self):
        return self.__graph_traverse__()['Artifact']

    def get_literals_reachable_from_starts(self):
        return self.__graph_traverse__()['Literal']

    @staticmethod
    def deserialize():
        return deserialize()



def deserialize() -> ExperimentGraph:
    with open('experiment_graph.pkl', 'rb') as f:
        out = dill.load(f)
    return out
