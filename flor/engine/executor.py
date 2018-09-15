#!/usr/bin/env python3

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from flor.experiment_graph import ExperimentGraph
from flor.light_object_model import *
from typing import List

import time
import os

EPOCH = 0.01


class Executor:

    @staticmethod
    def __run__(eg: 'ExperimentGraph', action: ActionLight):
        inputs = eg.b[action]
        outputs = eg.d[action]
        output_ids = {}
        kwargs = {}

        for i in inputs:
            if type(i) == ArtifactLight and not i.parent:
                kwargs[i.name] = i.get_location()
            elif type(i) == ArtifactLight:
                kwargs[i.name] = i.get_isolated_location()
            elif type(i) == LiteralLight:
                kwargs[i.name] = i.v
            else:
                raise TypeError("Unknown type: {}".format(type(i)))

        for o in outputs:
            if type(o) == ArtifactLight:
                kwargs[o.name] = o.get_isolated_location()
            else:
                output_ids[o.name] = id(o)

        response = action.func(**kwargs) or []

        for kee in response:
            eg.update_value(kee, output_ids[kee], response[kee])

    @staticmethod
    def __isolate_location__(id_num, location):
        # Currently only supports localhost, singlenode
        location_array = location.split('.')
        return "{}_{}.{}".format('.'.join(location_array[0:-1]), id_num, location_array[-1])

    @staticmethod
    def __is_finished__(eg: 'ExperimentGraph', action: ActionLight):
        outputs = eg.d[action]

        def helper(x):
            if type(x) == LiteralLight:
                return x.v is not None
            elif type(x) == ArtifactLight:
                return os.path.exists(x.get_isolated_location())

        return all(map(helper, outputs))

    @staticmethod
    def __is_ready__(eg: 'ExperimentGraph', action: ActionLight):
        producing_actions = Executor.__get_producing_actions__(eg, action)
        return all(map(lambda x: not x.pending, producing_actions))

    @staticmethod
    def __get_consuming_actions__(eg: 'ExperimentGraph', action: ActionLight):
        outputs = eg.d[action]
        consuming_actions = set([])

        for o in outputs:
            consuming_actions |= eg.d[o]

        return consuming_actions

    @staticmethod
    def __get_producing_actions__(eg: 'ExperimentGraph', action: ActionLight):
        inputs = eg.b[action]
        producing_actions = set([])

        for i in inputs:
            producing_actions |= eg.b[i]

        return producing_actions

    @staticmethod
    def __no_child_is_pending__(eg: 'ExperimentGraph', action: ActionLight):
        consuming_actions = Executor.__get_consuming_actions__(eg, action)
        return all(map(lambda x: not x.pending, consuming_actions))

    @staticmethod
    def execute(eg: 'ExperimentGraph'):
        # Initialize empty lists
        ready: List[ActionLight] = []
        running: List[ActionLight] = []
        done: List[ActionLight] = []
        explored: List[ActionLight] = []

        for root in eg.actions_at_depth[1]:
            ready.append(root)

        new_done = False

        while True:
            for each in range(len(ready)):
                action = ready.pop(0)
                Executor.__run__(eg, action)
                running.append(action)
            time.sleep(EPOCH)
            # TODO: Make for-loop concurrent
            for each in range(len(running)):
                action = running.pop(0)
                if Executor.__is_finished__(eg, action):
                    action.pending = False
                    done.append(action)
                    new_done = True
                else:
                    running.append(action)
            if eg.is_none_pending():
                break
            if new_done:
                examined: List[ActionLight] = []
                for each in range(len(done)):
                    d = done.pop(0)
                    if Executor.__no_child_is_pending__(eg, d):
                        explored.append(d)
                    else:
                        done.append(d)
                        for child in Executor.__get_consuming_actions__(eg, d):
                            if child.pending:
                                if child not in examined:
                                    if Executor.__is_ready__(eg, child):
                                        ready.append(child)
                                    examined.append(child)
                new_done = False






