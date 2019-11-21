from flor.writer import Writer
from flor.skipblock.namespace_stack import NamespaceStack
from flor.constants import *

import flor.stateful as state

import torch.nn as nn
import torch.optim as optim
from torch import cuda
import copy


class SkipBlock:
    """
    TODO: We've temporarily overfit to PyTorch for agility

    USE

    block = SkipBlock("my_code")
    if block.should_execute(predicates):
        # Execute Block of Code
        ...
        block.register_side_effects(*args)
    *args = block.proc_side_effects()

    """

    def __init__(self, global_key=None):
        """
        :param global_key: Unique static identifier for code block
        The global key allows us to identify stored state in a memo
        and match it unambiguously at reexecution runtime for loads.
        """

        if state.MODE is EXEC:
            self.global_key = global_key
            Writer.store(LBRACKET, self.global_key)
        else:
            self.global_key = Writer.lbrack_load()
        self.block_executed = False
        self.proc_side_effects_called = False
        self.args = []

    def should_execute(self, predicate):
        self.block_executed = predicate
        return predicate

    def register_side_effects(self, *args):
        self.args = args

    def proc_side_effects(self, *args):
        # TODO: For selective replay, we will want to skip some loads. Add predicate for skipping.
        # TODO: Bug, the cpu() call won't copy if object is already in CPU
        # WARNING: MAY ONLY BE CALLED ONCE
        assert not self.args or not args
        assert self.args or args
        assert not self.proc_side_effects_called
        self.proc_side_effects_called = True

        if args:
            self.args = args

        if state.MODE is EXEC:
            # Code ran so we need to store the side-effects
            forced = NamespaceStack.get_forced()
            objects = [each[2] for each in forced]

            materialize_additionals = False

            for arg in self.args:
                if not isinstance(arg, (nn.Module, optim.Optimizer)) or arg not in objects:
                    if not hasattr(arg, 'state_dict'):
                        Writer.store(copy.deepcopy(arg), self.global_key)
                    else:
                        Writer.store(copy.deepcopy(arg.state_dict()), self.global_key)
                else:
                    Writer.store(REDUNDANT, self.global_key)
                    materialize_additionals = True
            Writer.store(SEPARATOR, self.global_key)
            if materialize_additionals:
                for l, k, v in forced:
                    Writer.store(l, self.global_key)
                    Writer.store(k, self.global_key)
                    sd = v.state_dict()
                    sd_copy = {}
                    for k in sd:
                        if hasattr(sd[k], 'cpu'):
                            sd_copy[k] = sd[k].cpu()
                        else:
                            sd_copy[k] = copy.deepcopy(sd[k])
                    Writer.store(sd_copy, self.global_key)
        else:
            # Code did not run, so we need to load the side-effects
            packed_state = Writer.load(self.global_key)
            raw_args, raw_forced = [], []
            current = raw_args
            for each in packed_state:
                if each is SEPARATOR:
                    current = raw_forced
                    continue
                current.append(each)

            forced = []
            current = []
            for i, each in enumerate(raw_forced):
                current.append(each)
                if (i + 1) % 3 == 0:
                    forced.append(current)
                    current = []

            NamespaceStack.set_forced(forced)

            mixed_args = []
            for i, arg in enumerate(raw_args):
                if arg is REDUNDANT:
                    mixed_args.append(self.args[i])
                else:
                    if hasattr(self.args[i], 'state_dict'):
                        self.args[i].load_state_dict(arg)
                        mixed_args.append(self.args[i])
                    else:
                        mixed_args.append(arg)

            self.args = mixed_args

        cuda.synchronize()

        if len(self.args) > 1:
            return self.args
        else:
            return self.args[0]


__all__ = ['SkipBlock']