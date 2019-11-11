from flor.writer import Writer
from flor.skipblock.namespace_stack import NamespaceStack
from flor.constants import REDUNDANT, SEPARATOR

import torch.nn as nn
import torch.optim as optim

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

    def __init__(self, global_key):
        """
        :param global_key: Unique static identifier for code block
        The global key allows us to identify stored state in a memo
        and match it unambiguously at reexecution runtime for loads.
        """
        assert isinstance(global_key, str)
        self.global_key = global_key
        self.block_executed = False
        self.args = []

    def should_execute(self, predicate):
        if predicate:
            self.block_executed = True
        return predicate

    def register_side_effects(self, *args):
        self.args = args

    def proc_side_effects(self, *args):
        # TODO: For selective replay, we will want to skip some loads. Add predicate for skipping.
        assert not self.args or not args
        assert self.args or args

        if args:
            self.args = args

        if self.block_executed:
            # Code ran so we need to store the side-effects
            forced = NamespaceStack.get_forced()
            objects = [each[2] for each in forced]

            materialize_additionals = False

            for arg in self.args:
                if not isinstance(arg, (nn.Module, optim.Optimizer)) or arg not in objects:
                    if not hasattr(arg, 'state_dict'):
                        Writer.store(arg, self.global_key)
                    else:
                        Writer.store(arg.state_dict(), self.global_key)
                else:
                    Writer.store(REDUNDANT, self.global_key)
                    materialize_additionals = True
            Writer.store(SEPARATOR, self.global_key)
            if materialize_additionals:
                for l, k, v in forced:
                    Writer.store(l, self.global_key)
                    Writer.store(k, self.global_key)
                    Writer.store(v.state_dict(), self.global_key)
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

        if len(self.args) > 1:
            return self.args
        else:
            return self.args[0]


__all__ = ['SkipBlock']