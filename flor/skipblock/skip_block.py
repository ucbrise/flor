from flor.writer import Writer
from flor.skipblock.namespace_stack import NamespaceStack
from flor.constants import *
from flor.utils import deepcopy_cpu

from .. import stateful as state

from types import ModuleType
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

    def __init__(self, static_key, global_key=None):
        """
        """
        self.static_key = int(static_key)
        if state.MODE is EXEC:
            # Execution stores
            self.global_key = int(global_key)
            Writer.store(LBRACKET, self.static_key, self.global_key)
        self.block_executed = False
        self.proc_side_effects_called = False
        self.args = []

    def should_execute(self, predicate):
        self.block_executed = predicate
        if state.MODE is REEXEC:
            # Re-execution that skips loads
            self.global_key = int(Writer.lbrack_load())
        return predicate

    def register_side_effects(self, *args):
        self.args = args

    def proc_side_effects(self, *args):
        # TODO: For selective replay, we will want to skip some loads. Add predicate for skipping.
        # TODO: Bug, the cpu() call won't copy if object is already in CPU
        # WARNING: MAY ONLY BE CALLED ONCE
        assert not self.args or not args                # The caller does not register_side_effects and proc_side_effects, maybe remove with refactor
        assert self.args or args                        # The caller registers_side_effects xor proc_side_effects. We just want to make sure that by this point we have state to proc
        assert not self.proc_side_effects_called        # You don't call proc_side_effects twice on the same SkipBlock.
        self.proc_side_effects_called = True

        if args:
            self.args = args                            # Refer to self.args from now on

        # Filter out ModuleTypes (torch.cuda) so we don't proc-them
        # We use this solution so we don't disrupt the Writer or the Logs
        filtered_args = [arg for arg in self.args if not isinstance(arg, ModuleType)]

        args = self.args                            # Store for later restore
        self.args = filtered_args

        def is_object(a):
            return all([not isinstance(a, list),
            not isinstance(a, dict),
            not isinstance(a, ModuleType),
            not hasattr(a, 'state_dict'),   # This is a pytorch object, handled separately.
            hasattr(a, '__dict__')])

        hooks = [arg for arg in self.args]
        self.args = [arg if not is_object(arg) else arg.__dict__ for arg in self.args]

        if state.MODE is EXEC:
            # Code ran so we need to store the side-effects
            self._store_side_effects()
        elif state.MODE is REEXEC and not self.block_executed:
            # Code did not run, so we need to load the side-effects
            self._load_side_effects()
            for lhs, rhs in zip(hooks, self.args):
                if isinstance(lhs, list):
                    lhs[:] = rhs
                elif isinstance(lhs, dict):
                    lhs.update(rhs)
                elif is_object(lhs):
                    lhs.__dict__.update(rhs)



        filtered_args = self.args
        self.args = [arg if isinstance(arg, ModuleType) else filtered_args.pop(0) for arg in args]
        assert not filtered_args, f"Should have depleted filtered_args, but len: {len(filtered_args)}"

        if len(self.args) > 1:
            return self.args
        else:
            # len(self.args) can't be 0...
            # see integrity constraint above
            return self.args[0]

    def _store_side_effects(self):
        """
        Sub routine for proc_side_effects
        """
        # Code ran so we need to store the side-effects
        forced = NamespaceStack.get_forced()
        forced_objects = [each[2] for each in forced]
        # Looks like everything in forced will be forced to disk
        # There may be some redundancies between forced and self.args, that's what REDUNDANT is for
        # On redundancies, we skip from self.args, not from namespace_stack

        materialize_additionals = False

        # First, write everything new in self.args to disk
        for arg in self.args:
            if arg in forced_objects:
                # arg will be written from forced
                Writer.store(REDUNDANT, self.static_key, self.global_key)
                # If optimizer was modified, you'll also want to materialize the network
                materialize_additionals = True
            else:
                # write this arg to disk, it's not in forced
                if hasattr(arg, 'state_dict'):
                    Writer.store(deepcopy_cpu(arg.state_dict()), self.static_key, self.global_key)
                else:
                    # Not state_dict()
                    if hasattr(arg, 'cpu'):
                        Writer.store(arg.cpu(), self.static_key, self.global_key)
                    else:
                        Writer.store(copy.deepcopy(arg), self.static_key, self.global_key)
        # Enter a separator
        Writer.store(SEPARATOR, self.static_key, self.global_key)
        # If I should materialize a node in a group, materialize the entire group (forced)
        if materialize_additionals:
            for l, k, v in forced:
                Writer.store(str(l), self.static_key, self.global_key)
                Writer.store(k, self.static_key, self.global_key)
                Writer.store(deepcopy_cpu(v.state_dict()), self.static_key, self.global_key)
        cuda.synchronize()

    def _load_side_effects(self):
        """
        Subroutine for proc_side_effects
        """
        # Global key is dynamic
        # Packed state is an array of serialized values
        # The logic for interpreting and organizing them is hard-coded here
        packed_state = Writer.load(self.global_key) # [BLOB, BLOB, ..., SEPARATOR, ..., BLOB]

        # Remember from _store_side_effects that the order is
        # state from self.args SEPARATOR state from forced
        # This is reflected below
        raw_args, raw_forced = [], []

        # Fill raw_args and raw_forced
        current = raw_args
        for each in packed_state:
            if each is SEPARATOR:
                current = raw_forced
                continue
            current.append(each)

        # Structure raw_forced into forced
        forced = []
        current = []
        for i, each in enumerate(raw_forced):
            current.append(each)
            if (i + 1) % 3 == 0:
                forced.append(current)
                current = []

        NamespaceStack.set_forced(forced)
        # We're done restoring forced
        # Next, we restore args for return

        mixed_args = []

        for i, arg in enumerate(raw_args):
            if arg is REDUNDANT:
                # set_forced already takes care of correctly restoring the state of self.args[i]
                mixed_args.append(self.args[i])
            else:
                if hasattr(self.args[i], 'state_dict'):
                    self.args[i].load_state_dict(arg)
                    mixed_args.append(self.args[i])
                else:
                    mixed_args.append(arg)

        self.args = mixed_args


__all__ = ['SkipBlock']
