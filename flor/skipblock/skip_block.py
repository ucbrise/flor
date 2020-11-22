from flor.writer import Writer
from flor.skipblock.namespace_stack import NamespaceStack
from flor.constants import *
from flor.utils import deepcopy_cpu

from .. import stateful as state

from types import ModuleType
from torch import cuda
import copy
import io
import time


CUMULATIVE_RATIO_TOLERANCE= 100
CUTOFF_RATIO = 15
# AVG Write Throughout for P3.8xLarge with Batched Background Materialization
BYTES_PER_SEC = 150627795

class SkipBlock:
    """
    USE

    block = SkipBlock("my_code")
    if block.should_execute(predicates):
        # Execute Block of Code
        ...
        block.register_side_effects(*args)
    *args = block.proc_side_effects()
    """

    #TODO: Refactoring move much of this state to the skipstack

    # FOR RETRAIN
    parallel = False

    # FOR ADAPTIVE CHECKPOINTING
    # contains static keys, so we only estimate once per loop
    skipblock_decisions = {}
    # the current nesting level of skipblocks relative to previously generated skipblocks.
    # This is an Optimization. Don't serialize (deeply nested) state that would be inaccessible during parallelism.
    nesting_level = 0
    # We need to the number of iterations for the outermost loop
    # A proxy is how many times a top nested loop is run
    top_nested_head_sk = -1
    # We want the max ratio for computing periodicity
    acc_ratios = {}
    # Some of this state is moved to STATEFUL to avoid circular dependencies

    # Added for Expert mode pending refactor
    global_key = 0
    stack = []

    def __init__(self, static_key, global_key=None):
        """
        """
        self.static_key = int(static_key)

        if state.MODE is EXEC:
            # Execution stores
            self.global_key = int(global_key)
            Writer.store(LBRACKET, self.static_key, self.global_key)

            if SkipBlock.top_nested_head_sk < 0:
                assert SkipBlock.nesting_level == 0
                SkipBlock.top_nested_head_sk = self.static_key
            if self.static_key == SkipBlock.top_nested_head_sk:
                if state.iterations_count == 1:
                    ratio = SkipBlock.acc_ratios[max(SkipBlock.acc_ratios)]
                    state.period = int(CUMULATIVE_RATIO_TOLERANCE / ratio)
                    state.pretraining = ratio >= CUTOFF_RATIO
                state.iterations_count += 1
        self.block_executed = False
        self.proc_side_effects_called = False
        self.args = []
        self.my_nesting_level = SkipBlock.nesting_level
        SkipBlock.nesting_level += 1

        self.start_time = time.time()

    @classmethod
    def step_into(cls, probed=False, uid=0):
        cond = state.MODE is EXEC or (not state.PSEUDORESUMING and probed)
        skipblock = cls(uid, cls.global_key)
        pred = skipblock.should_execute(cond)
        cls.global_key += 1
        SkipBlock.stack.append(skipblock)
        return pred

    start = step_into

    @classmethod
    def end(cls, *args):
        skipblock = cls.stack.pop()
        return skipblock.proc_side_effects(*args)


    @property
    def top_nested_level(self):
        """
        Whether this loop is the top-nested loop in the code
        We have:
        Outermost Loop
        Top nested loop (in outermost loop)
        Deeply nested loops (in top nested loop)
        """
        # TODO: Generalize to cases
        # 1. Case when outermost loop is successfully transformed
        # 2. Case when there is more than a single outermost loop
        return self.my_nesting_level == 0

    @property
    def period_enabled(self):
        return state.period > 0 and (state.iterations_count % state.period == 0)

    def should_execute(self, predicate):
        self.block_executed = predicate
        if state.MODE is REEXEC:
            self.global_key = int(Writer.lbrack_load())
            self.block_executed = self.block_executed or self.global_key in state.rbracket_gk
        return self.block_executed

    def register_side_effects(self, *args):
        self.args = args

    def proc_side_effects(self, *args):
        # TODO: For selective replay, we will want to skip some loads. Add predicate for skipping.
        # TODO: Bug, the cpu() call won't copy if object is already in CPU
        # WARNING: MAY ONLY BE CALLED ONCE
        self.end_time = time.time()
        SkipBlock.nesting_level -= 1
        assert SkipBlock.nesting_level >= 0
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

        if state.MODE is EXEC:
            # Code ran so we need to store the side-effects
            if self.static_key not in SkipBlock.skipblock_decisions:
                size_in_bytes, tiempo = self._getsizeof_side_effects()
                loop_time = self.end_time - self.start_time
                # write_time = tiempo
                write_time = size_in_bytes / BYTES_PER_SEC
                ratio = loop_time / write_time
                SkipBlock.skipblock_decisions[self.static_key] = ratio >= CUTOFF_RATIO
                SkipBlock.acc_ratios[loop_time] = max(SkipBlock.acc_ratios.get(loop_time, -float('inf')), ratio)
            if SkipBlock.skipblock_decisions[self.static_key] or (not state.pretraining and self.period_enabled and self.top_nested_level):
                self._store_side_effects()
            else:
                # This helps us garbage collect unmatched LBRACKETS
                Writer.store(RBRACKET, self.static_key, self.global_key)
        elif state.MODE is REEXEC and not self.block_executed:
            # Code did not run, so we need to load the side-effects
            self._load_side_effects()
            inpt = filtered_args
            outpt = self.args

            next_outpt = []

            for lhs, rhs in zip(inpt, outpt):
                if type(lhs) != type(rhs):
                    next_outpt.append(rhs)
                elif isinstance(lhs, list):
                    lhs[:] = rhs
                    next_outpt.append(lhs)
                elif isinstance(lhs, dict):
                    lhs.update(rhs)
                    next_outpt.append(lhs)
                elif is_object(lhs):
                    lhs.__dict__.update(rhs.__dict__)
                    next_outpt.append(lhs)
                else:
                    next_outpt.append(rhs)

            self.args = next_outpt

        filtered_args = self.args
        self.args = [arg if isinstance(arg, ModuleType) else filtered_args.pop(0) for arg in args]
        assert not filtered_args, f"Should have depleted filtered_args, but len: {len(filtered_args)}"

        if len(self.args) > 1:
            return self.args
        else:
            # len(self.args) can't be 0...
            # see integrity constraint above
            return self.args[0]

    def _getsizeof_side_effects(self):
        start_time = time.time()
        size_in_bytes = 0
        pickle = Writer.pickler

        forced = NamespaceStack.get_forced()
        forced_objects = [each[2] for each in forced]
        materialize_additionals = False
        # First, write everything new in self.args to disk
        for arg in self.args:
            f = io.BytesIO()
            if NamespaceStack.is_comparable(arg) and arg in forced_objects:
                # arg will be written from forced
                # If optimizer was modified, you'll also want to materialize the network
                materialize_additionals = True
            else:
                # write this arg to disk, it's not in forced
                if hasattr(arg, 'state_dict'):
                    pickle.dump(arg.state_dict(), f)
                    size_in_bytes += f.tell()
                else:
                    # Not state_dict()
                    if hasattr(arg, 'cpu'):
                        pickle.dump(arg.cpu(), f)
                        size_in_bytes += f.tell()
                    else:
                        pickle.dump(arg, f)
                        size_in_bytes += f.tell()
        # Enter a separator
        # If I should materialize a node in a group, materialize the entire group (forced)
        if materialize_additionals:
            for l, k, v in forced:
                f = io.BytesIO()
                pickle.dump(v.state_dict(), f)
                size_in_bytes += f.tell()
        return size_in_bytes, time.time() - start_time


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
            if NamespaceStack.is_comparable(arg) and arg in forced_objects:
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
        if cuda.is_available():
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
