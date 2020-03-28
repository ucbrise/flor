from flor.skipblock.skip_block import SkipBlock
from flor.skipblock.seem_block import SeemBlock
from flor.constants import *
from .. import stateful as state
import time

class SkipStack:

    stack = []
    id = 0
    worst_time = - float('inf')
    memory = set([])

    @staticmethod
    def auto_incr_id():
        id = SkipStack.id
        SkipStack.id += 1
        return id

    @staticmethod
    def new(static_key, skip_type=True):
        if skip_type:
            stack = SkipStack.stack
            global_key = SkipStack.auto_incr_id() if state.MODE is EXEC else None
            block = SkipBlock(static_key, global_key)
            stack.append(block)
        elif static_key not in SkipStack.memory:
            stack = SkipStack.stack
            global_key = SkipStack.auto_incr_id() if state.MODE is EXEC else None
            block = SeemBlock(static_key, global_key)
            stack.append(block)
        else:
            SkipStack.stack.append(None)

    @staticmethod
    def peek():
        stack = SkipStack.stack
        assert stack
        return stack[-1]

    @staticmethod
    def pop():
        stack = SkipStack.stack
        assert stack
        block = stack.pop()
        if block is not None and block.static_key not in SkipStack.memory:
            SkipStack.memory.add(block.static_key)
            tottime = time.time() - block.start_time
            if tottime > SkipStack.worst_time:
                SkipStack.worst_time = tottime
                state.outermost_sk = block.static_key
        return block
