from flor.skipblock.skip_block import SkipBlock
from flor.constants import *
import flor.stateful as state


class SkipStack:

    stack = []
    id = 0

    @staticmethod
    def auto_incr_id():
        id = SkipStack.id
        SkipStack.id += 1
        return id

    @staticmethod
    def new(static_key):
        stack = SkipStack.stack
        global_key = SkipStack.auto_incr_id() if state.MODE is EXEC else None
        block = SkipBlock(static_key, global_key)
        stack.append(block)

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
        return block
