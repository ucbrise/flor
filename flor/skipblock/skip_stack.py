from flor.skipblock.skip_block import SkipBlock


class SkipStack:

    stack = []
    id = 0

    @staticmethod
    def auto_incr_id():
        id = SkipStack.id
        SkipStack.id += 1
        return str(id)

    @staticmethod
    def new():
        stack = SkipStack.stack
        block = SkipBlock(SkipStack.auto_incr_id())
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
