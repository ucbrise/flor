from flor.writer import Writer

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

    def __init__(self, global_key):
        """
        :param global_key: Unique static identifier for code block
        The global key allows us to identify stored state in a memo
        and match it unambiguously at reexecution runtime for loads.
        """
        assert isinstance(global_key, str)
        self.global_key = global_key
        self.block_executed = False

    def should_execute(self, predicate):
        if predicate:
            self.block_executed = True
        return predicate

    def register_side_effects(self, *args):
        self.args = args

    def proc_side_effects(self):
        # TODO: For selective replay, we will want to skip some loads. Add predicate for skipping.
        if self.block_executed:
            # Code ran so we need to store the side-effects
            for arg in self.args:
                Writer.store(arg, self.global_key)
        else:
            # Code did not run, so we need to load the side-effects
            self.args = Writer.load(self.global_key)

        if len(self.args) > 1:
            return self.args
        else:
            return self.args[0]


__all__ = ['SkipBlock']