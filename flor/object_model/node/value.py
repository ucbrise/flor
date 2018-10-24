#!/usr/bin/env python3

class Value:

    def __init__(self, v, name=None, stack_scoped=False):
        """
        Formerly a Literal
        :param v: value
        :param name: name
        :param stack_scoped:
            True => Exists in the scope of the Execution
            False => Input or Output of Execution
        """
        self.v = v
        self.name = name
        self.stack_scoped = stack_scoped
        self.instruction_no = -1

