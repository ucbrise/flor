#!/usr/bin/env python3

class Reference:

    def __init__(self, path, name=None, stack_scoped=False):
        """
        Formerly a Literal
        :param path: path
        :param name: name
        :param stack_scoped:
            True => Exists in the scope of the Execution
                    Releases lock when Execution ends
            False => Input or Output of Execution
                    Retains lock throughout Experiment
        """
        self.path = path
        self.name = name
        self.stack_scoped = stack_scoped
        self.instruction_no = -1
