import inspect
import astunparse

class Struct:
    def __init__(self, assignee=None, value=None, typ=None,
                 instruction_no=None, keyword_name=None,
                 caller=None, pos=None, from_arg=None):
        self.assignee = assignee
        self.value = value
        self.typ = typ
        self.instruction_no = instruction_no
        self.keyword_name = keyword_name
        self.caller = caller
        self.pos = pos
        self.from_arg = from_arg

    def to_dict(self):
        d = {}
        for attr in [i for i in dir(self) if not inspect.ismethod(getattr(self, i)) and '__' != i[0:2]]:
            d[attr] = getattr(self, attr, None)
        return d

    def __str__(self):
        return str(self.to_dict())

    @classmethod
    def from_dict(cls, d):
        return cls(**d)