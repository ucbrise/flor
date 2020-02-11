from typing import Any

# TORCH RULESET
import torch.nn as nn
import torch.optim as optim


class NamespaceStack:
    """
    TODO: We've temporarily overfit to PyTorch for agility
    """

    stack = [{}, ]

    @staticmethod
    def new():
        stack = NamespaceStack.stack
        stack.append({})

    @staticmethod
    def test_force(o: Any, name: str):
        stack = NamespaceStack.stack
        assert stack
        namespace = stack[-1]
        if NamespaceStack.is_comparable(o):
            namespace[name] = o

    @staticmethod
    def pop():
        stack = NamespaceStack.stack
        assert stack
        stack.pop()

    @staticmethod
    def get_forced():
        stack = NamespaceStack.stack
        assert stack
        forced = []
        for layer, ns in enumerate(stack):
            for k, v in ns.items():
                forced.append((layer, k, v))
        return forced

    @staticmethod
    def set_forced(forced):
        stack = NamespaceStack.stack
        assert stack
        for layer, k, state_dict in forced:
            namespace = stack[int(layer)]
            obj = namespace[k]
            obj.load_state_dict(state_dict)

    @staticmethod
    def is_comparable(o):
        return isinstance(o, (nn.Module, optim.Optimizer))


