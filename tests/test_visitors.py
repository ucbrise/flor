import ast

import pytest

from flordb.hlast.visitors import (
    LoggedExpVisitor,
    ResumeBlockVisitor,
    WithExpVisitor,
)

NESTED = '''
import flordb as flor

lr = flor.arg("lr", 1e-3)

for epoch in flor.loop("epoch", range(5)):
    for i, batch in flor.loop("step", enumerate(loader)):
        flor.log("loss", 0.5)
    flor.log("val_acc", 90)

flor.log("accuracy", 99)
'''


def visit(source, visitor):
    visitor.visit(ast.parse(source))
    return visitor


class TestLoggedExpVisitor:
    def test_names_and_linenos(self):
        lev = visit(NESTED, LoggedExpVisitor())
        assert set(lev.names) == {"loss", "val_acc", "accuracy"}
        assert lev.linenos[lev.names["loss"]] == "loss"

    def test_nesting_levels(self):
        lev = visit(NESTED, LoggedExpVisitor())
        # The level is what picks the replay narrowing strategy.
        assert lev.line2level[lev.names["loss"]] == 2
        assert lev.line2level[lev.names["val_acc"]] == 1
        assert lev.line2level[lev.names["accuracy"]] == 0

    def test_loop_names_outermost_first(self):
        assert visit(NESTED, LoggedExpVisitor()).loop_names == ["epoch", "step"]

    def test_keyless_log_is_rejected(self):
        with pytest.raises(IndexError):
            visit("import flordb as flor\nflor.log(0.5)\n", LoggedExpVisitor())


class TestWithExpVisitor:
    def test_flor_loop_is_a_checkpointable_scope(self):
        # v4: a flor.loop alone is enough, no `with flor.checkpointing(...)`.
        assert visit(NESTED, WithExpVisitor()).found

    def test_checkpointing_block(self):
        source = (
            "import flordb as flor\n"
            "with flor.checkpointing(model=m):\n"
            "    pass\n"
        )
        assert visit(source, WithExpVisitor()).found

    def test_plain_for_loop_is_not_found(self):
        source = "for i in range(3):\n    print(i)\n"
        assert not visit(source, WithExpVisitor()).found


RESUME = '''
import torch

model = Net()
optimizer = torch.optim.Adam(model.parameters())

_resume = torch.load("ckpt.pth")
model.load_state_dict(_resume["model"])
optimizer.load_state_dict(_resume["optimizer"])
'''


class TestResumeBlockVisitor:
    def test_module_scope_pattern(self):
        rbv = visit(RESUME, ResumeBlockVisitor())
        assert rbv.found
        assert rbv.path == "ckpt.pth"
        assert rbv.lhs_name == "_resume"
        assert rbv.applies == [("model", "model"), ("optimizer", "optimizer")]

    def test_function_scoped_pattern_is_flagged_not_used(self):
        source = (
            "import torch\n"
            "\n"
            "def resume(model):\n"
            '    state = torch.load("ckpt.pth")\n'
            '    model.load_state_dict(state["model"])\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert not rbv.found
        assert rbv.unscoped_match

    def test_no_pattern(self):
        rbv = visit("x = 1\n", ResumeBlockVisitor())
        assert not rbv.found
        assert not rbv.unscoped_match
