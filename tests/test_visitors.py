import ast

import pytest

from flordb.hlast.visitors import (
    LoggedExpVisitor,
    ResumeBlockVisitor,
    SetupLoadVisitor,
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
    def test_flor_loop_is_found(self):
        assert visit(NESTED, WithExpVisitor()).found

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

    def test_function_scoped_pattern_records_its_scope(self):
        source = (
            "import torch\n"
            "\n"
            "def resume(model):\n"
            '    state = torch.load("ckpt.pth")\n'
            '    model.load_state_dict(state["model"])\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert rbv.found
        assert rbv.scope_name == "resume"
        assert rbv.scope_lineno == 3

    def test_no_pattern(self):
        rbv = visit("x = 1\n", ResumeBlockVisitor())
        assert not rbv.found

    def test_flat_one_liner(self):
        # Pairs with torch.save(model.state_dict(), path): no dict, no key, so
        # the whole file is the target's state -- recorded as a key of None.
        source = (
            "import torch\n"
            'model.load_state_dict(torch.load("ckpt.pth"))\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert rbv.found
        assert rbv.path == "ckpt.pth"
        assert rbv.lhs_name is None
        assert rbv.applies == [("model", None)]

    def test_flat_one_liner_inside_an_existence_guard(self):
        # How it is actually written. `if` doesn't nest scope, so it still
        # counts as module scope.
        source = (
            "import os\n"
            "import torch\n"
            'if os.path.exists("ckpt.pth"):\n'
            '    model.load_state_dict(torch.load("ckpt.pth", map_location="cpu"))\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert rbv.found
        assert rbv.applies == [("model", None)]

    def test_flat_one_liner_with_a_computed_path_is_not_matched(self):
        # flor has to name the mirror file before the script runs.
        source = (
            "import torch\n"
            "p = 'ckpt' + '.pth'\n"
            "model.load_state_dict(torch.load(p))\n"
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert not rbv.found

    def test_flat_one_liner_in_a_function_records_its_scope(self):
        source = (
            "import torch\n"
            "\n"
            "def resume(model):\n"
            '    model.load_state_dict(torch.load("ckpt.pth"))\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert rbv.found
        assert rbv.scope_name == "resume"
        assert rbv.scope_lineno == 3
        assert rbv.applies == [("model", None)]

    def test_two_flat_targets_from_one_file(self):
        source = (
            "import torch\n"
            'model.load_state_dict(torch.load("ckpt.pth"))\n'
            'ema.load_state_dict(torch.load("ckpt.pth"))\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert rbv.found
        assert rbv.applies == [("model", None), ("ema", None)]

    def test_two_different_files_are_refused(self):
        # ResumeSpec addresses one file; guessing which half to restore would
        # leave the other object on some other iteration's state.
        source = (
            "import torch\n"
            'model.load_state_dict(torch.load("model.pth"))\n'
            'optimizer.load_state_dict(torch.load("opt.pth"))\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert not rbv.found
        assert rbv.multi_path

    def test_keyed_and_flat_over_different_files_are_refused(self):
        source = (
            "import torch\n"
            '_resume = torch.load("ckpt.pth")\n'
            'model.load_state_dict(_resume["model"])\n'
            'optimizer.load_state_dict(torch.load("opt.pth"))\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert not rbv.found
        assert rbv.multi_path

    def test_load_and_apply_in_different_scopes_do_not_match(self):
        source = (
            'state = torch.load("ckpt.pth")\n'
            'def prep(state, model):\n'
            '    model.load_state_dict(state["model"])\n'
        )
        assert not visit(source, ResumeBlockVisitor()).found

    def test_two_resume_scopes_are_ambiguous_even_with_the_same_path(self):
        source = (
            'def prep(model):\n'
            '    model.load_state_dict(torch.load("ckpt.pth"))\n'
            'def other(model):\n'
            '    model.load_state_dict(torch.load("ckpt.pth"))\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert not rbv.found
        assert rbv.ambiguous_scope

    def test_nested_method_records_the_method_scope(self):
        source = (
            'class Setup:\n'
            '    @staticmethod\n'
            '    def prep(model):\n'
            '        model.load_state_dict(torch.load("ckpt.pth"))\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert rbv.found
        assert rbv.scope_name == "prep"
        assert rbv.scope_lineno == 2

    def test_two_keyed_files_in_one_scope_are_refused(self):
        source = (
            'def prep(model, optimizer):\n'
            '    state = torch.load("model.pth")\n'
            '    model.load_state_dict(state["model"])\n'
            '    state = torch.load("optimizer.pth")\n'
            '    optimizer.load_state_dict(state["optimizer"])\n'
        )
        rbv = visit(source, ResumeBlockVisitor())
        assert not rbv.found
        assert rbv.multi_path


class TestSetupLoadVisitor:
    def test_load_before_the_loop(self):
        source = (
            "import torch\n"
            "state = torch.load(path)\n"
            'for e in flor.loop("epoch", range(3)):\n'
            "    pass\n"
        )
        assert visit(source, SetupLoadVisitor()).found

    def test_load_inside_the_loop_is_not_setup(self):
        source = (
            "import torch\n"
            'for e in flor.loop("epoch", range(3)):\n'
            '    blob = torch.load("data.pt")\n'
        )
        assert not visit(source, SetupLoadVisitor()).found

    def test_load_inside_flor_iteration_is_not_setup(self):
        source = (
            "import torch\n"
            'with flor.iteration("epoch", 0, None):\n'
            '    blob = torch.load("data.pt")\n'
        )
        assert not visit(source, SetupLoadVisitor()).found

    def test_load_in_a_helper_counts(self):
        # Nothing in the source says where the helper is called from.
        source = (
            "import torch\n"
            "def prep(model):\n"
            "    model.load_state_dict(torch.load(path))\n"
        )
        assert visit(source, SetupLoadVisitor()).found

    def test_no_load(self):
        assert not visit("x = 1\n", SetupLoadVisitor()).found
