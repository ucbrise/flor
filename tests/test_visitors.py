import ast

import pytest

from flordb.hlast.visitors import (
    LoggedExpVisitor,
    ResumeBlockVisitor,
    SaveShapeVisitor,
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


SAVE_PREAMBLE = "import torch\nimport flordb as flor\n"


def save_src(body: str, loop: str = 'for e in flor.loop("epoch", range(3)):') -> str:
    return SAVE_PREAMBLE + loop + "\n" + body


class TestSaveShapeVisitor:
    def test_flat_save_in_a_loop(self):
        ssv = visit(
            save_src('    torch.save(net.state_dict(), "ckpt.pth")\n'),
            SaveShapeVisitor(),
        )
        assert ssv.found
        assert ssv.path == "ckpt.pth"
        assert ssv.applies == [("net", None)]
        assert ssv.lineno == 4

    def test_keyed_save_skips_the_non_state_entries(self):
        # "epoch" and "loss" ride along in every real checkpoint dict and are
        # not restore targets; they are dropped, not treated as a refusal.
        ssv = visit(
            save_src(
                "    torch.save({\n"
                '        "model": net.state_dict(),\n'
                '        "optimizer": opt.state_dict(),\n'
                '        "epoch": e,\n'
                '        "loss": loss.item(),\n'
                '    }, "ckpt.pth")\n'
            ),
            SaveShapeVisitor(),
        )
        assert ssv.found
        assert ssv.applies == [("net", "model"), ("opt", "optimizer")]

    def test_save_under_flor_iteration(self):
        source = (
            SAVE_PREAMBLE
            + 'with flor.iteration("epoch", 3, None):\n'
            '    torch.save(net.state_dict(), "ckpt.pth")\n'
        )
        ssv = visit(source, SaveShapeVisitor())
        assert ssv.found
        assert ssv.applies == [("net", None)]

    def test_save_sharing_a_function_scope_with_the_loop(self):
        # `main()` is the ordinary place to put both; the loop's frame is where
        # restore resolves these names, and it is the same frame.
        source = (
            SAVE_PREAMBLE
            + "def main():\n"
            "    net = build()\n"
            '    for e in flor.loop("epoch", range(3)):\n'
            '        torch.save(net.state_dict(), "ckpt.pth")\n'
        )
        ssv = visit(source, SaveShapeVisitor())
        assert ssv.found
        assert ssv.applies == [("net", None)]

    def test_save_in_a_helper_is_flagged_not_used(self):
        # `m` is a local of save_ckpt; replay resolves names against the loop's
        # frame, where that name does not exist.
        source = (
            SAVE_PREAMBLE
            + "def save_ckpt(m):\n"
            '    torch.save(m.state_dict(), "ckpt.pth")\n'
            "\n"
            'for e in flor.loop("epoch", range(3)):\n'
            "    save_ckpt(net)\n"
        )
        ssv = visit(source, SaveShapeVisitor())
        assert not ssv.found
        assert ssv.unscoped_match

    def test_save_outside_any_loop_is_flagged_not_used(self):
        source = SAVE_PREAMBLE + 'torch.save(net.state_dict(), "ckpt.pth")\n'
        ssv = visit(source, SaveShapeVisitor())
        assert not ssv.found
        assert ssv.unscoped_match

    def test_computed_path_is_not_matched(self):
        ssv = visit(
            save_src('    torch.save(net.state_dict(), f"ckpt-{e}.pth")\n'),
            SaveShapeVisitor(),
        )
        assert not ssv.found
        assert not ssv.unscoped_match

    def test_saving_a_whole_module_is_not_matched(self):
        # torch.save(net, p) pickles the module; loading it back yields a
        # module, not a state dict, so there is no mapping to read off it.
        ssv = visit(
            save_src('    torch.save(net, "ckpt.pth")\n'), SaveShapeVisitor()
        )
        assert not ssv.found

    def test_attribute_receiver_is_not_matched(self):
        # DataParallel: restoring means unwrapping `net` the same way, which
        # the save site does not say and flor will not assume.
        ssv = visit(
            save_src('    torch.save(net.module.state_dict(), "ckpt.pth")\n'),
            SaveShapeVisitor(),
        )
        assert not ssv.found

    def test_two_files_are_refused(self):
        ssv = visit(
            save_src(
                '    torch.save(net.state_dict(), "model.pth")\n'
                '    torch.save(opt.state_dict(), "opt.pth")\n'
            ),
            SaveShapeVisitor(),
        )
        assert not ssv.found
        assert ssv.multi_path

    def test_two_flat_saves_of_one_file_are_refused(self):
        # Only whichever ran last is what the file holds. Two flat *loads* of
        # one file are consistent and ResumeBlockVisitor takes them; two saves
        # of it contradict each other.
        ssv = visit(
            save_src(
                '    torch.save(net.state_dict(), "ckpt.pth")\n'
                '    torch.save(ema.state_dict(), "ckpt.pth")\n'
            ),
            SaveShapeVisitor(),
        )
        assert not ssv.found
        assert ssv.conflicting_shape

    def test_flat_and_keyed_saves_of_one_file_are_refused(self):
        ssv = visit(
            save_src(
                '    torch.save(net.state_dict(), "ckpt.pth")\n'
                '    torch.save({"model": net.state_dict()}, "ckpt.pth")\n'
            ),
            SaveShapeVisitor(),
        )
        assert not ssv.found
        assert ssv.conflicting_shape

    def test_repeated_identical_saves_are_one_mapping(self):
        # The same save in two branches is not a contradiction.
        ssv = visit(
            save_src(
                "    if best:\n"
                '        torch.save({"model": net.state_dict()}, "ckpt.pth")\n'
                "    else:\n"
                '        torch.save({"model": net.state_dict()}, "ckpt.pth")\n'
            ),
            SaveShapeVisitor(),
        )
        assert ssv.found
        assert ssv.applies == [("net", "model")]

    def test_best_model_is_inferred_as_written(self):
        # The case the caller has to announce: this is a clean, successful
        # inference that may still name the wrong object.
        ssv = visit(
            save_src('    torch.save(best_model.state_dict(), "ckpt.pth")\n'),
            SaveShapeVisitor(),
        )
        assert ssv.found
        assert ssv.applies == [("best_model", None)]
