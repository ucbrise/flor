import ast
import types

import pandas as pd
import pytest

from flordb import repl
from flordb.cli import IterSpec
from flordb.hlast.visitors import LoggedExpVisitor

SCRIPT = '''
import flordb as flor

for epoch in flor.loop("epoch", range(5)):
    for i in flor.loop("step", loader):
        flor.log("loss", 0.5)
    flor.log("val_acc", 90)
'''


@pytest.fixture
def lev():
    v = LoggedExpVisitor()
    v.visit(ast.parse(SCRIPT))
    return v


class TestSpecToCli:
    @pytest.mark.parametrize("verb", ["all", "last", "none"])
    def test_verbs_round_trip(self, verb):
        assert repl._spec_to_cli(IterSpec(verb)) == verb

    def test_indices(self):
        assert repl._spec_to_cli(IterSpec("indices", (0, 2))) == "0,2"


class TestApplyVarResolution:
    def test_name_passes_through(self, lev):
        assert repl._apply_var_to_name("loss", lev) == "loss"

    def test_lineno_resolves_to_name(self, lev):
        assert repl._apply_var_to_name(f"@{lev.names['val_acc']}", lev) == "val_acc"

    def test_lineno_form_detected(self):
        assert repl._apply_var_is_lineno("@42")
        assert not repl._apply_var_is_lineno("loss")

    def test_unknown_name_lists_the_known_ones(self, lev):
        with pytest.raises(RuntimeError, match="val_acc"):
            repl._apply_var_to_name("grad_norm", lev)

    def test_lineno_that_is_not_a_log_call(self, lev):
        with pytest.raises(RuntimeError, match="not a"):
            repl._apply_var_to_name("@1", lev)


class TestNarrowArgs:
    def schedule(self, num_outer=5, ts="2026-05-22T10:00:00.000000"):
        return types.SimpleNamespace(
            df=pd.DataFrame({"tstamp": [ts], "num_outer": [num_outer]})
        )

    def test_user_narrowing_wins(self, lev):
        user = [("epoch", IterSpec("indices", (1,)))]
        assert (
            repl._narrow_args(1, "2026-05-22T10:00:00.000000", self.schedule(), lev, user)
            == user
        )

    @pytest.mark.parametrize("loglvl", [0, 3])
    def test_no_narrowing_outside_the_loop_levels(self, lev, loglvl):
        assert repl._narrow_args(loglvl, "t", self.schedule(), lev, None) == []

    def test_outer_loop_level_enumerates_every_outer_iter(self, lev):
        ts = "2026-05-22T10:00:00.000000"
        args = repl._narrow_args(1, ts, self.schedule(num_outer=3, ts=ts), lev, None)
        assert args == [("epoch", IterSpec("indices", (0, 1, 2)))]

    def test_nested_level_adds_the_inner_loop(self, lev):
        ts = "2026-05-22T10:00:00.000000"
        args = repl._narrow_args(2, ts, self.schedule(num_outer=2, ts=ts), lev, None)
        assert [name for name, _ in args] == ["epoch"]
