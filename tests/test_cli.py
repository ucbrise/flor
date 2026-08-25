import argparse

import pytest

from flordb import cli


class TestParseIterSpec:
    @pytest.mark.parametrize("verb", ["all", "last", "none"])
    def test_verbs(self, verb):
        assert cli.parse_iter_spec(verb) == cli.IterSpec(verb)

    def test_empty_is_none(self):
        # `--iter step=` (the v3 "skip entirely" spelling) still means skip.
        assert cli.parse_iter_spec("") == cli.IterSpec("none")

    def test_indices_are_sorted_and_deduped(self):
        assert cli.parse_iter_spec("5, 0, 2, 0") == cli.IterSpec(
            "indices", (0, 2, 5)
        )

    def test_whitespace_tolerated(self):
        assert cli.parse_iter_spec("  all ") == cli.IterSpec("all")

    def test_non_integer_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_iter_spec("first,second")


class TestParseArgs:
    def test_iter_arg(self):
        assert cli.parse_iter_arg("epoch=0,2") == ("epoch", cli.IterSpec("indices", (0, 2)))

    def test_iter_arg_needs_equals(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_iter_arg("epoch")

    def test_iter_arg_needs_name(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_iter_arg("=all")

    def test_override_splits_once(self):
        # Values may legitimately contain '='.
        assert cli.parse_override_arg("device=cuda:0=x") == ("device", "cuda:0=x")

    def test_override_needs_equals(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_override_arg("device")

    def test_apply_vars_trimmed(self):
        assert cli.parse_apply_vars(" loss , val_acc ,") == ["loss", "val_acc"]


class TestArgvMentions:
    def test_bare_token(self):
        assert cli._argv_mentions(["train.py", "--replay_flor"], ["--replay_flor"])

    def test_flag_equals_value(self):
        # The form that used to slip through and silently run forward.
        assert cli._argv_mentions(["--apply=loss"], ["--apply"])

    def test_unrelated_argv(self):
        assert not cli._argv_mentions(["train.py", "--lr=0.1"], ["--apply", "replay"])


class TestResolveReplayArgs:
    def make(self, **kw):
        base = dict(VARS=None, where_clause=None, replay_apply=None, replay_where=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_new_surface(self):
        args = self.make(replay_apply=["loss", "val_acc"], replay_where="epoch > 2")
        assert cli.resolve_replay_args(args) == (["loss", "val_acc"], "epoch > 2")

    def test_legacy_positional(self):
        args = self.make(VARS=["loss"], where_clause="epoch > 2")
        assert cli.resolve_replay_args(args) == (["loss"], "epoch > 2")

    def test_positional_vars_with_named_where(self):
        args = self.make(VARS=["loss"], replay_where="epoch > 2")
        assert cli.resolve_replay_args(args) == (["loss"], "epoch > 2")

    def test_mixing_apply_with_positional_is_rejected(self):
        # Without this guard "epoch > 2" lands in the VARS slot and gets
        # replayed as if it were a variable name.
        args = self.make(replay_apply=["loss"], VARS=["epoch > 2"])
        with pytest.raises(RuntimeError, match="--where"):
            cli.resolve_replay_args(args)

    def test_nothing_to_apply(self):
        with pytest.raises(RuntimeError, match="nothing to apply"):
            cli.resolve_replay_args(self.make())


class TestIterSpecFor:
    def test_explicit_spec_wins(self, clean_flags):
        clean_flags.iter_specs = {"epoch": cli.IterSpec("all")}
        assert cli.iter_spec_for("epoch") == cli.IterSpec("all")

    def test_unmentioned_loop_defaults_to_last(self, clean_flags):
        cli._defaulted_loops.discard("step")
        assert cli.iter_spec_for("step") == cli.DEFAULT_ITER_SPEC

    def test_default_tip_prints_once(self, clean_flags, capsys):
        cli._defaulted_loops.discard("batch")
        cli.iter_spec_for("batch")
        cli.iter_spec_for("batch")
        assert capsys.readouterr().out.count("not given; defaulting") == 1


class TestReplayModeGate:
    def test_replay_flor_is_the_mode_switch(self, clean_flags):
        assert not cli.in_replay_mode()
        clean_flags.replay_flor = True
        assert cli.in_replay_mode()
