"""In-process tests for the replay-time selection rules in flordb.api.

These drive api.loop / api.iteration and the planning helpers directly with
cli.flags set the way the CLI would set it, so they cover the selection
contract without a checkpoint store or torch.
"""

import pytest

from flordb import api, cli


@pytest.fixture
def replaying(clean_flags):
    """Put the process in replay mode with a clean api buffer."""
    clean_flags.replay_flor = True
    clean_flags.wev_found = True
    clean_flags.old_tstamp = "2026-05-22T10:00:00.000000"
    api.output_buffer.clear()
    api.layers.clear()
    api.context.clear()
    api._suppress_logs = False
    api._setup_emitted = False
    api._last_main_exit_time = None
    api.skip_cleanup = False  # keep _deferred_init out of the way
    try:
        yield clean_flags
    finally:
        api.output_buffer.clear()
        api.layers.clear()
        api.context.clear()
        api._suppress_logs = False
        api.skip_cleanup = True


def logged(names_only=True):
    records = [r for r in api.output_buffer if not r.name.startswith("time::")]
    return [r.name for r in records] if names_only else records


class TestSelection:
    def test_all(self, replaying):
        replaying.iter_specs = {"step": cli.IterSpec("all")}
        assert api._requested("step", 3) == [0, 1, 2]

    def test_none(self, replaying):
        replaying.iter_specs = {"step": cli.IterSpec("none")}
        assert api._requested("step", 3) == []

    def test_last(self, replaying):
        replaying.iter_specs = {"step": cli.IterSpec("last")}
        assert api._requested("step", 3) == [2]

    def test_indices(self, replaying):
        replaying.iter_specs = {"step": cli.IterSpec("indices", (0, 2))}
        assert api._requested("step", 3) == [0, 2]

    def test_out_of_range_names_the_valid_range(self, replaying):
        replaying.iter_specs = {"step": cli.IterSpec("indices", (7,))}
        with pytest.raises(RuntimeError, match=r"only has 3 iteration.*0\.\.2"):
            api._requested("step", 3)

    def test_unmentioned_loop_defaults_to_last(self, replaying):
        cli._defaulted_loops.discard("step")
        assert api._requested("step", 3) == [2]

    @pytest.mark.parametrize("kind", ["all", "last"])
    def test_empty_loop_requests_nothing(self, replaying, kind):
        replaying.iter_specs = {"step": cli.IterSpec(kind)}
        assert api._requested("step", 0) == []


class TestOuterReplayPlan:
    def test_starts_at_zero_and_stops_after_the_last_request(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("indices", (1, 3))}
        plan, silent = api._build_outer_replay_plan("epoch", list(range(5)))
        assert plan == [(0, 0), (1, 1), (2, 2), (3, 3)]
        assert silent == {0, 2}

    def test_all(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("all")}
        plan, silent = api._build_outer_replay_plan("epoch", list(range(3)))
        assert plan == [(0, 0), (1, 1), (2, 2)]
        assert silent == set()

    def test_last_runs_everything_and_logs_the_end(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("last")}
        plan, silent = api._build_outer_replay_plan("epoch", list(range(3)))
        assert plan == [(0, 0), (1, 1), (2, 2)]
        assert silent == {0, 1}

    def test_none_runs_nothing(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("none")}
        assert api._build_outer_replay_plan("epoch", list(range(3))) == ([], set())

    def test_out_of_range_names_the_valid_range(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("indices", (9,))}
        with pytest.raises(RuntimeError, match=r"valid range: 0\.\.2"):
            api._build_outer_replay_plan("epoch", list(range(3)))

    def test_no_flor_loop_in_the_source_runs_everything(self, replaying):
        replaying.wev_found = False
        replaying.iter_specs = {"epoch": cli.IterSpec("none")}
        plan, silent = api._build_outer_replay_plan("epoch", list(range(2)))
        assert plan == [(0, 0), (1, 1)]
        assert silent == set()


def nested(epochs=3, steps=2):
    """Run a two-level flor.loop, logging from both levels; return what ran."""
    ran = []
    for epoch in api.loop("epoch", range(epochs)):
        for step in api.loop("step", range(steps)):
            ran.append((epoch, step))
            api.log("loss", 10 * epoch + step)
        api.log("val_acc", epoch)
    return ran


class TestLoopReplay:
    def test_nested_loop_runs_in_full_but_logs_only_its_selection(self, replaying):
        replaying.iter_specs = {
            "epoch": cli.IterSpec("indices", (1,)), "step": cli.IterSpec("last"),
        }
        ran = nested()
        assert ran == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert [(r.name, r.value) for r in logged(False)] == [
            ("loss", 11), ("val_acc", 1),
        ]

    def test_step_none_still_runs_every_step(self, replaying):
        replaying.iter_specs = {
            "epoch": cli.IterSpec("all"), "step": cli.IterSpec("none"),
        }
        ran = nested()
        assert len(ran) == 6
        assert [(r.name, r.value) for r in logged(False)] == [
            ("val_acc", 0), ("val_acc", 1), ("val_acc", 2),
        ]

    def test_suppression_does_not_leak_past_the_loop(self, replaying):
        replaying.iter_specs = {
            "epoch": cli.IterSpec("indices", (0,)), "step": cli.IterSpec("none"),
        }
        nested()
        api.log("after", 1)
        assert logged()[-1] == "after"
        assert api._suppress_logs is False

    def test_forward_mode_runs_and_logs_everything(self, clean_flags):
        api.output_buffer.clear()
        api.skip_cleanup = False
        try:
            assert len(nested()) == 6
            assert logged().count("loss") == 6
        finally:
            api.output_buffer.clear()
            api.skip_cleanup = True


class TestIterationReplay:
    """flor.iteration used to raise a bare `raise` under --replay_flor."""

    def test_runs_and_logs_under_replay(self, replaying):
        with api.iteration("epoch", 2, None):
            api.log("val_acc", 91)
        assert logged() == ["val_acc"]

    def test_records_the_loop_context(self, replaying):
        with api.iteration("epoch", 2, None):
            api.log("val_acc", 91)
        record = [r for r in api.output_buffer if r.name == "val_acc"][0]
        assert [(s.name, s.iteration) for s in record.ctx] == [("epoch", 2)]

    def test_unrequested_iterations_are_silent(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("indices", (0, 2))}
        for i in range(4):
            with api.iteration("epoch", i, None):
                api.log("val_acc", 90 + i)
        values = [r.value for r in api.output_buffer if r.name == "val_acc"]
        assert values == [90, 92]

    def test_none_suppresses_every_iteration(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("none")}
        with api.iteration("epoch", 0, None):
            api.log("val_acc", 90)
        assert logged() == []

    def test_last_is_not_decidable_so_everything_logs(self, replaying, capsys):
        # flor can't know which iteration is last when the user drives the loop.
        replaying.iter_specs = {"epoch": cli.IterSpec("last")}
        api._unbounded_last_warned.discard("epoch")
        for i in range(2):
            with api.iteration("epoch", i, None):
                api.log("val_acc", 90 + i)
        assert logged() == ["val_acc", "val_acc"]
        assert capsys.readouterr().out.count("not decidable") == 1

    def test_suppression_does_not_leak_past_the_block(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("none")}
        with api.iteration("epoch", 0, None):
            api.log("inside", 1)
        api.log("outside", 2)
        assert logged() == ["outside"]

    def test_emits_iteration_timing(self, replaying):
        with api.iteration("epoch", 0, None):
            pass
        assert [r.name for r in api.output_buffer] == ["time::setup", "time::iter"]

    def test_state_is_unwound_after_an_exception(self, replaying):
        with pytest.raises(ValueError):
            with api.iteration("epoch", 0, None):
                raise ValueError("boom")
        assert api.layers == {}
        assert api.context == []
        assert api._suppress_logs is False

    def test_forward_mode_is_unchanged(self, clean_flags):
        api.output_buffer.clear()
        api.skip_cleanup = False
        try:
            with api.iteration("epoch", 1, None):
                api.log("val_acc", 90)
            assert logged() == ["val_acc"]
            assert api.layers == {}
        finally:
            api.output_buffer.clear()
            api.skip_cleanup = True


class TestArgErrors:
    def test_missing_arg_without_default_says_what_to_do(self, clean_flags):
        api.skip_cleanup = False
        try:
            with pytest.raises(RuntimeError, match="has no default"):
                api.arg("undeclared")
        finally:
            api.skip_cleanup = True
            api.output_buffer.clear()

    def test_replay_missing_arg_without_default(self, replaying):
        with pytest.raises(RuntimeError, match="--override"):
            api.arg("undeclared")
