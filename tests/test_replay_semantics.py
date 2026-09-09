"""In-process tests for the replay-time narrowing rules in flordb.api.

These drive api.slice / api.loop / api.iteration directly with cli.flags set
the way the CLI would set it, so they cover the narrowing contract without a
checkpoint store or torch.
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


class TestSlice:
    def test_all(self, replaying):
        replaying.iter_specs = {"step": cli.IterSpec("all")}
        assert api.slice("step", "abc") == [(0, "a"), (1, "b"), (2, "c")]

    def test_none(self, replaying):
        replaying.iter_specs = {"step": cli.IterSpec("none")}
        assert api.slice("step", "abc") == []

    def test_last(self, replaying):
        replaying.iter_specs = {"step": cli.IterSpec("last")}
        assert api.slice("step", "abc") == [(2, "c")]

    def test_indices(self, replaying):
        replaying.iter_specs = {"step": cli.IterSpec("indices", (0, 2))}
        assert api.slice("step", "abc") == [(0, "a"), (2, "c")]

    def test_out_of_range_is_rejected(self, replaying):
        replaying.iter_specs = {"step": cli.IterSpec("indices", (7,))}
        with pytest.raises(RuntimeError, match="only has 3 iteration"):
            api.slice("step", "abc")

    def test_unmentioned_loop_defaults_to_last(self, replaying):
        cli._defaulted_loops.discard("step")
        assert api.slice("step", "abc") == [(2, "c")]

    def test_forward_mode_passes_the_iterator_through(self, clean_flags):
        it = iter("abc")
        assert api.slice("step", it) is it


class TestOuterReplayPlan:
    def test_indices_without_resume_spec_run_as_requested(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("indices", (1, 3))}
        plan, mirror, silent, logical = api._build_outer_replay_plan(
            "epoch", list(range(5))
        )
        assert plan == [(1, 1), (3, 3)]
        assert (mirror, silent, logical) == (None, set(), False)

    def test_all(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("all")}
        plan, _, _, logical = api._build_outer_replay_plan("epoch", list(range(3)))
        assert plan == [(0, 0), (1, 1), (2, 2)]
        assert not logical

    def test_last(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("last")}
        plan, _, _, _ = api._build_outer_replay_plan("epoch", list(range(3)))
        assert plan == [(2, 2)]

    def test_out_of_range_names_the_valid_range(self, replaying):
        replaying.iter_specs = {"epoch": cli.IterSpec("indices", (9,))}
        with pytest.raises(RuntimeError, match=r"valid range: 0\.\.2"):
            api._build_outer_replay_plan("epoch", list(range(3)))

    def test_no_checkpointable_scope_runs_everything(self, replaying):
        replaying.wev_found = False
        replaying.iter_specs = {"epoch": cli.IterSpec("none")}
        plan, _, _, _ = api._build_outer_replay_plan("epoch", list(range(2)))
        assert plan == [(0, 0), (1, 1)]


class TestMirrorAddressing:
    def test_layer_matches_what_the_forward_run_wrote(self, replaying):
        # flor.loop forward sets layers[name] = (index, str(value) if jsonable).
        # A mismatch here means replay looks for a mirror filename that the
        # forward run never wrote.
        assert api._layer_for(list(range(5)), 3) == (3, "3")

    def test_unjsonable_values_fall_back_to_the_index(self, replaying):
        batches = [object(), object()]
        assert api._layer_for(batches, 1) == (1, None)

    def test_layer_swap_restores_previous_state(self, replaying):
        api.layers["epoch"] = (7, "7")
        with api._layer_swapped("epoch", 2, "2"):
            assert api.layers["epoch"] == (2, "2")
        assert api.layers["epoch"] == (7, "7")

    def test_layer_swap_removes_key_it_introduced(self, replaying):
        with api._layer_swapped("epoch", 2, "2"):
            assert "epoch" in api.layers
        assert "epoch" not in api.layers

    def test_restore_is_a_noop_without_a_resume_spec(self, replaying):
        assert api._restore_from_mirror("epoch", 0, "0") is False


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


class TestPredecessorPlan:
    @pytest.mark.parametrize("inner, mirror, logical", [
        ("none", None, False), ("all", 0, True), ("last", 0, True),
    ])
    def test_nested_selection_controls_checkpoint_boundary(self, replaying, monkeypatch, inner, mirror, logical):
        replaying.resume_spec = cli.ResumeSpec("ckpt.pth", None, [], source="explicit")
        replaying.loop_children = {"epoch": ["step"]}
        replaying.iter_specs = {
            "epoch": cli.IterSpec("indices", (1,)), "step": cli.IterSpec(inner),
        }
        monkeypatch.setattr(api, "_mirror_exists_at", lambda *args: True)
        plan, actual_mirror, silent, actual_logical = api._build_outer_replay_plan("epoch", [0, 1, 2])
        assert plan == [(1, 1)]
        assert (actual_mirror, silent, actual_logical) == (mirror, set(), logical)

    def test_skipped_parent_hides_its_descendants(self, replaying):
        replaying.loop_children = {"epoch": ["step"], "step": ["microbatch"]}
        replaying.iter_specs = {"step": cli.IterSpec("none"), "microbatch": cli.IterSpec("all")}
        assert not api._replay_enters_nested_loop("epoch")

    @pytest.mark.parametrize("kind", ["all", "last"])
    def test_empty_loop_has_no_checkpoint_to_restore(self, replaying, kind):
        replaying.iter_specs = {"epoch": cli.IterSpec(kind)}
        assert api._build_outer_replay_plan("epoch", []) == ([], None, set(), False)
