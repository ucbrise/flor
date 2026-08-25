"""Full-process tests: a forward run, then a replay of it.

Everything here goes through subprocesses in a throwaway git repo, so the
import-time machinery (project root discovery, `.flor/` layout, shadow branch,
auto-commit, JSONL write, sqlite cache) is exercised for real. No torch: the
checkpoint piggy-back has its own unit coverage, and requiring torch would make
the suite unrunnable in a bare environment.
"""

import json
import os

import pytest

pytestmark = pytest.mark.slow

TRAIN = '''
import flordb as flor

lr = flor.arg("lr", 1e-3)
epochs = flor.arg("epochs", 2)
steps = flor.arg("steps", 3)

for epoch in flor.loop("epoch", range(epochs)):
    for step in flor.loop("step", range(steps)):
        flor.log("loss", round(1.0 / (epoch + step + 1), 4))
    flor.log("val_acc", 90 + epoch)

flor.log("accuracy", 99)
'''


@pytest.fixture
def trained(project):
    project.write("train.py", TRAIN)
    project.run("train.py")
    return project


class TestForwardRun:
    def test_writes_one_jsonl_per_run(self, trained):
        assert len(trained.run_files()) == 1
        trained.run("train.py")
        assert len(trained.run_files()) == 2

    def test_run_file_is_named_for_the_run_tstamp(self, trained):
        records = trained.records()
        tstamp = os.path.basename(trained.run_files()[0])[: -len(".jsonl")]
        assert {r["tstamp"] for r in records} == {tstamp}

    def test_logs_land_with_self_describing_ctx(self, trained):
        records = trained.records()
        loss = next(r for r in records if r["name"] == "loss")
        assert [(s["name"], s["iteration"]) for s in loss["ctx"]] == [
            ("epoch", 0),
            ("step", 0),
        ]
        # Run-level records carry no ctx at all.
        assert next(r for r in records if r["name"] == "accuracy")["ctx"] is None

    def test_args_are_recorded_at_run_level(self, trained):
        records = trained.records()
        args = {r["name"]: r["value"] for r in records if r["ctx"] is None}
        assert args["lr"] == 1e-3
        assert args["epochs"] == 2

    def test_cli_kwargs_override_defaults(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py", "--kwargs", "epochs=1", "lr=5e-4")
        records = project.records()
        args = {r["name"]: r["value"] for r in records if r["ctx"] is None}
        assert args["epochs"] == 1
        assert args["lr"] == pytest.approx(5e-4)

    def test_iteration_times_are_summarized_not_per_iter(self, trained):
        records = trained.records()
        names = [r["name"] for r in records]
        # One summary triple per loop scope, not one record per inner iter.
        assert names.count("time::iter::n") == names.count("time::iter")
        assert names.count("time::iter::std") == names.count("time::iter")
        outer_n = next(
            r for r in records if r["name"] == "time::iter::n" and r["ctx"] is None
        )
        assert outer_n["value"] == 2  # epochs

    def test_script_wall_time_always_emitted(self, trained):
        assert any(r["name"] == "time::script" for r in trained.records())

    def test_flor_dir_is_gitignored_and_cmd_file_tracked(self, trained):
        with open(os.path.join(trained.root, ".gitignore")) as f:
            entries = f.read().split()
        assert ".flor/*" in entries
        assert "!.flor/runs/" in entries
        # The legacy bare entry would keep git from descending into .flor/ at
        # all, making the re-include below it dead.
        assert ".flor/" not in entries
        assert os.path.exists(os.path.join(trained.root, ".flor.cmd"))

    def test_run_jsonl_is_committed_but_cache_and_objstore_are_not(self, trained):
        tracked = trained.git_tracked_files()
        assert any(f.startswith(".flor/runs/") and f.endswith(".jsonl") for f in tracked)
        assert not any(f.startswith(".flor/obj_store/") for f in tracked)
        assert not any(f.endswith(".db") for f in tracked)

    def test_each_run_gets_a_commit_with_args_in_the_body(self, trained):
        trained.run("train.py", "--kwargs", "epochs=1")
        auto = [m for m in trained.git_log() if "FLOR::Auto-commit::" in m]
        assert len(auto) == 2
        assert "epochs=1" in auto[0]

    def test_cache_matches_jsonl(self, trained):
        conn = trained.db()
        try:
            rows = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT source) FROM logs WHERE source='forward'"
            ).fetchone()
        finally:
            conn.close()
        assert rows[0] == len(trained.records())


class TestUnpack:
    def test_rebuilds_the_cache_from_jsonl(self, trained):
        conn = trained.db()
        try:
            conn.execute("DELETE FROM logs")
            conn.commit()
        finally:
            conn.close()

        trained.run("-m", "flordb", "unpack")

        conn = trained.db()
        try:
            count = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        finally:
            conn.close()
        assert count == len(trained.records())

    def test_is_idempotent(self, trained):
        trained.run("-m", "flordb", "unpack")
        trained.run("-m", "flordb", "unpack")
        conn = trained.db()
        try:
            count = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        finally:
            conn.close()
        assert count == len(trained.records())


class TestReplay:
    """The script side of replay: `python train.py --replay_flor ...`."""

    def replay(self, project, *flags, check=True):
        return project.run("train.py", "--replay_flor", *flags, check=check)

    def rows(self, project, source):
        conn = project.db()
        try:
            return conn.execute(
                "SELECT value_name, value, ctx FROM logs WHERE source = ?", (source,)
            ).fetchall()
        finally:
            conn.close()

    def test_replay_does_not_write_a_new_run_file(self, trained):
        self.replay(trained, "--apply", "val_acc", "--iter", "epoch=all", "--iter", "step=none")
        assert len(trained.run_files()) == 1

    def test_replay_rows_are_tagged_replay(self, trained):
        self.replay(trained, "--apply", "val_acc", "--iter", "epoch=all", "--iter", "step=none")
        names = {name for name, _, _ in self.rows(trained, "replay")}
        assert "val_acc" in names

    def test_apply_projects_away_other_logs(self, trained):
        self.replay(trained, "--apply", "val_acc", "--iter", "epoch=all", "--iter", "step=none")
        names = {name for name, _, _ in self.rows(trained, "replay")}
        # loss is not projected; args ride along regardless so replayed rows
        # stay joinable to the config that produced them.
        assert "loss" not in names
        assert "lr" in names

    def test_iter_narrowing_selects_iterations(self, trained):
        self.replay(trained, "--apply", "val_acc", "--iter", "epoch=1", "--iter", "step=none")
        ctxs = [
            json.loads(ctx)
            for name, _, ctx in self.rows(trained, "replay")
            if name == "val_acc"
        ]
        assert [c[0]["iteration"] for c in ctxs] == [1]

    def test_replay_reuses_the_historical_tstamp(self, trained):
        self.replay(trained, "--apply", "val_acc", "--iter", "epoch=all", "--iter", "step=none")
        conn = trained.db()
        try:
            tstamps = conn.execute(
                "SELECT DISTINCT tstamp FROM logs WHERE source='replay'"
            ).fetchall()
        finally:
            conn.close()
        run_tstamp = os.path.basename(trained.run_files()[0])[: -len(".jsonl")]
        assert [t[0] for t in tstamps] == [run_tstamp]

    def test_unpack_wipes_replay_rows(self, trained):
        self.replay(trained, "--apply", "val_acc", "--iter", "epoch=all", "--iter", "step=none")
        assert self.rows(trained, "replay")
        trained.run("-m", "flordb", "unpack")
        assert self.rows(trained, "replay") == []

    def test_override_of_a_logged_hyperparameter_is_rejected(self, trained):
        proc = self.replay(trained, "--apply", "val_acc", "--override", "lr=1e-2", check=False)
        assert proc.returncode != 0
        assert "--override rejected" in proc.stderr

    def test_override_of_an_allowlisted_knob_is_accepted(self, trained):
        proc = self.replay(
            trained,
            "--apply",
            "val_acc",
            "--iter",
            "epoch=last",
            "--iter",
            "step=none",
            "--override",
            "ckpt_interval_s=0",
            check=False,
        )
        assert proc.returncode == 0, proc.stderr

    def test_narrowing_flags_require_replay_mode(self, trained):
        proc = trained.run("train.py", "--apply", "val_acc", check=False)
        assert proc.returncode != 0
        assert "require --replay_flor" in proc.stderr

    def test_kwargs_cannot_be_combined_with_replay(self, trained):
        proc = self.replay(trained, "--apply", "val_acc", "--kwargs", "lr=1e-2", check=False)
        assert proc.returncode != 0
        assert "Cannot combine --kwargs" in proc.stderr


ITERATION_SCRIPT = '''
import flordb as flor

epochs = flor.arg("epochs", 3)

for epoch in range(epochs):          # the user drives the loop
    with flor.iteration("epoch", epoch, None):
        flor.log("val_acc", 90 + epoch)

flor.log("accuracy", 99)
'''


class TestIterationEndToEnd:
    """flor.iteration used to abort with a bare `raise` under --replay_flor."""

    @pytest.fixture
    def trained_iteration(self, project):
        project.write("train.py", ITERATION_SCRIPT)
        project.run("train.py")
        return project

    def test_forward_run_records_each_iteration(self, trained_iteration):
        records = trained_iteration.records()
        accs = [r for r in records if r["name"] == "val_acc"]
        assert [r["ctx"][0]["iteration"] for r in accs] == [0, 1, 2]

    def test_replay_completes(self, trained_iteration):
        proc = trained_iteration.run(
            "train.py", "--replay_flor", "--apply", "val_acc", check=False
        )
        assert proc.returncode == 0, proc.stderr

    def test_replay_records_the_projected_variable(self, trained_iteration):
        trained_iteration.run("train.py", "--replay_flor", "--apply", "val_acc")
        conn = trained_iteration.db()
        try:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT value_name FROM logs WHERE source='replay'"
                )
            }
        finally:
            conn.close()
        assert "val_acc" in names

    def test_replay_narrowing_filters_iterations(self, trained_iteration):
        trained_iteration.run(
            "train.py", "--replay_flor", "--apply", "val_acc", "--iter", "epoch=0,2"
        )
        conn = trained_iteration.db()
        try:
            ctxs = [
                json.loads(row[0])
                for row in conn.execute(
                    "SELECT ctx FROM logs WHERE source='replay' AND value_name='val_acc'"
                )
            ]
        finally:
            conn.close()
        assert sorted(c[0]["iteration"] for c in ctxs) == [0, 2]
