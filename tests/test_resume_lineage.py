"""Replay of a run that resumed from an earlier run's checkpoint.

A forward run records where its training started -- which earlier run's copy of
the checkpoint it loaded, by that run's tstamp and commit -- and replay loads
the same copy instead of starting from initialization.
"""

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.slow

# Resumes by loading the file the previous run saved.
IMPLICIT = '''
import os

import torch
import torch.nn as nn

import flordb as flor

epochs = flor.arg("epochs", 2)

torch.manual_seed(flor.arg("seed", 42))
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

if os.path.exists("ckpt.pth"):
    _resume = torch.load("ckpt.pth")
    model.load_state_dict(_resume["model"])
    optimizer.load_state_dict(_resume["optimizer"])

x = torch.ones(4, 2)
y = torch.zeros(4, 1)

for epoch in flor.loop("epoch", range(epochs)):
    for step in flor.loop("step", range(2)):
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    flor.log("weight_sum", float(sum(p.sum() for p in model.parameters())))
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        "ckpt.pth",
    )
'''

# Resumes from whichever earlier forward run a query over flor picks -- here the
# latest one, which is a different run by the time this one is replayed.
EXPLICIT = '''
import torch
import torch.nn as nn

import flordb as flor

epochs = flor.arg("epochs", 2)

torch.manual_seed(flor.arg("seed", 42))
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

runs = flor.dataframe("weight_sum")
if len(runs):
    latest = runs[runs.source == "forward"].tstamp.max()
    state = flor.load_checkpoint(latest, "ckpt.pth")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

x = torch.ones(4, 2)
y = torch.zeros(4, 1)

for epoch in flor.loop("epoch", range(epochs)):
    for step in flor.loop("step", range(2)):
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    flor.log("weight_sum", float(sum(p.sum() for p in model.parameters())))
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        "ckpt.pth",
    )
'''


def tstamp_of(run_file):
    return os.path.basename(run_file)[: -len(".jsonl")]


def starts(project, run_file):
    return [
        json.loads(record["value"])
        for record in project.records(run_file)
        if record["type"] == 4
    ]


def values(project, source, tstamp=None):
    conn = project.db()
    try:
        sql = (
            "SELECT ctx, value FROM logs "
            "WHERE source = ? AND value_name = 'weight_sum'"
        )
        params = [source]
        if tstamp is not None:
            sql += " AND tstamp = ?"
            params.append(tstamp)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return {json.loads(ctx)[0]["iteration"]: float(v) for ctx, v in rows}


def replay(project, check=True):
    return project.run(
        "train.py", "--replay_flor", "--apply", "weight_sum", "--iter", "epoch=all",
        check=check,
    )


def two_runs(project, source):
    project.write("train.py", source)
    project.run("train.py")
    project.run("train.py")
    return project.run_files()


class TestImplicitResume:
    @pytest.mark.parametrize("missing_copy", [False, True])
    def test_skipped_file_guard_refuses_before_recording_values(self, project, missing_copy):
        first, second = two_runs(project, IMPLICIT)
        (Path(project.root) / "ckpt.pth").unlink()
        if missing_copy:
            shutil.rmtree(Path(project.root) / ".flor" / "obj_store" / tstamp_of(first))

        proc = replay(project, check=False)

        assert proc.returncode != 0
        message = "isn't in .flor/obj_store/" if missing_copy else "skipped that setup load"
        assert message in proc.stdout + proc.stderr
        assert values(project, "replay") == {}
        assert not (Path(project.root) / "ckpt.pth").exists()

        if not missing_copy:
            # Let the load execute without recreating or modifying the user's
            # checkpoint; the replay hook must supply the historical copy.
            source = IMPLICIT.replace('if os.path.exists("ckpt.pth"):\n', "")
            for statement in (
                '_resume = torch.load("ckpt.pth")',
                'model.load_state_dict(_resume["model"])',
                'optimizer.load_state_dict(_resume["optimizer"])',
            ):
                source = source.replace("    " + statement, statement)
            project.write("train.py", source)
            replay(project)
            assert values(project, "replay") == pytest.approx(
                values(project, "forward", tstamp_of(second)), rel=1e-6,
            )
            assert not (Path(project.root) / "ckpt.pth").exists()

    def test_importing_flor_before_torch_tracks_early_resume(self, project):
        source = "import flordb as flor\n" + IMPLICIT.replace("import flordb as flor\n", "")
        source = source.replace('epochs = flor.arg("epochs", 2)', "epochs = 2")
        source = source.replace('torch.manual_seed(flor.arg("seed", 42))', "torch.manual_seed(42)")
        first, second = two_runs(project, source)

        [start] = starts(project, second)
        assert pd.Timestamp(start["tstamp"]) == pd.Timestamp(tstamp_of(first))
        replay(project)
        assert values(project, "replay") == pytest.approx(
            values(project, "forward", tstamp_of(second)), rel=1e-6,
        )

    def test_second_run_records_the_first_as_its_start(self, project):
        first, second = two_runs(project, IMPLICIT)

        assert starts(project, first) == []
        [start] = starts(project, second)
        assert start["via"] == "torch.load"
        assert start["key"] == "ckpt.pth"
        assert start["name"] == "ckpt.pth"
        assert pd.Timestamp(start["tstamp"]) == pd.Timestamp(tstamp_of(first))
        assert start["commit"]

    def test_replay_starts_where_the_run_did(self, project):
        # Starting from initialization instead would reproduce the first run's
        # numbers, not the second's.
        first, second = two_runs(project, IMPLICIT)
        forward = values(project, "forward", tstamp_of(second))
        assert forward != values(project, "forward", tstamp_of(first))

        replay(project)

        assert values(project, "replay") == pytest.approx(forward, rel=1e-6)

    def test_a_file_changed_outside_flor_is_not_attributed(self, project):
        project.write("train.py", IMPLICIT)
        project.run("train.py")
        # Rewritten without flor: same contents, but no run saved this file.
        project.run("-c", "import torch; torch.save(torch.load('ckpt.pth'), 'ckpt.pth')")
        project.run("train.py")

        assert starts(project, project.run_files()[-1]) == []

    def test_missing_copy_stops_replay_and_names_the_run(self, project):
        first, _ = two_runs(project, IMPLICIT)
        shutil.rmtree(Path(project.root) / ".flor" / "obj_store" / tstamp_of(first))

        proc = replay(project, check=False)

        combined = proc.stdout + proc.stderr
        assert proc.returncode != 0
        assert "isn't in .flor/obj_store/" in combined
        assert str(pd.Timestamp(tstamp_of(first)).isoformat(timespec="microseconds")) in combined


class TestExplicitResume:
    def test_fresh_run_refuses_a_query_that_now_finds_a_checkpoint(self, project):
        project.write("train.py", EXPLICIT)
        project.run("train.py")

        proc = replay(project, check=False)

        assert proc.returncode != 0
        assert "historical run recorded no matching load" in proc.stdout + proc.stderr
        assert values(project, "replay") == {}

    @pytest.mark.parametrize("name", [None, "ckpt.pth"])
    @pytest.mark.parametrize("in_loop", [False, True])
    def test_same_name_can_load_two_different_runs(self, project, name, in_loop):
        first, second = two_runs(project, IMPLICIT)
        source = f'''
import flordb as flor
runs = flor.dataframe("weight_sum")
stamps = sorted(runs[runs.source == "forward"].tstamp.unique())
'''
        if in_loop:
            source += f'''
for epoch in flor.loop("epoch", range(2)):
    state = flor.load_checkpoint(stamps[epoch], {name!r})
    flor.log("weight_sum", float(state["model"]["weight"].sum()))
'''
        else:
            source += f'''
a = flor.load_checkpoint(stamps[0], {name!r})
b = flor.load_checkpoint(stamps[1], {name!r})
for epoch in flor.loop("epoch", range(1)):
    flor.log("weight_sum", float(a["model"]["weight"].sum() - b["model"]["weight"].sum()))
'''
        project.write("train.py", source)
        project.run("train.py")
        run = project.run_files()[-1]
        recorded = starts(project, run)
        assert [s["call"] for s in recorded] == [0, 1]
        assert [pd.Timestamp(s["tstamp"]) for s in recorded] == [
            pd.Timestamp(tstamp_of(first)), pd.Timestamp(tstamp_of(second)),
        ]

        replay(project)

        assert values(project, "replay") == pytest.approx(
            values(project, "forward", tstamp_of(run)), rel=1e-6,
        )

    def test_old_lineage_without_call_numbers_still_replays(self, project):
        _, second = two_runs(project, EXPLICIT)
        records = project.records(second)
        for record in records:
            if record["type"] == 4:
                start = json.loads(record["value"])
                start.pop("call")
                start.pop("setup")
                record["value"] = json.dumps(start)
        Path(second).write_text("".join(json.dumps(r) + "\n" for r in records))

        replay(project)

        assert values(project, "replay") == pytest.approx(
            values(project, "forward", tstamp_of(second)), rel=1e-6,
        )

    def test_the_query_answer_is_recorded(self, project):
        first, second = two_runs(project, EXPLICIT)

        assert starts(project, first) == []
        [start] = starts(project, second)
        assert start["via"] == "load_checkpoint"
        assert start["key"] == "ckpt.pth"
        assert pd.Timestamp(start["tstamp"]) == pd.Timestamp(tstamp_of(first))
        assert start["commit"]

    def test_replay_uses_the_recorded_run_not_todays_answer(self, project):
        # Replaying the second run, the query's latest forward run is the
        # second run itself. The recorded answer -- the first -- has to win.
        _, second = two_runs(project, EXPLICIT)
        forward = values(project, "forward", tstamp_of(second))

        proc = replay(project)

        assert "now selects run" in proc.stdout + proc.stderr
        assert values(project, "replay") == pytest.approx(forward, rel=1e-6)

    def test_reading_from_python_c_records_no_run(self, project):
        project.write("train.py", EXPLICIT)
        project.run("train.py")
        before = project.run_files()

        project.run("-c", (
            "import flordb as flor\n"
            "stamp = flor.dataframe('weight_sum').tstamp.max()\n"
            "flor.load_checkpoint(stamp, 'ckpt.pth')\n"
        ))

        assert project.run_files() == before
