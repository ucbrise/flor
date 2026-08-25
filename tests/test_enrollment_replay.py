"""Replay through `flor.checkpointing(...)` enrollment, with no torch in sight.

This is the other restore path: instead of an AST-detected `torch.load` resume
block, the user enrolls objects explicitly and flor serializes them at every
adaptive trigger. It has the same fresh-clone problem as the torch path -- a
teammate's clone carries runs/*.jsonl and no obj_store -- and has to reach the
same answer.
"""

import os
import shutil

import pytest

pytestmark = pytest.mark.slow

TRAIN = '''
import flordb as flor

flor.set_ckpt_interval(flor.arg("ckpt_interval_s", 0.0))
epochs = flor.arg("epochs", 3)

state = {"w": 0.0}

with flor.checkpointing(state=state):
    for epoch in flor.loop("epoch", range(epochs)):
        for step in flor.loop("step", range(2)):
            state["w"] += 1.0
        flor.log("w", state["w"])
'''


def shelf_dir(project):
    tstamp = os.path.basename(project.run_files()[0])[: -len(".jsonl")]
    return os.path.join(project.root, ".flor", "obj_store", tstamp)


def mirrors(project):
    shelf = shelf_dir(project)
    return sorted(os.listdir(shelf)) if os.path.isdir(shelf) else []


def values(project, source):
    import json

    conn = project.db()
    try:
        rows = conn.execute(
            "SELECT ctx, value FROM logs WHERE source = ? AND value_name = 'w'",
            (source,),
        ).fetchall()
    finally:
        conn.close()
    return {json.loads(ctx)[0]["iteration"]: float(v) for ctx, v in rows}


@pytest.fixture
def trained(project):
    project.write("train.py", TRAIN)
    project.run("train.py", "epochs=3")
    return project


class TestEnrolledForward:
    def test_each_iteration_is_shelved(self, trained):
        assert len(mirrors(trained)) == 3

    def test_values_advance(self, trained):
        assert values(trained, "forward") == {0: 2.0, 1: 4.0, 2: 6.0}


class TestEnrolledReplay:
    def test_narrowed_replay_restores_the_enrolled_object(self, trained):
        trained.run(
            "train.py", "--replay_flor", "--apply", "w",
            "--iter", "epoch=1", "--iter", "step=none",
        )
        # step=none means no work happens, so 4.0 can only come from epoch 1's
        # shelved dict being restored.
        assert values(trained, "replay") == {1: 4.0}

    def test_fresh_clone_replays_from_zero(self, trained):
        forward = values(trained, "forward")
        shutil.rmtree(shelf_dir(trained))

        proc = trained.run(
            "train.py", "--replay_flor", "--apply", "w",
            "--iter", "epoch=2", "--iter", "step=none", check=False,
        )

        # Nothing to restore from, so the loop reruns from iteration 0. There is
        # no module-scope resume block on this path, so the script's own
        # initialization is already the right starting state.
        assert proc.returncode == 0, proc.stderr
        assert values(trained, "replay")[2] == forward[2]

    def test_fresh_clone_replay_warms_the_shelf(self, trained):
        shutil.rmtree(shelf_dir(trained))

        trained.run(
            "train.py", "--replay_flor", "--apply", "w",
            "--iter", "epoch=2", "--iter", "step=none",
        )

        assert len(mirrors(trained)) == 3

    def test_second_replay_restores_instead_of_recomputing(self, trained):
        forward = values(trained, "forward")
        shutil.rmtree(shelf_dir(trained))

        for _ in range(2):
            trained.run(
                "train.py", "--replay_flor", "--apply", "w",
                "--iter", "epoch=2", "--iter", "step=none",
            )

        # The second pass finds epoch 2's warmed mirror and restores it directly.
        # step=none is honored again once no fast-forward is needed, so a wrong
        # warmed value would show up here as drift from the forward run.
        assert values(trained, "replay")[2] == forward[2]

    def test_throttled_forward_falls_back_to_the_earlier_mirror(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py", "--kwargs", "ckpt_interval_s=3600", "epochs=3")
        forward = values(project, "forward")
        # The throttle lets through epoch 0 (first iteration) and epoch 2 (the
        # forced save at outermost loop exit), so epoch 1 is the one with no
        # mirror of its own -- ask for exactly that.
        shelved = mirrors(project)
        assert "state_epoch_1.pkl" not in shelved

        proc = project.run(
            "train.py", "--replay_flor", "--apply", "w",
            "--iter", "epoch=1", "--iter", "step=none", check=False,
        )

        assert proc.returncode == 0, proc.stderr
        assert values(project, "replay")[1] == forward[1]
