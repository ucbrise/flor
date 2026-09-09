"""Replay through `flor.checkpointing(...)` enrollment, with no torch in sight.

This is the other restore path: instead of an AST-detected `torch.load` resume
block, the user enrolls objects explicitly and flor serializes them at every
adaptive trigger. It has the same fresh-clone problem as the torch path -- a
teammate's clone carries runs/*.jsonl and no obj_store -- and has to reach the
same answer.
"""

import importlib.util
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


def values(project, source, name="w"):
    import json

    conn = project.db()
    try:
        rows = conn.execute(
            "SELECT ctx, value FROM logs WHERE source = ? AND value_name = ?",
            (source, name),
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


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def _has_grad_scaler() -> bool:
    if not _has_torch():
        return False
    import torch

    return hasattr(getattr(torch, "amp", None), "GradScaler")


requires_torch = pytest.mark.skipif(not _has_torch(), reason="torch not installed")
requires_grad_scaler = pytest.mark.skipif(
    not _has_grad_scaler(), reason="torch.amp.GradScaler not available"
)


# Everything an accelerated training loop keeps beside the weights. The
# scheduler and the scaler are neither a Module nor an Optimizer, so matching
# on those two classes used to drop them into cloudpickle and fail on restore;
# they carry state_dict()/load_state_dict() like the rest, which is what the
# serializer dispatches on now.
ACCEL_TRAIN = '''
import torch
import torch.nn as nn

import flordb as flor

flor.set_ckpt_interval(flor.arg("ckpt_interval_s", 0.0))
epochs = flor.arg("epochs", 3)

torch.manual_seed(0)
net = nn.Linear(2, 1)
opt = torch.optim.SGD(net.parameters(), lr=0.1)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
scaler = torch.amp.GradScaler("cuda", enabled=False)

with flor.checkpointing(model=net, optimizer=opt, sched=sched, scaler=scaler):
    for epoch in flor.loop("epoch", range(epochs)):
        for step in flor.loop("step", range(2)):
            with torch.no_grad():
                net.weight += 1.0
            sched.step()
        flor.log("w", round(float(net.weight[0][0].detach()), 4))
        flor.log("lr", round(sched.get_last_lr()[0], 8))
'''


@requires_torch
@requires_grad_scaler
class TestEnrolledTorchStack:
    @pytest.fixture
    def trained(self, project):
        project.write("train.py", ACCEL_TRAIN)
        project.run("train.py", "--kwargs", "epochs=3")
        return project

    def test_every_enrolled_object_is_shelved_as_a_state_dict(self, trained):
        # .pth, not .pkl: all four went through the state_dict backend.
        assert set(mirrors(trained)) == {
            f"{name}_epoch_{i}.pth"
            for name in ("model", "optimizer", "sched", "scaler")
            for i in range(3)
        }

    def test_replay_restores_the_module(self, trained):
        forward = values(trained, "forward")
        trained.run(
            "train.py", "--replay_flor", "--apply", "w,lr",
            "--iter", "epoch=1", "--iter", "step=none",
        )
        assert values(trained, "replay")[1] == forward[1]

    def test_replay_restores_the_scheduler(self, trained):
        # The scheduler's own state, not a value recomputable from the weights:
        # if it were skipped, the replayed lr would be epoch 0's, not epoch 1's.
        forward = values(trained, "forward", "lr")
        trained.run(
            "train.py", "--replay_flor", "--apply", "w,lr",
            "--iter", "epoch=1", "--iter", "step=none",
        )
        assert values(trained, "replay", "lr")[1] == forward[1]


PLAIN_TRAIN = '''
import flordb as flor

flor.set_ckpt_interval(0.0)


class Tracker:
    def __init__(self):
        self.n = 0


tracker = Tracker()

with flor.checkpointing(tracker=tracker):
    for epoch in flor.loop("epoch", range(3)):
        for step in flor.loop("step", range(2)):
            tracker.n += 1
        flor.log("w", tracker.n)
'''

SLOTS_TRAIN = '''
import flordb as flor


class Slotted:
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0


with flor.checkpointing(s=Slotted()):
    for epoch in flor.loop("epoch", range(2)):
        flor.log("w", 1)
'''

NESTED_TRAIN = '''
import flordb as flor

flor.set_ckpt_interval(0.0)
outer = {"a": 0}
inner = {"b": 0}

with flor.checkpointing(outer=outer):
    with flor.checkpointing(inner=inner):
        pass
    for epoch in flor.loop("epoch", range(2)):
        outer["a"] += 1
        flor.log("w", outer["a"])
'''


class TestEnrolledPlainObjects:
    def test_a_plain_object_round_trips(self, project):
        project.write("train.py", PLAIN_TRAIN)
        project.run("train.py")
        forward = values(project, "forward")

        project.run(
            "train.py", "--replay_flor", "--apply", "w",
            "--iter", "epoch=1", "--iter", "step=none",
        )

        # cloudpickle has no trouble writing a Tracker; putting one back means
        # moving state across in place, through its instance dict.
        assert values(project, "replay")[1] == forward[1]

    def test_an_unrestorable_object_is_refused_at_enrollment(self, project):
        project.write("train.py", SLOTS_TRAIN)
        proc = project.run("train.py", check=False)

        # __slots__, so no instance dict to restore through. Refusing here beats
        # shelving a run's worth of snapshots that only fail on the replay that
        # needed one.
        assert proc.returncode != 0
        assert "flor.checkpointing(s=...)" in proc.stdout + proc.stderr

    def test_a_nested_block_keeps_the_outer_enrollment(self, project):
        project.write("train.py", NESTED_TRAIN)
        project.run("train.py")

        # The inner block's exit used to clear the whole enrollment list, so the
        # loop that follows checkpointed nothing at all.
        assert mirrors(project) == ["outer_epoch_0.pkl", "outer_epoch_1.pkl"]


# Default ckpt_interval_s (60s) and a loop that finishes in well under it, so
# exactly one iteration is a checkpoint iteration. Which one each stream picks
# is the whole question: the enrolled object goes through ckpt() at the end of
# the iteration, the script's own torch.save through the mirror hook during it.
ALIGNED_TRAIN = '''
import torch
import torch.nn as nn

import flordb as flor

epochs = flor.arg("epochs", 4)

torch.manual_seed(0)
net = nn.Linear(2, 1)

with flor.checkpointing(model=net):
    for epoch in flor.loop("epoch", range(epochs)):
        for step in flor.loop("step", range(2)):
            with torch.no_grad():
                net.weight += 1.0
        torch.save(net.state_dict(), "ckpt.pth")
        flor.log("w", round(float(net.weight[0][0].detach()), 4))
'''

INNER_SAVE_TRAIN = ALIGNED_TRAIN.replace(
    '        torch.save(net.state_dict(), "ckpt.pth")\n', ""
).replace(
    "                net.weight += 1.0\n",
    '                net.weight += 1.0\n            torch.save(net.state_dict(), "ckpt.pth")\n',
)


@requires_torch
class TestStreamsAgreeOnTheSameIterations:
    """The enrolled stream and the torch.save mirror share one trigger.

    They used to share a clock instead, which is not the same thing: the hook
    runs inside the loop body and got there first every time, so it spent the
    interval and the ckpt() at the end of that iteration was throttled out. The
    enrolled object then only ever landed on the forced save at loop exit --
    never on an iteration the mirror also covered, which is precisely the
    pairing _mirror_exists_at needs before it will restore rather than recompute.
    """

    @pytest.fixture
    def trained(self, project):
        project.write("train.py", ALIGNED_TRAIN)
        project.run("train.py", "--kwargs", "epochs=4")
        return project

    def test_both_streams_land_on_the_same_iteration(self, trained):
        shelved = mirrors(trained)
        assert "ckpt_epoch_0.pth" in shelved
        assert "model_epoch_0.pth" in shelved

    def test_the_paired_iteration_restores_without_recomputing(self, trained):
        forward = values(trained, "forward")

        proc = trained.run(
            "train.py", "--replay_flor", "--apply", "w",
            "--iter", "epoch=0", "--iter", "step=none",
        )

        assert values(trained, "replay")[0] == forward[0]
        assert "replaying from iteration 0" not in proc.stdout + proc.stderr

    def test_a_save_in_an_inner_loop_mirrors_once_per_iteration(self, project):
        project.write("train.py", INNER_SAVE_TRAIN)
        project.run("train.py", "--kwargs", "epochs=4")

        # Armed is a per-iteration decision, so without a one-shot guard the
        # hook would mirror on every step of the inner loop.
        assert len([m for m in mirrors(project) if m.startswith("ckpt_")]) == 1
