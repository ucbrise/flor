"""The per-run copy of the script's checkpoint, and replay's handling of it.

A forward run keeps one copy of each file the script saves with torch.save.
Replay keeps no checkpoints at all: it recomputes from iteration 0, leaves the
user's file alone, and turns the script's resume block into a no-op so the
recomputation starts from the script's own initialization.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.slow

TRAIN = '''
import os

import torch
import torch.nn as nn

import flordb as flor

epochs = flor.arg("epochs", 3)

torch.manual_seed(flor.arg("seed", 42))
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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

# Momentum carries state across epochs, so only a faithful recomputation from
# iteration 0 lands on the forward run's numbers.
MOMENTUM_TRAIN = TRAIN.replace("lr=0.1)", "lr=0.1, momentum=0.9)")


# Same training, but the user drives the outer loop and marks each pass with
# flor.iteration instead of handing the iterator to flor.loop.
ITERATION_TRAIN = '''
import os

import torch
import torch.nn as nn

import flordb as flor

epochs = flor.arg("epochs", 3)

torch.manual_seed(flor.arg("seed", 42))
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

if os.path.exists("ckpt.pth"):
    _resume = torch.load("ckpt.pth")
    model.load_state_dict(_resume["model"])
    optimizer.load_state_dict(_resume["optimizer"])

x = torch.ones(4, 2)
y = torch.zeros(4, 1)

for epoch in range(epochs):
    with flor.iteration("epoch", epoch, None):
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


@pytest.fixture
def trained(project):
    project.write("train.py", MOMENTUM_TRAIN)
    project.run("train.py")
    return project


def store_root(project):
    return Path(project.root) / ".flor" / "obj_store"


def shelf_dir(project):
    tstamp = os.path.basename(project.run_files()[0])[: -len(".jsonl")]
    return store_root(project) / tstamp


def store_contents(project):
    """Every file under .flor/obj_store, with its bytes."""
    root = store_root(project)
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def values(project, source):
    conn = project.db()
    try:
        rows = conn.execute(
            "SELECT ctx, value FROM logs "
            "WHERE source = ? AND value_name = 'weight_sum'",
            (source,),
        ).fetchall()
    finally:
        conn.close()
    return {json.loads(ctx)[0]["iteration"]: float(v) for ctx, v in rows}


def replay(project, *selections, extra=(), check=True):
    argv = ["train.py", "--replay_flor", "--apply", "weight_sum"]
    for selection in selections:
        argv += ["--iter", selection]
    return project.run(*argv, *extra, check=check)


class TestForwardCopy:
    def test_one_copy_per_run(self, trained):
        shelf = shelf_dir(trained)
        # Nothing per iteration: the run's copy under .latest/ is all there is.
        assert [p.name for p in shelf.iterdir()] == [".latest"]
        index = json.loads((shelf / ".latest" / "index.json").read_text())
        assert list(index) == ["ckpt.pth"]

    def test_copy_holds_the_last_save(self, trained):
        latest = shelf_dir(trained) / ".latest"
        entry = json.loads((latest / "index.json").read_text())["ckpt.pth"]
        user_file = Path(trained.root) / "ckpt.pth"
        assert (latest / entry["file"]).read_bytes() == user_file.read_bytes()

    def test_each_run_keeps_its_own_copy(self, project):
        project.write("train.py", MOMENTUM_TRAIN)
        project.run("train.py", "--kwargs", "epochs=1")
        project.run("train.py", "--kwargs", "epochs=2")
        copies = sorted(store_root(project).glob("*/.latest/*.pth"))
        assert len(copies) == 2
        assert copies[0].read_bytes() != copies[1].read_bytes()


class TestReplay:
    @pytest.mark.parametrize("checkpoint_exists", [True, False])
    @pytest.mark.parametrize("overrides", [(), ("--override", "device=cpu")])
    def test_replay_does_not_write_users_checkpoint(
        self, trained, checkpoint_exists, overrides
    ):
        ckpt = Path(trained.root) / "ckpt.pth"
        before = ckpt.read_bytes()
        mtime = ckpt.stat().st_mtime_ns
        if not checkpoint_exists:
            ckpt.unlink()

        replay(trained, "epoch=1", extra=overrides)

        if checkpoint_exists:
            assert ckpt.read_bytes() == before
            assert ckpt.stat().st_mtime_ns == mtime
        else:
            assert not ckpt.exists()

    def test_replay_writes_nothing_to_the_object_store(self, trained):
        before = store_contents(trained)

        replay(trained, "epoch=all")

        assert store_contents(trained) == before

    @pytest.mark.parametrize("selection, expected", [
        ("0", [0]), ("1", [1]), ("0,2", [0, 2]),
        ("all", [0, 1, 2]), ("last", [2]),
    ])
    def test_replayed_values_match_the_forward_run(self, trained, selection, expected):
        forward = values(trained, "forward")

        replay(trained, "epoch=" + selection, "step=all")

        assert values(trained, "replay") == pytest.approx(
            {i: forward[i] for i in expected}, rel=1e-6,
        )

    def test_step_none_still_trains(self, trained):
        # --iter only chooses what logs. Skipping the training steps would
        # leave every epoch after the first on the wrong weights.
        forward = values(trained, "forward")

        replay(trained, "epoch=all", "step=none")

        assert values(trained, "replay") == pytest.approx(forward, rel=1e-6)

    def test_user_driven_iteration_replays_from_the_top(self, project):
        # flor.iteration can't skip the body -- the user's own `for` drives it
        # -- so it runs every pass and stays quiet for the unrequested ones.
        project.write("train.py", ITERATION_TRAIN)
        project.run("train.py")
        forward = values(project, "forward")

        replay(project, "epoch=1")

        assert values(project, "replay") == pytest.approx({1: forward[1]}, rel=1e-6)

    def test_fresh_clone_matches_forward(self, trained):
        # A teammate's clone has runs/*.jsonl and neither the object store nor
        # the user's ckpt.pth. Replay needs neither.
        forward = values(trained, "forward")
        shutil.rmtree(store_root(trained))
        os.remove(os.path.join(trained.root, "ckpt.pth"))

        replay(trained, "epoch=2")

        assert values(trained, "replay") == pytest.approx({2: forward[2]}, rel=1e-6)

    def test_users_checkpoint_is_neutralized_not_loaded(self, trained):
        # ckpt.pth holds end-of-run state. The module-scope resume block would
        # load it over the fresh init, and recomputing on top of that would
        # report epoch 2 as something else entirely.
        forward = values(trained, "forward")
        assert os.path.exists(os.path.join(trained.root, "ckpt.pth"))

        replay(trained, "epoch=2")

        assert values(trained, "replay") == pytest.approx({2: forward[2]}, rel=1e-6)
