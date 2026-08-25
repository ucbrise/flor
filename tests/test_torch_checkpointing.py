"""The torch.save piggy-back and its replay counterpart.

This is the path with no explicit `flor.checkpointing(...)` enrollment: flor
mirrors the user's own torch.save into `.flor/obj_store/<tstamp>/`, and on
replay redirects torch.load to the mirror for the iteration being replayed.
Mirror filenames are derived from the loop context, so forward and replay have
to agree on the addressing exactly -- that agreement is what these check.
"""

import glob
import os

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.slow

TRAIN = '''
import os

import torch
import torch.nn as nn

import flordb as flor

flor.set_ckpt_interval(flor.arg("ckpt_interval_s", 0.0))
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


# Same training, but the user drives the outer loop and marks each pass with
# flor.iteration instead of handing the iterator to flor.loop.
ITERATION_TRAIN = '''
import os

import torch
import torch.nn as nn

import flordb as flor

flor.set_ckpt_interval(flor.arg("ckpt_interval_s", 0.0))
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
    project.write("train.py", TRAIN)
    project.run("train.py")
    return project


def mirrors(project):
    tstamp = os.path.basename(project.run_files()[0])[: -len(".jsonl")]
    shelf = os.path.join(project.root, ".flor", "obj_store", tstamp)
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(shelf, "*")))


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
    import json

    return {json.loads(ctx)[0]["iteration"]: float(v) for ctx, v in rows}


class TestForwardMirroring:
    def test_user_torch_save_is_mirrored_per_iteration(self, trained):
        # No flor.checkpointing block in the script -- these exist purely
        # because the user called torch.save inside a flor.loop.
        assert mirrors(trained) == ["ckpt_epoch_0.pth", "ckpt_epoch_1.pth", "ckpt_epoch_2.pth"]

    def test_throttle_bounds_the_mirrors(self, project):
        project.write("train.py", TRAIN)
        # A big interval means only the first save (and the forced one at
        # outermost loop exit) get through.
        project.run("train.py", "--kwargs", "ckpt_interval_s=3600")
        assert len(mirrors(project)) < 3

    def test_user_checkpoint_file_is_still_written(self, trained):
        assert os.path.exists(os.path.join(trained.root, "ckpt.pth"))


class TestReplayRestore:
    def test_replayed_values_match_the_forward_run(self, trained):
        forward = values(trained, "forward")

        trained.run(
            "train.py",
            "--replay_flor",
            "--apply",
            "weight_sum",
            "--iter",
            "epoch=all",
            "--iter",
            "step=none",
        )

        replayed = values(trained, "replay")
        assert set(replayed) == set(forward)
        # step=none means no training happens, so each epoch's value can only
        # be right if that epoch's mirror was found and restored.
        for epoch, value in forward.items():
            assert replayed[epoch] == pytest.approx(value, rel=1e-6)

    def test_narrowed_replay_restores_the_requested_epoch(self, trained):
        forward = values(trained, "forward")

        trained.run(
            "train.py",
            "--replay_flor",
            "--apply",
            "weight_sum",
            "--iter",
            "epoch=1",
            "--iter",
            "step=none",
        )

        replayed = values(trained, "replay")
        assert list(replayed) == [1]
        assert replayed[1] == pytest.approx(forward[1], rel=1e-6)

    def test_replay_never_overwrites_an_existing_mirror(self, trained):
        before = mirrors(trained)
        digests = {
            name: os.path.getsize(
                os.path.join(
                    trained.root,
                    ".flor",
                    "obj_store",
                    os.path.basename(trained.run_files()[0])[: -len(".jsonl")],
                    name,
                )
            )
            for name in before
        }
        trained.run(
            "train.py",
            "--replay_flor",
            "--apply",
            "weight_sum",
            "--iter",
            "epoch=all",
            "--iter",
            "step=none",
        )
        assert mirrors(trained) == before
        for name, size in digests.items():
            shelf = os.path.join(
                trained.root,
                ".flor",
                "obj_store",
                os.path.basename(trained.run_files()[0])[: -len(".jsonl")],
                name,
            )
            assert os.path.getsize(shelf) == size

    def test_user_driven_iteration_restores_its_own_mirror(self, project):
        # flor.iteration under --replay_flor used to abort outright. It now
        # restores the mirror for the iteration it is marking, and stays quiet
        # for iterations the user didn't ask for (it can't skip the body --
        # the user's own `for` drives it).
        project.write("train.py", ITERATION_TRAIN)
        project.run("train.py")
        forward = values(project, "forward")

        project.run(
            "train.py",
            "--replay_flor",
            "--apply",
            "weight_sum",
            "--iter",
            "epoch=1",
            "--iter",
            "step=none",
        )

        replayed = values(project, "replay")
        assert list(replayed) == [1]
        assert replayed[1] == pytest.approx(forward[1], rel=1e-6)

    def test_throttled_mirror_falls_back_to_logical_replay(self, project):
        project.write("train.py", TRAIN)
        # Throttled hard enough that only epoch 0 keeps a mirror. Asking for
        # epoch 1 must restart from epoch 0's mirror and fast-forward through
        # epoch 1 (inner loop in full, --iter step=none deliberately ignored)
        # rather than replay an uninitialized model.
        project.run("train.py", "--kwargs", "ckpt_interval_s=3600", "epochs=3")
        forward = values(project, "forward")
        assert len(mirrors(project)) == 1

        proc = project.run(
            "train.py",
            "--replay_flor",
            "--apply",
            "weight_sum",
            "--iter",
            "epoch=1",
            "--iter",
            "step=none",
            check=False,
        )

        assert proc.returncode == 0, proc.stderr
        replayed = values(project, "replay")
        assert list(replayed) == [1]
        assert replayed[1] == pytest.approx(forward[1], rel=1e-6)


def shelf_dir(project):
    tstamp = os.path.basename(project.run_files()[0])[: -len(".jsonl")]
    return os.path.join(project.root, ".flor", "obj_store", tstamp)


def simulate_fresh_clone(project):
    """What a teammate gets from `git clone`: runs/*.jsonl but no obj_store.

    The user's own ckpt.pth goes too -- a clone that still had it would be
    carrying end-of-run weights that the script's module-scope resume block
    would load over its initialization.
    """
    import shutil

    shutil.rmtree(shelf_dir(project))
    ckpt = os.path.join(project.root, "ckpt.pth")
    if os.path.exists(ckpt):
        os.remove(ckpt)


class TestCheckpointWarming:
    def test_fast_forward_shelves_the_mirrors_it_passes(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py", "--kwargs", "ckpt_interval_s=3600", "epochs=3")
        assert len(mirrors(project)) == 1  # only epoch 0 survived the throttle

        project.run(
            "train.py", "--replay_flor", "--apply", "weight_sum",
            "--iter", "epoch=1", "--iter", "step=none",
        )

        # The replay recomputed epoch 1 to answer the query; warming keeps that
        # state instead of throwing it away.
        assert "ckpt_epoch_1.pth" in mirrors(project)

    def test_warmed_mirror_makes_the_next_replay_a_direct_restore(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py", "--kwargs", "ckpt_interval_s=3600", "epochs=3")
        forward = values(project, "forward")

        for _ in range(2):
            project.run(
                "train.py", "--replay_flor", "--apply", "weight_sum",
                "--iter", "epoch=1", "--iter", "step=none",
            )

        # Second replay restores epoch 1's warmed mirror directly rather than
        # fast-forwarding from epoch 0, and must land on the same value.
        replayed = values(project, "replay")
        assert replayed[1] == pytest.approx(forward[1], rel=1e-6)

    def test_warming_is_off_when_an_override_could_move_the_numbers(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py", "--kwargs", "ckpt_interval_s=3600", "epochs=3")
        before = mirrors(project)

        project.run(
            "train.py", "--replay_flor", "--apply", "weight_sum",
            "--iter", "epoch=1", "--iter", "step=none",
            "--override", "device=cpu",
        )

        # device=cpu is allowlisted for replay but cpu/cuda kernels don't agree
        # bit-for-bit, so whatever this recomputed is not forward-run truth and
        # must not be shelved as if it were.
        assert mirrors(project) == before

    def test_fresh_clone_replays_from_zero_and_matches_forward(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py", "epochs=3")
        forward = values(project, "forward")
        simulate_fresh_clone(project)

        proc = project.run(
            "train.py", "--replay_flor", "--apply", "weight_sum",
            "--iter", "epoch=2", "--iter", "step=none", check=False,
        )

        assert proc.returncode == 0, proc.stderr
        replayed = values(project, "replay")
        assert replayed[2] == pytest.approx(forward[2], rel=1e-6)
        # And it paid the recompute once: the mirrors are back on the shelf.
        assert "ckpt_epoch_2.pth" in mirrors(project)

    def test_stale_user_checkpoint_blocks_replay_from_zero(self, project):
        project.write("train.py", TRAIN)
        project.run("train.py", "epochs=3")
        import shutil

        shutil.rmtree(shelf_dir(project))  # obj_store gone, ckpt.pth left behind

        proc = project.run(
            "train.py", "--replay_flor", "--apply", "weight_sum",
            "--iter", "epoch=2", "--iter", "step=none", check=False,
        )

        # Iteration 0 is not reconstructible here: the module-scope resume block
        # already loaded end-of-run weights. Better to refuse than to report
        # fast-forwarded numbers that started from the wrong state.
        assert proc.returncode != 0
        assert "no longer reconstructible" in (proc.stderr + proc.stdout)
