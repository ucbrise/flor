"""Declaring restore semantics, and refusing to guess when they're missing.

The forward half of the torch.save piggy-back needs no instrumentation: flor
mirrors whatever the script saves. The return half needs to know which object
each mirror belongs in, and flor only *infers* that, by pattern-matching one
narrow shape in the script's source. These cover what happens when the
inference doesn't fit -- the declaration that replaces it (`flor.restore`), and
the failures that used to pass silently and now don't.
"""

import os

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.slow


HEAD = '''
import os

import torch
import torch.nn as nn

import flordb as flor

flor.set_ckpt_interval(flor.arg("ckpt_interval_s", 0.0))
epochs = flor.arg("epochs", 3)

torch.manual_seed(flor.arg("seed", 42))
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
'''

TAIL = '''
x = torch.ones(4, 2)
y = torch.zeros(4, 1)

for epoch in flor.loop("epoch", range(epochs)):
    for step in flor.loop("step", range(2)):
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    flor.log("weight_sum", float(sum(p.sum() for p in model.parameters())))
'''

# The single most common PyTorch idiom, and one ResumeBlockVisitor cannot see:
# the save is a bare state_dict (no keys to map) and the load is a one-line
# Expr, not the Assign + Subscript pair the visitor matches on.
FLAT_SAVE = '''    torch.save(model.state_dict(), "ckpt.pth")
'''

FLAT_RESUME_BLOCK = '''
if os.path.exists("ckpt.pth"):
    model.load_state_dict(torch.load("ckpt.pth"))
'''

FLAT_RESTORE_CALL = '''
flor.restore("ckpt.pth", model)
'''

# Still out of inference's reach: flor has to name the mirror file before the
# script runs, and a computed path is only known once it does.
COMPUTED_PATH_BLOCK = '''
CKPT = "ckpt" + ".pth"
if os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT))
'''

COMPUTED_PATH_RESTORE = '''
CKPT = "ckpt" + ".pth"
flor.restore(CKPT, model)
'''

# Two checkpoint files: inferable individually, refused together, because a
# ResumeSpec addresses one file per run.
SPLIT_BLOCK = '''
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth"))
    optimizer.load_state_dict(torch.load("opt.pth"))
'''

SPLIT_SAVE = '''    torch.save(model.state_dict(), "model.pth")
    torch.save(optimizer.state_dict(), "opt.pth")
'''

# Dict save, but the inferred block reaches for a key the save never wrote.
# Forward runs fine (the block is guarded and the file is absent on run 1);
# only replay touches the bad key.
PARTIAL_SAVE = '''    torch.save({"model": model.state_dict()}, "ckpt.pth")
'''

FULL_RESUME_BLOCK = '''
if os.path.exists("ckpt.pth"):
    _resume = torch.load("ckpt.pth")
    model.load_state_dict(_resume["model"])
    optimizer.load_state_dict(_resume["optimizer"])
'''

KEYED_SAVE = '''    torch.save(
        {"net": model.state_dict(), "opt": optimizer.state_dict()},
        "ckpt.pth",
    )
'''

KEYED_RESTORE_CALL = '''
flor.restore("ckpt.pth", net=model, opt=optimizer)
'''

ITERATION_TAIL = '''
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
'''


DICT_SAVE = '''    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        "ckpt.pth",
    )
'''

# Declares the mapping *and* reports which spec ends up in force, so a test can
# tell the declaration from the inference that would otherwise look identical.
RESTORE_AND_REPORT = '''
flor.restore("ckpt.pth", model=model, optimizer=optimizer)
print("spec-source:", flor.cli.flags.resume_spec.source)
'''

ENROLLED = HEAD + '''
x = torch.ones(4, 2)
y = torch.zeros(4, 1)

with flor.checkpointing(model=model, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(epochs)):
        for step in flor.loop("step", range(2)):
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        flor.log("weight_sum", float(sum(p.sum() for p in model.parameters())))
        torch.save(model.state_dict(), "ckpt.pth")
'''

# `width` is a flor.arg, so replay restores its historical value -- the only way
# to change the model between forward and replay is to change the structure the
# script builds around it, which is exactly the edit-and-forget case.
WIDTH_TRAIN = '''
import torch
import torch.nn as nn

import flordb as flor

flor.set_ckpt_interval(flor.arg("ckpt_interval_s", 0.0))
epochs = flor.arg("epochs", 3)
width = flor.arg("width", 4)

torch.manual_seed(flor.arg("seed", 42))
model = nn.Linear(2, {width})
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

flor.restore("ckpt.pth", model)

x = torch.ones(4, 2)
y = torch.zeros_like(model(x))

for epoch in flor.loop("epoch", range(epochs)):
    for step in flor.loop("step", range(2)):
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    flor.log("weight_sum", float(sum(p.sum() for p in model.parameters())))
    torch.save(model.state_dict(), "ckpt.pth")
'''


def script(resume="", tail=TAIL, save=FLAT_SAVE):
    return HEAD + resume + tail + save


def values(project, source):
    import json

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


def replay(project, *extra, check=True):
    return project.run(
        "train.py",
        "--replay_flor",
        "--apply",
        "weight_sum",
        "--iter",
        "step=none",
        *extra,
        check=check,
    )


class TestUndeclaredRestoreIsAnnounced:
    def test_unmatched_resume_idiom_warns_on_replay(self, project):
        # Neither inference nor declaration covers this script, so replay can
        # restore nothing. The run still happens -- it just has to say so.
        project.write("train.py", script(resume=COMPUTED_PATH_BLOCK))
        project.run("train.py")

        proc = replay(project, "--iter", "epoch=1", check=False)

        combined = proc.stdout + proc.stderr
        assert "declares no way to load it back" in combined
        assert "flor.restore" in combined

    def test_enrolled_script_is_not_warned(self, project):
        # flor.checkpointing says how to restore; there is nothing to warn about.
        project.write("train.py", ENROLLED)
        project.run("train.py")

        proc = replay(project, "--iter", "epoch=1", check=False)

        assert "declares no way to load it back" not in proc.stdout + proc.stderr

    def test_declared_script_is_not_warned(self, project):
        project.write("train.py", script(resume=FLAT_RESTORE_CALL))
        project.run("train.py")

        proc = replay(project, "--iter", "epoch=1", check=False)

        assert proc.returncode == 0, proc.stderr
        assert "declares no way to load it back" not in proc.stdout + proc.stderr

    def test_script_without_torch_save_is_not_warned(self, project):
        # No piggy-back checkpoints at all -- nothing to restore, nothing to say.
        project.write("train.py", HEAD + TAIL)
        project.run("train.py")

        proc = replay(project, "--iter", "epoch=1", check=False)

        assert "declares no way to load it back" not in proc.stdout + proc.stderr


class TestInferredFlatIdiom:
    """`torch.save(model.state_dict(), p)` + `model.load_state_dict(torch.load(p))`.

    The most ordinary checkpoint code in PyTorch, and for a long time the shape
    flor could say least about. It needs no declaration now.
    """

    def test_flat_idiom_replays_faithfully_with_no_declaration(self, project):
        project.write("train.py", script(resume=FLAT_RESUME_BLOCK))
        project.run("train.py")
        forward = values(project, "forward")

        proc = replay(project, "--iter", "epoch=all", check=False)

        assert proc.returncode == 0, proc.stderr
        assert "declares no way to load it back" not in proc.stdout + proc.stderr
        replayed = values(project, "replay")
        assert set(replayed) == set(forward)
        # step=none means no training runs, so each epoch's number can only be
        # right if that epoch's mirror was found and applied.
        for epoch, value in forward.items():
            assert replayed[epoch] == pytest.approx(value, rel=1e-6)

    def test_flat_idiom_survives_a_fresh_clone(self, project):
        # No mirrors and no ckpt.pth, so replay has to rebuild from iteration 0
        # -- which is only sound if the module-scope one-liner was neutralized
        # into a no-op rather than left to load end-of-run weights.
        import shutil

        project.write("train.py", script(resume=FLAT_RESUME_BLOCK))
        project.run("train.py")
        forward = values(project, "forward")

        tstamp = os.path.basename(project.run_files()[0])[: -len(".jsonl")]
        shutil.rmtree(
            os.path.join(project.root, ".flor", "obj_store", tstamp)
        )

        proc = replay(project, "--iter", "epoch=2", check=False)

        assert proc.returncode == 0, proc.stderr
        replayed = values(project, "replay")
        assert replayed[2] == pytest.approx(forward[2], rel=1e-6)

    def test_two_checkpoint_files_refuse_inference(self, project):
        # Inferable one file at a time, ambiguous together. Restoring only the
        # model and leaving the optimizer on another iteration's momentum is
        # exactly the half-right replay this refuses to produce.
        project.write(
            "train.py", script(resume=SPLIT_BLOCK, save=SPLIT_SAVE)
        )
        project.run("train.py")

        proc = replay(project, "--iter", "epoch=1", check=False)

        combined = proc.stdout + proc.stderr
        assert "more than one checkpoint file" in combined
        assert "flor.restore" in combined


class TestExplicitRestore:
    def test_flat_state_dict_replays_faithfully(self, project):
        # One positional target: the whole file is that object's state_dict.
        project.write("train.py", script(resume=FLAT_RESTORE_CALL))
        project.run("train.py")
        forward = values(project, "forward")

        replay(project, "--iter", "epoch=all")

        replayed = values(project, "replay")
        assert set(replayed) == set(forward)
        # step=none means no training runs, so each epoch's number can only be
        # right if that epoch's mirror was found and applied.
        for epoch, value in forward.items():
            assert replayed[epoch] == pytest.approx(value, rel=1e-6)

    def test_keyword_targets_map_to_saved_keys(self, project):
        # Keys deliberately unlike the variable names, so a match can only come
        # from the declaration and not from a lucky coincidence.
        project.write(
            "train.py", script(resume=KEYED_RESTORE_CALL, save=KEYED_SAVE)
        )
        project.run("train.py")
        forward = values(project, "forward")

        replay(project, "--iter", "epoch=all")

        replayed = values(project, "replay")
        for epoch, value in forward.items():
            assert replayed[epoch] == pytest.approx(value, rel=1e-6)

    def test_declaration_overrides_inference(self, project):
        # A script carrying both an inferable resume block and a declaration.
        # The declaration has to be the spec replay uses -- being able to
        # correct a bad guess is the entire point -- and the run must stay
        # faithful under it.
        both = HEAD + FULL_RESUME_BLOCK + RESTORE_AND_REPORT + TAIL + DICT_SAVE
        project.write("train.py", both)
        project.run("train.py")
        forward = values(project, "forward")

        proc = replay(project, "--iter", "epoch=all", check=False)

        assert proc.returncode == 0, proc.stderr
        assert "spec-source: explicit" in proc.stdout
        replayed = values(project, "replay")
        for epoch, value in forward.items():
            assert replayed[epoch] == pytest.approx(value, rel=1e-6)

    def test_forward_run_resumes_from_the_users_file(self, project):
        # flor.restore stands in for the resume block it replaces, so an
        # interrupted forward run still picks up where it left off.
        project.write("train.py", script(resume=FLAT_RESTORE_CALL))
        project.run("train.py", "--kwargs", "epochs=2")
        first = values(project, "forward")

        project.run("train.py", "--kwargs", "epochs=2")

        conn = project.db()
        try:
            rows = conn.execute(
                "SELECT DISTINCT value FROM logs "
                "WHERE source = 'forward' AND value_name = 'weight_sum'"
            ).fetchall()
        finally:
            conn.close()
        # A second run that resumed continues descending rather than retracing
        # the first run's numbers.
        assert len(rows) > len(first)


class TestRestoreArgumentValidation:
    """Pure argument checks -- they run before flor.restore touches anything."""

    @staticmethod
    def _restore(*args, **kwargs):
        from flordb import api

        return api.restore(*args, **kwargs)

    def test_mixing_positional_and_keyword_is_rejected(self):
        import torch.nn as nn

        with pytest.raises(TypeError, match="not both"):
            self._restore("ckpt.pth", nn.Linear(1, 1), other=nn.Linear(1, 1))

    def test_multiple_positional_targets_are_rejected(self):
        import torch.nn as nn

        with pytest.raises(TypeError, match="at most one positional"):
            self._restore("ckpt.pth", nn.Linear(1, 1), nn.Linear(1, 1))

    def test_no_targets_is_rejected(self):
        with pytest.raises(TypeError, match="at least one target"):
            self._restore("ckpt.pth")

    def test_target_without_load_state_dict_is_rejected(self):
        with pytest.raises(TypeError, match="no load_state_dict"):
            self._restore("ckpt.pth", model={"weights": 1})


class TestReplayRefusesToGuess:
    def test_missing_mirror_does_not_fall_back_to_end_of_run_state(self, project):
        # flor.iteration has no replay planner, so it cannot route around a
        # missing mirror the way flor.loop's logical replay does. Throttling
        # leaves only epoch 0 shelved; asking for epoch 1 has to fail rather
        # than silently load ckpt.pth, which holds end-of-run weights.
        project.write(
            "train.py", script(resume=FLAT_RESTORE_CALL, tail=ITERATION_TAIL)
        )
        project.run("train.py", "--kwargs", "ckpt_interval_s=3600", "epochs=3")

        proc = replay(project, "--iter", "epoch=1", check=False)

        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "no checkpoint mirror" in combined
        assert "end-of-run state" in combined

    def test_state_that_no_longer_fits_its_target_fails_loudly(self, project):
        # The model changed shape between the forward run and the replay -- the
        # ordinary "I edited the script and forgot" case. The mirror's tensors
        # no longer fit, so load_state_dict raises. That used to be swallowed
        # per-target: the object silently kept whatever state it had, and the
        # metrics computed off it were logged as historical.
        #
        # flor.restore does not load at module scope on replay, so nothing
        # catches this earlier -- this is the per-iteration restore's own
        # failure to report.
        project.write("train.py", WIDTH_TRAIN.format(width="width"))
        project.run("train.py")

        project.write("train.py", WIDTH_TRAIN.format(width="width + 4"))
        proc = replay(project, "--iter", "epoch=1", check=False)

        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "failed for 1 of 1 target(s)" in combined
        assert "size mismatch" in combined or "shape" in combined

    def test_unrelated_torch_load_is_left_alone(self, project):
        # The strictness must key on "this file has siblings on the shelf", not
        # on "a torch.load happened during replay" -- otherwise every cached
        # tensor a script loads inside its loop becomes a replay failure.
        source = (
            HEAD
            + FLAT_RESTORE_CALL
            + '''
torch.save(torch.ones(3), "data.pt")
x = torch.ones(4, 2)
y = torch.zeros(4, 1)

for epoch in flor.loop("epoch", range(epochs)):
    blob = torch.load("data.pt")
    for step in flor.loop("step", range(2)):
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    flor.log("weight_sum", float(sum(p.sum() for p in model.parameters())) + float(blob.sum()) * 0)
'''
            + FLAT_SAVE
        )
        project.write("train.py", source)
        project.run("train.py")
        forward = values(project, "forward")

        proc = replay(project, "--iter", "epoch=all", check=False)

        assert proc.returncode == 0, proc.stderr
        replayed = values(project, "replay")
        for epoch, value in forward.items():
            assert replayed[epoch] == pytest.approx(value, rel=1e-6)


class TestUnmatchedIdiomIsReplayableOnceDeclared:
    def test_restore_makes_the_flat_idiom_faithful(self, project):
        """The end-to-end shape of the workflow this is all for.

        Forward stays as the developer wrote it -- a bare torch.save and a
        one-line resume, neither of which flor can interpret. Adding one
        declaration before replay is what makes the run reconstructible, and
        replay re-parses the script from disk, so no re-run is needed.
        """
        project.write("train.py", script(resume=COMPUTED_PATH_BLOCK))
        project.run("train.py")
        forward = values(project, "forward")

        # The developer reads the warning and declares the mapping. The
        # checkpoint file on disk is now flor.restore's business, so the
        # unmatched block goes with it.
        project.write("train.py", script(resume=COMPUTED_PATH_RESTORE))

        proc = replay(project, "--iter", "epoch=all", check=False)

        assert proc.returncode == 0, proc.stderr
        replayed = values(project, "replay")
        assert set(replayed) == set(forward)
        for epoch, value in forward.items():
            assert replayed[epoch] == pytest.approx(value, rel=1e-6)
