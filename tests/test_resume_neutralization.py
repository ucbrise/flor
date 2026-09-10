"""Keeping a script's resume block from undermining replay from iteration 0.

A forward run that resumes from its own checkpoint loads whatever the file
holds. On replay that file holds end-of-run weights, which would land on top of
the initialization replay recomputes from. flor recognizes the block and turns
its load into a no-op; these cover the shapes it recognizes, the ones it
doesn't (and says so), and the retired API older scripts still call.
"""

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.slow


HEAD = '''
import os

import torch
import torch.nn as nn

import flordb as flor

epochs = flor.arg("epochs", 3)

torch.manual_seed(flor.arg("seed", 42))
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
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

FLAT_SAVE = '''    torch.save(model.state_dict(), "ckpt.pth")
'''

DICT_SAVE = '''    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        "ckpt.pth",
    )
'''

FLAT_RESUME_BLOCK = '''
if os.path.exists("ckpt.pth"):
    model.load_state_dict(torch.load("ckpt.pth"))
'''

FULL_RESUME_BLOCK = '''
if os.path.exists("ckpt.pth"):
    _resume = torch.load("ckpt.pth")
    model.load_state_dict(_resume["model"])
    optimizer.load_state_dict(_resume["optimizer"])
'''

# Out of recognition's reach: flor matches the load it neutralizes by the file
# it names, and a computed path only names one once the script runs.
COMPUTED_PATH_BLOCK = '''
CKPT = "ckpt" + ".pth"
if os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT))
'''

# Two checkpoint files: a ResumeSpec addresses one, so neither is taken.
SPLIT_BLOCK = '''
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth"))
    optimizer.load_state_dict(torch.load("opt.pth"))
'''

SPLIT_SAVE = '''    torch.save(model.state_dict(), "model.pth")
    torch.save(optimizer.state_dict(), "opt.pth")
'''

# Recognized, but two flat targets can't share one no-op state: whichever
# object's own state came back, the other would take it too.
TWO_TARGET_BLOCK = '''
ema = nn.Linear(2, 1)
if os.path.exists("ckpt.pth"):
    model.load_state_dict(torch.load("ckpt.pth"))
    ema.load_state_dict(torch.load("ckpt.pth"))
'''

# Written against the retired API: every call has to keep running, because
# replay executes historical versions of the script.
RETIRED = HEAD + '''
flor.set_ckpt_interval(0)
flor.restore("ckpt.pth", model=model, optimizer=optimizer)

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


def script(resume="", save=FLAT_SAVE):
    return HEAD + resume + TAIL + save


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
        "train.py", "--replay_flor", "--apply", "weight_sum", *extra, check=check,
    )


WARNING = "replay can't neutralize"


class TestRecognizedBlocks:
    @pytest.mark.parametrize("resume, save", [
        (FLAT_RESUME_BLOCK, FLAT_SAVE),
        (FULL_RESUME_BLOCK, DICT_SAVE),
    ])
    def test_block_is_neutralized(self, project, resume, save):
        project.write("train.py", script(resume=resume, save=save))
        project.run("train.py")
        forward = values(project, "forward")
        assert os.path.exists(os.path.join(project.root, "ckpt.pth"))

        proc = replay(project, "--iter", "epoch=all")

        assert WARNING not in proc.stdout + proc.stderr
        assert values(project, "replay") == pytest.approx(forward, rel=1e-6)

    def test_forward_run_still_resumes(self, project):
        project.write("train.py", script(resume=FULL_RESUME_BLOCK, save=DICT_SAVE))
        project.run("train.py")
        first = values(project, "forward")

        project.run("train.py")

        conn = project.db()
        try:
            count = conn.execute(
                "SELECT COUNT(DISTINCT value) FROM logs "
                "WHERE source = 'forward' AND value_name = 'weight_sum'"
            ).fetchone()[0]
        finally:
            conn.close()
        # The second run continued from the first one's weights, so none of
        # its values repeat the first run's.
        assert count == 2 * len(first)

    def test_a_block_it_cannot_neutralize_stops_replay(self, project):
        project.write("train.py", script(resume=TWO_TARGET_BLOCK))
        project.run("train.py")

        proc = replay(project, "--iter", "epoch=1", check=False)

        assert proc.returncode != 0
        assert "cannot replay from iteration 0" in proc.stdout + proc.stderr


class TestUnrecognizedLoadsAreAnnounced:
    def test_computed_path_warns(self, project):
        project.write("train.py", script(resume=COMPUTED_PATH_BLOCK))
        project.run("train.py")

        proc = replay(project, "--iter", "epoch=1", check=False)

        combined = proc.stdout + proc.stderr
        assert WARNING in combined
        assert "literal" in combined

    def test_two_checkpoint_files_warn(self, project):
        project.write("train.py", script(resume=SPLIT_BLOCK, save=SPLIT_SAVE))
        project.run("train.py")

        proc = replay(project, "--iter", "epoch=1", check=False)

        assert "more than one checkpoint file" in proc.stdout + proc.stderr

    def test_script_without_a_setup_load_is_not_warned(self, project):
        project.write("train.py", script())
        project.run("train.py")
        forward = values(project, "forward")

        proc = replay(project, "--iter", "epoch=2")

        assert WARNING not in proc.stdout + proc.stderr
        assert values(project, "replay") == pytest.approx({2: forward[2]}, rel=1e-6)

    def test_load_inside_the_loop_is_left_alone(self, project):
        # A cached tensor read every epoch is not a resume block: no warning,
        # and the script gets the file it names.
        source = HEAD + '''
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
''' + FLAT_SAVE
        project.write("train.py", source)
        project.run("train.py")
        forward = values(project, "forward")

        proc = replay(project, "--iter", "epoch=all")

        assert WARNING not in proc.stdout + proc.stderr
        assert values(project, "replay") == pytest.approx(forward, rel=1e-6)


class TestRetiredCalls:
    def test_forward_runs_and_says_so_once(self, project):
        project.write("train.py", RETIRED)

        proc = project.run("train.py")

        combined = proc.stdout + proc.stderr
        for name in ("set_ckpt_interval", "restore", "checkpointing"):
            assert combined.count(f"flor.{name}(...) no longer has an effect") == 1
        assert len(values(project, "forward")) == 3

    def test_only_the_torch_save_is_kept(self, project):
        import json

        project.write("train.py", RETIRED)
        project.run("train.py")

        tstamp = os.path.basename(project.run_files()[0])[: -len(".jsonl")]
        shelf = os.path.join(project.root, ".flor", "obj_store", tstamp)
        assert os.listdir(shelf) == [".latest"]
        with open(os.path.join(shelf, ".latest", "index.json")) as f:
            assert list(json.load(f)) == ["ckpt.pth"]

    def test_replay_still_works(self, project):
        project.write("train.py", RETIRED)
        project.run("train.py")
        forward = values(project, "forward")

        replay(project, "--iter", "epoch=all")

        assert values(project, "replay") == pytest.approx(forward, rel=1e-6)


def function_resume_script(*, flat=False, guarded=True, method=False):
    from textwrap import indent

    split = HEAD.index("torch.manual_seed")
    setup = HEAD[split:]
    resume = (
        'model.load_state_dict(torch.load("ckpt.pth"))\n'
        if flat else
        '_resume = torch.load("ckpt.pth")\n'
        'model.load_state_dict(_resume["model"])\n'
        'optimizer.load_state_dict(_resume["optimizer"])\n'
    )
    if guarded:
        resume = 'if os.path.exists("ckpt.pth"):\n' + indent(resume, "    ")
    setup += resume + "return model, optimizer\n"
    training = TAIL + (FLAT_SAVE if flat else DICT_SAVE)
    if not flat:
        # Momentum has independent history; checking it catches a no-op that
        # covers the model but lets the optimizer load end-of-run state.
        training = training.replace(
            'flor.log("weight_sum", float(sum(p.sum() for p in model.parameters())))',
            'flor.log("weight_sum", float(sum(p.sum() for p in model.parameters()))'
            ' + sum(float(s["momentum_buffer"].sum()) for s in optimizer.state.values()))',
        )
    # The loop has different parameter names from the resume block, and no
    # global model/optimizer exists for a name-based fallback to find.
    training = training.replace("model.", "trained.").replace("model(", "trained(")
    training = training.replace("optimizer.", "optim.")
    prep = "def prep():\n" + indent(setup, "    ")
    if method:
        prep = "class Setup:\n" + indent("@staticmethod\n" + prep, "    ")
    return (
        HEAD[:split] + prep
        + "\ndef train(trained, optim):\n" + indent(training, "    ")
        + "\ndef main():\n"
        + ("    net, opt = Setup.prep()\n" if method else "    net, opt = prep()\n")
        + "    train(net, opt)\n\nmain()\n"
    )


class TestFunctionResume:
    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("file_present", [True, False])
    def test_prep_block_is_neutralized(self, project, flat, file_present):
        project.write("train.py", function_resume_script(flat=flat))
        project.run("train.py")
        forward = values(project, "forward")
        if not file_present:
            os.unlink(os.path.join(project.root, "ckpt.pth"))

        replay(project, "--iter", "epoch=0,1,2")

        assert values(project, "replay") == pytest.approx(forward, rel=1e-6)

    def test_forward_prep_still_loads_the_users_checkpoint(self, project):
        project.write("train.py", function_resume_script())
        project.run("train.py")
        first = values(project, "forward")

        project.run("train.py")

        assert values(project, "forward")[0] != pytest.approx(first[0])

    def test_unguarded_prep_replays_without_the_original_file(self, project):
        project.write("train.py", function_resume_script())
        project.run("train.py")
        forward = values(project, "forward")
        os.unlink(os.path.join(project.root, "ckpt.pth"))
        project.write("train.py", function_resume_script(guarded=False))

        replay(project, "--iter", "epoch=all")

        assert values(project, "replay") == pytest.approx(forward, rel=1e-6)

    def test_resume_in_a_method_binds_its_local_objects(self, project):
        project.write("train.py", function_resume_script(method=True))
        project.run("train.py")
        forward = values(project, "forward")

        replay(project, "--iter", "epoch=all")

        assert values(project, "replay") == pytest.approx(forward, rel=1e-6)
