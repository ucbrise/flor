"""Notebook reads must select the requested run without starting another run."""

import json
from pathlib import Path

import pandas as pd
import pytest

import flordb as flor
from flordb import checkpoint_io, obj_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(obj_store, "OBJSTORE_DIR", str(tmp_path))
    return tmp_path


def test_missing_run_is_read_only(store):
    stamp = pd.Timestamp("2026-01-02T03:04:05.000000")
    assert flor.checkpoints(stamp).empty
    with pytest.raises(FileNotFoundError, match="do not travel with git"):
        flor.load_checkpoint(stamp)
    assert list(store.iterdir()) == []
    with pytest.raises(ValueError):
        flor.checkpoints("../../outside")


def test_legacy_snapshots_require_explicit_selection(store):
    import cloudpickle

    stamp = "2026-01-02T03:04:05.000000"
    shelf = store / stamp
    shelf.mkdir()
    (shelf / "state_epoch_10.pkl").write_bytes(cloudpickle.dumps({"weight": 10}))
    (shelf / "state_epoch_2.pkl").write_bytes(cloudpickle.dumps({"weight": 2}))
    with pytest.raises(FileNotFoundError):
        flor.load_checkpoint(pd.Timestamp(stamp))
    assert flor.load_checkpoint(stamp, "state_epoch_10.pkl") == {"weight": 10}
    assert set(flor.checkpoints(stamp).kind) == {"iteration"}


def test_run_copies_and_ambiguous_default(store, monkeypatch):
    torch = pytest.importorskip("torch")

    stamp = "2026-01-02T03:04:05.000000"
    shelf = store / stamp
    shelf.mkdir()
    monkeypatch.setattr(obj_store, "get_shelf", lambda: shelf)

    def saving(weight):
        # torch.serialization.save, not torch.save: other tests in this process
        # may have installed flor's torch.save hook, which would record a run.
        return lambda path: torch.serialization.save({"weight": weight}, path)

    checkpoint_io._save("ckpt.pt", saving(1))
    assert flor.load_checkpoint(stamp) == {"weight": 1}
    checkpoint_io._save("ckpt.pt", saving(2))
    checkpoint_io._save("other.pt", saving(3))
    assert flor.load_checkpoint(stamp, "ckpt.pt") == {"weight": 2}
    assert set(flor.checkpoints(stamp).kind) == {"run"}
    with pytest.raises(ValueError, match="explicit name"):
        flor.load_checkpoint(stamp)


def test_failed_copy_preserves_previous_state(store, monkeypatch):
    stamp = "2026-01-02T03:04:05.000000"
    shelf = store / stamp
    shelf.mkdir()
    monkeypatch.setattr(obj_store, "get_shelf", lambda: shelf)
    checkpoint_io._save("ckpt.pt", lambda path: path.write_bytes(b"first"))
    before = {path.name: path.read_bytes() for path in (shelf / ".latest").iterdir()}

    def fail(path):
        path.write_bytes(b"partial checkpoint")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        checkpoint_io._save("ckpt.pt", fail)
    assert {path.name: path.read_bytes() for path in (shelf / ".latest").iterdir()} == before


@pytest.mark.slow
def test_dataframe_loads_latest_weights_for_each_run_and_replay_preserves_them(project):
    pytest.importorskip("torch")
    project.write("train.py", '''
import torch
import flordb as flor
value = flor.arg("gain", 1)
model = torch.nn.Linear(1, 1, bias=False)
for epoch in flor.loop("epoch", range(3)):
    with torch.no_grad():
        model.weight.fill_(value + epoch)
    flor.log("weight", model.weight.item())
    torch.save({"model": model.state_dict()}, "ckpt.pt")
# Saves outside the loop must also replace this run's copy.
with torch.no_grad():
    model.weight.fill_(value + 10)
torch.save({"model": model.state_dict()}, "ckpt.pt")
''')
    project.run("train.py", "--kwargs", "gain=1")
    project.run("train.py", "--kwargs", "gain=7")
    before_runs = project.run_files()
    latest = sorted((Path(project.root) / ".flor" / "obj_store").glob("*/.latest/*"))
    before = {str(path): path.read_bytes() for path in latest}
    result = project.run("-c", '''
import json
import torch
import flordb as flor
results = []
for row in flor.dataframe("gain").drop_duplicates("tstamp").itertuples():
    state = flor.load_checkpoint(row.tstamp, "ckpt.pt")
    model = torch.nn.Linear(1, 1, bias=False)
    model.load_state_dict(state["model"])
    assert model.weight.device.type == "cpu"
    model(torch.ones(1, 1)).square().sum().backward()
    results.append([int(row.gain), model.weight.item(), model.weight.grad.item()])
print(json.dumps(sorted(results)))
''')
    assert json.loads(result.stdout) == [[1, 11, 22], [7, 17, 34]]
    assert project.run_files() == before_runs
    project.run(
        "train.py", "--replay_flor", "--apply", "weight",
        "--iter", "epoch=0",
    )
    assert {str(path): path.read_bytes() for path in latest} == before


@pytest.mark.slow
def test_each_saved_path_keeps_its_own_copy(project):
    pytest.importorskip("torch")
    project.write("train.py", '''
from pathlib import Path
import torch
import flordb as flor
Path("first").mkdir()
Path("second").mkdir()
for epoch in flor.loop("epoch", range(3)):
    torch.save({"weight": torch.tensor(epoch)}, Path("first/model.pt"))
    torch.save({"weight": torch.tensor(epoch + 10)}, "second/model.pt")
flor.log("done", True)
''')
    project.run("train.py")
    project.run("-c", '''
import flordb as flor
stamp = flor.dataframe().iloc[0].tstamp
available = flor.checkpoints(stamp)
assert set(available.loc[available.kind == "run", "name"]) == {
    "first/model.pt", "second/model.pt",
}
assert set(available.kind) == {"run"}
assert flor.load_checkpoint(stamp, "first/model.pt")["weight"].item() == 2
assert flor.load_checkpoint(stamp, "second/model.pt")["weight"].item() == 12
''')
