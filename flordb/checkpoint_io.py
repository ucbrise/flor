"""Run checkpoint capture and read-only access from notebooks."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from .constants import CURRDIR
from . import obj_store


def _atomic_write(path, write):
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    temporary = Path(temporary)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _project_relative(path) -> str:
    """`path` relative to the project root, the way run copies record it."""
    return os.path.relpath(os.path.abspath(os.fsdecode(path)), CURRDIR)


def _identity(path) -> dict:
    """What a copy records about the file it copied: where, how big, when written."""
    stat = os.stat(path)
    return {
        "path": _project_relative(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _save(name, write, source=None):
    """Replace this run's copy of the file the script saved at `name`.

    `source` is the file being copied; its identity is kept with the copy so a
    later run that loads the same file can tell which run wrote it.
    """
    directory = obj_store.get_shelf() / ".latest"
    directory.mkdir(exist_ok=True)
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    filename = hashlib.sha256(name.encode()).hexdigest() + ".pth"
    _atomic_write(directory / filename, write)
    entry = {"backend": "state_dict", "file": filename}
    if source is not None:
        entry["source"] = _identity(source)
    index[name] = entry
    _atomic_write(index_path, lambda path: path.write_text(json.dumps(index)))


def _writer_of(path, exclude=None):
    """The (tstamp, name) of the latest run whose copy is this exact file.

    Matched on the metadata each copy recorded when its run saved the file --
    project-relative path, size, and modification time -- so a file edited or
    replaced since then, or one no flor run wrote, matches nothing.
    """
    try:
        identity = _identity(path)
    except OSError:
        return None
    found = None
    for index_path in Path(obj_store.OBJSTORE_DIR).glob("*/.latest/index.json"):
        tstamp = index_path.parent.parent.name
        if tstamp == exclude or (found is not None and tstamp <= found[0]):
            continue
        for name, entry in json.loads(index_path.read_text()).items():
            if entry.get("source") == identity:
                found = (tstamp, name)
                break
    return found


def _stamp(tstamp) -> str:
    """A run tstamp as the object store names it.

    dataframe() supplies pandas.Timestamp; run records use microsecond ISO
    strings. Parsing also prevents a timestamp from escaping the object store.
    """
    stamp = pd.Timestamp(tstamp)
    if pd.isna(stamp):
        raise ValueError("A run tstamp is required.")
    return stamp.isoformat(timespec="microseconds")


def _shelf(tstamp):
    return Path(obj_store.OBJSTORE_DIR) / _stamp(tstamp)


def checkpoints(tstamp):
    """List saved state for one dataframe tstamp, without loading any objects.

    `kind == 'run'` is the run's copy of a file it saved with torch.save, named
    by the path it was saved to. `kind == 'iteration'` lists per-iteration
    snapshots by exact filename; only runs recorded before FlorDB kept one copy
    per run have them. Missing local checkpoints produce an empty dataframe.
    """
    shelf = _shelf(tstamp)
    records = []
    index_path = shelf / ".latest" / "index.json"
    if index_path.exists():
        for name, entry in json.loads(index_path.read_text()).items():
            records.append({
                "name": name, "kind": "run", "backend": entry["backend"],
                "path": str(index_path.parent / entry["file"]),
            })
    extensions = {ext: backend for backend, ext in obj_store.BACKENDS}
    extensions[".pt"] = "state_dict"
    for path in sorted(shelf.glob("*")):
        if path.is_file() and path.suffix in extensions:
            records.append({
                "name": path.name, "kind": "iteration",
                "backend": extensions[path.suffix], "path": str(path),
            })
    return pd.DataFrame(records, columns=["name", "kind", "backend", "path"])


def _select(tstamp, name=None) -> dict:
    """The one checkpoints() row `name` picks out for this run, as a dict."""
    available = checkpoints(tstamp)
    selected = available[
        available["kind"].eq("run") if name is None else available["name"].eq(name)
    ]
    if selected.empty:
        raise FileNotFoundError(
            f"FLOR: no {'run checkpoint' if name is None else repr(name)} for {tstamp}. "
            "Use flor.checkpoints(tstamp) to inspect local snapshots. Checkpoints "
            "do not travel with git; copy the run's object-store directory from "
            "the training machine if needed."
        )
    if len(selected) != 1:
        raise ValueError(
            f"FLOR: multiple checkpoints for {tstamp}: {selected['name'].tolist()}. "
            "Pass an explicit name from flor.checkpoints(tstamp)."
        )
    return selected.iloc[0].to_dict()


def _read(entry: dict, map_location="cpu", weights_only=True):
    path = Path(entry["path"])
    if entry["backend"] == "state_dict":
        import torch

        # torch.serialization.load rather than torch.load, which flor's replay
        # hook may have replaced: this is flor reading its own copy.
        return torch.serialization.load(
            path, map_location=map_location, weights_only=weights_only
        )
    if entry["backend"] == "numpy":
        import numpy as np

        return np.load(path)
    if entry["backend"] == "pandas":
        return pd.read_parquet(path)
    import cloudpickle

    with path.open("rb") as stream:
        return cloudpickle.load(stream)


def load_checkpoint(tstamp, name=None, *, map_location="cpu", weights_only=True):
    """Read one run's saved state. flor.load_checkpoint wraps this; see there."""
    return _read(_select(tstamp, name), map_location, weights_only)
