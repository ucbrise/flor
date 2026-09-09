import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Optional

from .constants import *
from .clock import Clock
from . import utils
from . import cli

import cloudpickle


def _torch():
    """torch, or None when it isn't installed."""
    try:
        import torch

        return torch
    except ImportError:
        return None


def _has_state_dict_protocol(obj) -> bool:
    """Whether `obj` carries its state the way accelerator frameworks do.

    torch.nn.Module and torch.optim.Optimizer are the obvious members, but a
    real training loop keeps objects beside them that are neither -- an LR
    scheduler, an amp.GradScaler -- and those hold state the run cannot be
    restored without. Matching the protocol rather than the two base classes is
    what lets them round-trip, and is where a framework that adopts the same
    pair plugs in later.
    """
    return callable(getattr(obj, "state_dict", None)) and callable(
        getattr(obj, "load_state_dict", None)
    )


def _is_ndarray(obj) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(obj, np.ndarray)


def _is_dataframe(obj) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(obj, pd.DataFrame)


# Backend -> the extension its mirrors carry. `deserialize` searches these in
# the same order, so a name that somehow has two mirrors resolves the way it
# was written.
BACKENDS = (
    ("state_dict", ".pth"),
    ("numpy", ".npy"),
    ("pandas", ".parquet"),
    ("pickle", ".pkl"),
)

_EXT_FOR = dict(BACKENDS)


def _select_backend(obj) -> str:
    """Which backend round-trips `obj`, decided by inspection.

    Dispatch is by predicate rather than by walking a chain of try/except: a
    backend that matches and then *fails* -- a full disk, an OOM partway
    through torch.save -- has to raise, not fall silently through to cloudpickle
    and shelve a file under an extension the restore path won't look for.
    """
    if _has_state_dict_protocol(obj) and _torch() is not None:
        return "state_dict"
    if _is_ndarray(obj):
        return "numpy"
    if _is_dataframe(obj):
        return "pandas"
    return "pickle"


def serialize(layers, name, obj):
    backend = _select_backend(obj)
    path = get_shelf() / utils.to_filename(layers, name, _EXT_FOR[backend])
    if backend == "state_dict":
        import torch

        torch.save(obj.state_dict(), path)
    elif backend == "numpy":
        import numpy as np

        np.save(path, obj)
    elif backend == "pandas":
        obj.to_parquet(path)
    else:
        with open(path, "wb") as f:
            cloudpickle.dump(obj, f)
    return path.name


def _restore_pickled(name, obj, loaded, path) -> None:
    """Put a cloudpickled snapshot back into the live object.

    `deserialize` is handed the object to restore *into*, not the name it is
    bound to, so it cannot rebind -- the state has to move across in place. A
    mapping takes it through clear/update; anything else through its instance
    dict, which is what cloudpickle captured to begin with.
    """
    if isinstance(obj, MutableMapping):
        obj.clear()
        obj.update(loaded)
        return
    obj_dict = getattr(obj, "__dict__", None)
    loaded_dict = getattr(loaded, "__dict__", None)
    if obj_dict is not None and loaded_dict is not None:
        # Clear first: an attribute the object grew after the snapshot was
        # taken is not part of the state being replayed.
        obj_dict.clear()
        obj_dict.update(loaded_dict)
        return
    raise RuntimeError(
        f"FLOR: cannot restore {name!r} from {path.name}: a "
        f"{type(obj).__name__} is neither a mapping nor an object with a "
        f"__dict__, so flor has no way to move the snapshot back into it in "
        f"place. Give it state_dict()/load_state_dict()."
    )


def unrestorable_reason(obj) -> Optional[str]:
    """Why `obj` could be shelved but never put back, or None if it round-trips.

    flor.checkpointing runs every enrollment through this. Serializing is the
    easy half -- cloudpickle takes almost anything -- and an object that only
    fails coming back costs a whole forward run of disk before saying so, on
    the replay that needed it. Refusing at enrollment moves that to the first
    second of the run that would have wasted the work.
    """
    backend = _select_backend(obj)
    if backend != "pickle":
        return None
    if isinstance(obj, MutableMapping):
        return None
    if getattr(obj, "__dict__", None) is not None:
        return None
    return (
        f"a {type(obj).__name__} is neither a mapping nor an object with a "
        f"__dict__ (__slots__?), so flor can shelve it but has no way to put "
        f"the snapshot back in place"
    )


def deserialize(layers, name, obj, missing_ok: bool = False):
    if (path := get_shelf() / utils.to_filename(layers, name, ".pth")).exists():
        import torch

        # map_location="cpu" so a mirror written on a GPU box restores on one
        # without, and vice versa: load_state_dict copies into the live
        # parameters, which already sit on whatever device this run put them.
        obj.load_state_dict(torch.load(path, map_location="cpu"))
    elif (path := get_shelf() / utils.to_filename(layers, name, ".npy")).exists():
        import numpy

        obj[:] = numpy.load(path)
    elif (path := get_shelf() / utils.to_filename(layers, name, ".parquet")).exists():
        import pandas as pd

        obj.iloc[:, :] = pd.read_parquet(path)
    elif (path := get_shelf() / utils.to_filename(layers, name, ".pkl")).exists():
        with open(path, "rb") as f:
            loaded_obj = cloudpickle.load(f)
        _restore_pickled(name, obj, loaded_obj, path)
    else:
        # Reached during replay when the requested iteration has no checkpoint
        # in the object store -- usually because the adaptive throttle skipped
        # it. Failing loudly beats replaying from an uninitialized object and
        # reporting the resulting numbers as historical fact.
        #
        # missing_ok is the fast-forward case, where the caller has already
        # restored an earlier iteration and is recomputing its way forward: a
        # gap in the shelf is the expected condition there, not a failure.
        if missing_ok:
            return False
        stem = utils.to_filename(layers, name, "").stem
        raise RuntimeError(
            f"FLOR: no checkpoint for {name!r} at this iteration. Looked for "
            f"{stem}.{{pth,npy,parquet,pkl}} in {get_shelf()}. The run being "
            f"replayed likely throttled this iteration's checkpoint "
            f"(flor.set_ckpt_interval); narrow to an iteration that has one, or "
            f"re-run forward with a smaller interval."
        )


# The extension set `deserialize` searches, in the same order. Kept beside it so
# `has_shelved` means exactly "deserialize would find something here".
SHELF_EXTENSIONS = tuple(ext for _, ext in BACKENDS)


def has_shelved(layers, name) -> bool:
    """True when some backend's mirror for (layers, name) is already shelved."""
    shelf = get_shelf()
    return any(
        (shelf / utils.to_filename(layers, name, ext)).exists()
        for ext in SHELF_EXTENSIONS
    )


def get_shelf():
    if not cli.in_replay_mode():
        tstamp = Clock.get_datetime()
    else:
        assert cli.flags.old_tstamp is not None
        tstamp = cli.flags.old_tstamp

    SHELF = Path(OBJSTORE_DIR) / tstamp
    os.makedirs(SHELF, exist_ok=True)
    return SHELF
