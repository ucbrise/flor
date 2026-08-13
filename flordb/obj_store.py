import os
from pathlib import Path

from .constants import *
from .clock import Clock
from . import utils
from . import cli

import cloudpickle


def serialize_torch(layers, name, obj):
    import torch
    import torch.nn
    import torch.optim

    if isinstance(obj, torch.nn.Module) or isinstance(obj, torch.optim.Optimizer):
        path = get_shelf() / utils.to_filename(layers, name, ".pth")
        torch.save(obj.state_dict(), path)
        return path.name
    else:
        # `serialize` walks the backends in order and catches these to try the
        # next one, so the exception is control flow, not a user-facing error.
        raise TypeError(f"{name!r} is not a torch Module or Optimizer")


def serialize_numpy(layers, name, obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        path = get_shelf() / utils.to_filename(layers, name, ".npy")
        np.save(path, obj)
        return path.name
    else:
        raise TypeError(f"{name!r} is not a numpy ndarray")


def serialize_scikit(layers, name, obj):
    import pickle  # Scikit-Learn
    import sklearn.base

    sklearn_base_classes = (
        sklearn.base.BaseEstimator,
        sklearn.base.ClassifierMixin,
        sklearn.base.RegressorMixin,
        sklearn.base.ClusterMixin,
        sklearn.base.TransformerMixin,
    )
    if isinstance(obj, sklearn_base_classes) or hasattr(obj, "fit"):
        path = get_shelf() / utils.to_filename(layers, name, ".pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return path.name
    else:
        raise TypeError(f"{name!r} is not a scikit-learn estimator")


def serialize_pandas(layers, name, obj):
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        path = get_shelf() / utils.to_filename(layers, name, ".parquet")
        obj.to_parquet(path)
        return path.name
    else:
        raise TypeError(f"{name!r} is not a pandas DataFrame")


def serialize(layers, name, obj):
    try:
        return serialize_torch(layers, name, obj)
    except:
        pass

    try:
        return serialize_scikit(layers, name, obj)
    except:
        pass

    try:
        return serialize_numpy(layers, name, obj)
    except:
        pass

    try:
        return serialize_pandas(layers, name, obj)
    except:
        pass

    with open(
        (path := get_shelf() / utils.to_filename(layers, name, ".pkl")), "wb"
    ) as f:
        cloudpickle.dump(obj, f)
    return path.name


def deserialize(layers, name, obj):
    if (path := get_shelf() / utils.to_filename(layers, name, ".pth")).exists():
        import torch

        obj.load_state_dict(torch.load(path))
    elif (path := get_shelf() / utils.to_filename(layers, name, ".npy")).exists():
        import numpy

        obj[:] = numpy.load(path)
    elif (path := get_shelf() / utils.to_filename(layers, name, ".parquet")).exists():
        import pandas as pd

        obj.iloc[:, :] = pd.read_parquet(path)
    elif (path := get_shelf() / utils.to_filename(layers, name, ".pkl")).exists():
        with open(path, "rb") as f:
            loaded_obj = cloudpickle.load(f)
        obj.clear()
        obj.update(loaded_obj)
    else:
        # Reached during replay when the requested iteration has no checkpoint
        # in the object store -- usually because the adaptive throttle skipped
        # it. Failing loudly beats replaying from an uninitialized object and
        # reporting the resulting numbers as historical fact.
        stem = utils.to_filename(layers, name, "").stem
        raise RuntimeError(
            f"FLOR: no checkpoint for {name!r} at this iteration. Looked for "
            f"{stem}.{{pth,npy,parquet,pkl}} in {get_shelf()}. The run being "
            f"replayed likely throttled this iteration's checkpoint "
            f"(flor.set_ckpt_interval); narrow to an iteration that has one, or "
            f"re-run forward with a smaller interval."
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
