import os
from pathlib import Path

from .constants import *
from . import utils

import cloudpickle

OBJSTORE = Path(HOMEDIR) / "obj_store"
SHELF = OBJSTORE / PROJID / TIMESTAMP
os.makedirs(SHELF, exist_ok=True)


def serialize_torch(layers, name, obj):
    import torch
    import torch.nn
    import torch.optim

    if isinstance(obj, torch.nn.Module) or isinstance(obj, torch.optim.Optimizer):
        path = SHELF / utils.to_filename(layers, name, ".pth")
        torch.save(obj.state_dict(), path)
        return path.name
    else:
        raise


def serialize_numpy(layers, name, obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        path = SHELF / utils.to_filename(layers, name, ".npy")
        np.save(path, obj)
        return path.name
    else:
        raise


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
        path = SHELF / utils.to_filename(layers, name, ".pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return path.name
    else:
        raise


def serialize_pandas(layers, name, obj):
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        path = SHELF / utils.to_filename(layers, name, ".parquet")
        obj.to_parquet(path)
        return path.name
    else:
        raise


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

    with open((path := SHELF / utils.to_filename(layers, name, ".pkl")), "wb") as f:
        cloudpickle.dump(obj, f)
    return path.name


def deserialize(layers, name, obj):
    if (path := SHELF / utils.to_filename(layers, name, ".pth")).exists():
        import torch

        obj.load_state_dict(torch.load(path))
    elif (path := SHELF / utils.to_filename(layers, name, ".npy")).exists():
        import numpy

        obj[:] = numpy.load(path)
    elif (path := SHELF / utils.to_filename(layers, name, ".parquet")).exists():
        import pandas as pd

        obj.iloc[:, :] = pd.read_parquet(path)
    elif (path := SHELF / utils.to_filename(layers, name, ".pkl")).exists():
        with open(path, "rb") as f:
            loaded_obj = cloudpickle.load(f)
        obj.clear()
        obj.update(loaded_obj)
