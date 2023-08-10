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
        torch.save(obj, SHELF / utils.to_filename(layers, name, ".pth"))
    else:
        raise


def serialize_numpy(layers, name, obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        np.save(SHELF / utils.to_filename(layers, name, ".npy"), obj)
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
        with open(SHELF / utils.to_filename(layers, name, ".pkl"), "wb") as f:
            pickle.dump(obj, f)
    else:
        raise


def serialize_pandas(layers, name, obj):
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        path = SHELF / utils.to_filename(layers, name, ".parquet")
        obj.to_parquet(path)
    else:
        raise


def serialize(layers, name, obj):
    try:
        serialize_torch(layers, name, obj)
    except:
        pass

    try:
        serialize_scikit(layers, name, obj)
    except:
        pass

    try:
        serialize_numpy(layers, name, obj)
    except:
        pass

    try:
        serialize_pandas(layers, name, obj)
    except:
        pass

    with open(SHELF / utils.to_filename(layers, name, ".pkl"), "wb") as f:
        cloudpickle.dump(obj, f)
