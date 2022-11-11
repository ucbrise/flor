from .value import Value as DataVal
from .reference import Reference as DataRef
from .dataframe import DataFrame

from typing import Type, Union

d_entry = Union[DataVal, DataRef, DataFrame]

__all__ = ["DataVal", "DataRef", "DataFrame", "d_entry"]

try:
    import torch
    from .torch_chkpt import Torch

    t_d_entry = Union[d_entry, Torch]
    __all__.extend(["Torch", "t_d_entry"])

except ModuleNotFoundError:
    pass
