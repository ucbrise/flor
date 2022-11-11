from .value import Value as DataVal
from .reference import Reference as DataRef

__all__ = ["DataVal", "DataRef"]

try:
    import torch
    from .torch_chkpt import Torch

    __all__.append("Torch")
except ModuleNotFoundError:
    pass
