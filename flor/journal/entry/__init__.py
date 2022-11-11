from .constants import LBRACKET, RBRACKET, METADATA
from .data import *
from .metadata import *

from .data import *
from .metadata import *

from typing import Union


def make_entry(
    json_dict: dict,
) -> Union[DataRef, DataVal, Bracket, EOF, DataFrame, Torch]:
    if METADATA in json_dict:
        # Metadata Record
        if Bracket.is_superclass(json_dict):
            return Bracket.cons(json_dict)
        else:
            assert EOF.is_superclass(json_dict)
            return EOF.cons(json_dict)
    else:
        # Data Record
        if DataVal.is_superclass(json_dict):
            return DataVal.cons(json_dict)
        elif DataFrame.is_superclass(json_dict):
            return DataFrame.cons(json_dict)
        elif DataRef.is_superclass(json_dict):
            return DataFrame.cons(json_dict)
        else:
            assert Torch.is_superclass(json_dict)
            return Torch.cons(json_dict)

del Union
