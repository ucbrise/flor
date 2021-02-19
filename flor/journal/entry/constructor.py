from .constants import *
from .data import *
from .metadata import *

from typing import Union


def make_entry(json_dict: dict) -> Union[DataRef, DataVal, Bracket, EOF]:
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
        else:
            assert DataRef.is_superclass(json_dict)
            return DataRef.cons(json_dict)