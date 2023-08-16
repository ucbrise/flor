from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from enum import Enum

Val_Type = Enum("Val_Type", "RAW REF AUTO")


@dataclass
class Loop:
    parent: Optional["Loop"]
    name: str
    entries: int
    iteration: int


@dataclass
class Log:
    projid: str
    tstamp: str
    filename: str
    loop: Loop
    name: str
    value: Any
    type: Val_Type


loops: Dict[Tuple[str, int, int], Loop] = {}


def get_loop(obj):
    loop = None
    for k, v in obj.items():
        if k == "value" or k == "value_name":
            continue
        loop = Loop(loop, k, 10, v)
    return loop
