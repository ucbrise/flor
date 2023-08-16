from dataclasses import dataclass
from typing import Any

from enum import Enum

Val_Type = Enum("Val_Type", "RAW REF AUTO CHKPT")


@dataclass
class Loop:
    parent: "Loop"
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
