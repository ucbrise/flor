from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


entries: Dict[str, int] = {}
logs: List[Log] = []


def parse_log(stem, obj, loop):
    log_record = Log(
        stem["PROJID"],
        stem["TSTAMP"],
        stem["FILENAME"],
        loop,
        obj["value_name"],
        obj["value"],
        Val_Type.AUTO if "::" in obj["value_name"] else Val_Type.RAW,
    )
    logs.append(log_record)
    return log_record


def parse_loop(obj):
    loop = None
    for k, v in entries.items():
        if k in obj:
            loop = Loop(loop, k, v, obj[k])
    return loop


def parse_entries(obj):
    if obj["value_name"].startswith("enter::"):
        entries[obj["value_name"].split("::")[1]] = (
            entries.get(obj["value_name"].split("::")[1], 0) + 1
        )
