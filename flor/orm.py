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

class ORM:
    def __init__(self) -> None:
        self.entries: Dict[str, int] = {}
        self.logs: List[Log] = []
        self.loops: Dict[Tuple[int, str, int, int], int] = {}


    def parse_log(self, stem, obj, loop):
        log_record = Log(
            stem["PROJID"],
            stem["TSTAMP"],
            stem["FILENAME"],
            loop,
            obj["value_name"],
            obj["value"],
            Val_Type.AUTO if "::" in obj["value_name"] else Val_Type.RAW,
        )
        self.logs.append(log_record)
        return log_record


    def parse_loop(self, obj):
        loop = None
        for k, v in self.entries.items():
            if k in obj:
                loop = Loop(loop, k, v, obj[k])
        return loop


    def parse_entries(self, obj):
        if obj["value_name"].startswith("enter::"):
            self.entries[obj["value_name"].split("::")[1]] = (
                self.entries.get(obj["value_name"].split("::")[1], 0) + 1
        )
