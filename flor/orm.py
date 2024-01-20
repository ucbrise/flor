from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
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


def to_json(output_buffer: List[Log]):
    def _serialize(obj):
        if is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, Enum):
            return obj.name
        return str(obj)  # Fallback for unrecognized types

    logs_as_dict = [asdict(log) for log in output_buffer]
    with open(".flor.json", "w") as f:
        json.dump(logs_as_dict, f, default=_serialize, indent=2)


# class ORM:
#     def __init__(self) -> None:
#         self.logs: List[Log] = []
#         self.loops: Dict[Tuple[int, str, int, int], int] = {}

#         self.prev_loops = []
#         self.entries = {}

#     def parse_log(self, stem, obj, loop):
#         log_record = Log(
#             stem["PROJID"],
#             stem["TSTAMP"],
#             stem["FILENAME"],
#             loop,
#             obj["value_name"],
#             obj["value"],
#             Val_Type.AUTO if "::" in obj["value_name"] else Val_Type.RAW,
#         )
#         self.logs.append(log_record)
#         return log_record

#     def parse_loop(self, obj):
#         if len(obj) > 2:
#             loop_name, loop_iteration = list(obj.items())[-3]
#             loop_address = tuple(obj.keys())[0:-2]

#             loop = Loop(
#                 None, loop_name, self.entries.get(loop_address, 1), loop_iteration
#             )

#             if (
#                 str(obj["value_name"]).startswith("delta::")
#                 and loop_address in self.entries
#             ):
#                 self.entries[loop_address] += 1
#             elif (
#                 str(obj["value_name"]).startswith("delta::")
#                 and loop_address not in self.entries
#             ):
#                 self.entries[loop_address] = 2

#             if (
#                 str(obj["value_name"]).startswith("delta::")
#                 and self.prev_loops
#                 and self.prev_loops[-1].name != loop_name
#             ):
#                 for orm_loop in self.prev_loops:
#                     orm_loop.parent = loop
#                 self.prev_loops.clear()
#             self.prev_loops.append(loop)
#             return loop
