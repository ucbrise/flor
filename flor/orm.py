from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
from enum import Enum

Val_Type = Enum("Val_Type", "RAW REF AUTO")


@dataclass
class Context:
    ctx_id: int
    parent_ctx_id: Optional[int]


@dataclass
class Loop(Context):
    name: str
    iteration: int
    value: str


@dataclass
class Func(Context):
    name: str
    int_arg: int
    txt_arg: str


@dataclass
class Log:
    projid: str
    tstamp: str
    filename: str
    ctx: Optional[Context]
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

    with open(".flor.json", "w") as f:
        json.dump(output_buffer, f, default=_serialize, indent=2)
