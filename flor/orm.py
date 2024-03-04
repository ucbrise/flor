from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import random


@dataclass
class Context:
    ctx_id: int
    p_ctx: Optional["Context"]


def generate_64bit_id() -> int:
    return random.randint(-(2**63), 2**63 - 1)


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


def to_context(d):
    if isinstance(d, Context):
        return d
    p_context = None if d["p_ctx"] is None else to_context(d["p_ctx"])
    if "iteration" in d:
        return Loop(d["ctx_id"], p_context, d["name"], d["iteration"], d["value"])
    else:
        return Func(d["ctx_id"], p_context, d["name", d["int_arg"], d["txt_arg"]])


@dataclass
class Log:
    projid: str
    tstamp: str
    filename: str
    ctx: Optional[Context]
    name: str
    value: Any
    type: int


def to_json(output_buffer: List[Log]):
    def _serialize(obj):
        if is_dataclass(obj):
            return asdict(obj)
        return str(obj)  # Fallback for unrecognized types

    output_buffer = [_serialize(o) for o in output_buffer]
    with open(".flor.json", "w") as f:
        json.dump(output_buffer, f, default=_serialize, indent=2)
