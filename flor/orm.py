from dataclasses import dataclass, asdict
from typing import Any, List, Optional
import json
import random


def generate_64bit_id() -> int:
    return random.randint(-(2**63), 2**63 - 1)


@dataclass
class Loop:
    ctx_id: int
    p_ctx: Optional["Loop"]
    name: str
    iteration: int
    value: str


@dataclass
class Log:
    projid: str
    tstamp: str
    filename: str
    ctx: Optional[Loop]
    name: str
    value: Any
    type: int


def to_json(output_buffer: List[Log]):
    output_buffer = [asdict(o) for o in output_buffer]
    with open(".flor.json", "w") as f:
        json.dump(output_buffer, f, indent=2)
