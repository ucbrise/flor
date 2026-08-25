from dataclasses import dataclass, asdict
from typing import Any, List, Optional
import json
import os

from .constants import RUNS_DIR


@dataclass
class Segment:
    name: str
    iteration: Optional[int]
    value: Optional[str]


@dataclass
class Log:
    projid: str
    tstamp: str
    filename: str
    ctx: Optional[List[Segment]]
    name: str
    value: Any
    type: int


def run_jsonl_path(tstamp: str) -> str:
    return os.path.join(RUNS_DIR, f"{tstamp}.jsonl")


def to_jsonl(output_buffer: List[Log], tstamp: str):
    path = run_jsonl_path(tstamp)
    with open(path, "w") as f:
        for record in output_buffer:
            f.write(json.dumps(asdict(record)) + "\n")


def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
