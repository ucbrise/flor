from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set
import glob
import json
import os

from .constants import EXTRACT_DIR, RUNS_DIR


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


def extract_jsonl_path(tstamp: str) -> str:
    return os.path.join(EXTRACT_DIR, f"{tstamp}.jsonl")


def extract_jsonl_paths() -> List[str]:
    return sorted(glob.glob(os.path.join(EXTRACT_DIR, "*.jsonl")))


def write_extractions(derived: List[Log], scope: Set[str]) -> List[str]:
    """Save extracted metrics one file per run, mirroring runs/<tstamp>.jsonl.

    `scope` is the set of runs whose captured text this pass actually read. A
    run in it that yielded nothing has its file removed -- the extraction rule
    changed and no longer finds what it used to. A run outside it is left
    alone.

    That distinction is the whole safety property. These files are tracked and
    committed, so treating "not written this pass" as "no longer yields" would
    let a pass over a partial cache -- a teammate's runs present in git but not
    yet unpacked -- delete a committed reading of a run it never looked at.
    """
    by_tstamp: Dict[str, List[Log]] = {}
    for record in derived:
        by_tstamp.setdefault(record.tstamp, []).append(record)

    written = []
    for tstamp, records in by_tstamp.items():
        path = extract_jsonl_path(tstamp)
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(asdict(record)) + "\n")
        written.append(path)

    for tstamp in scope - set(by_tstamp):
        path = extract_jsonl_path(tstamp)
        if os.path.exists(path):
            os.remove(path)
    return sorted(written)


def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
