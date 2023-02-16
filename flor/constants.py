from dataclasses import dataclass
from enum import Enum
from pathlib import Path

REPLAY_MODE = Enum("REPLAY_MODE", "weak strong")
SHADOW_BRANCH_PREFIX = "flor.shadow"
LOG_RECORDS = Path(".flor") / "log_records.csv"
REPLAY_JSON = Path(".flor") / ".replay.json"


@dataclass
class REPLAY_PARALLEL:
    pid: int
    ngpus: int
