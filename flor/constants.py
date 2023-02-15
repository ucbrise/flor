from dataclasses import dataclass
from enum import Enum

REPLAY_MODE = Enum("REPLAY_MODE", "weak strong")
SHADOW_BRANCH_PREFIX = "flor.shadow"
FLORFILE = "log_records.csv"
REPLAY_JSON = ".replay.json"


@dataclass
class REPLAY_PARALLEL:
    pid: int
    ngpus: int
