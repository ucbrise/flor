from dataclasses import dataclass
from enum import Enum

REPLAY_MODE = Enum("REPLAY_MODE", "weak strong")
RECORD_MODE = Enum("RECORD_MODE", "chkpt_restore")
SHADOW_BRANCH_PREFIX = "flor.shadow"
FLORFILE = ".replay.json"


@dataclass
class REPLAY_PARALLEL:
    pid: int
    ngpus: int


__all__ = [
    "REPLAY_MODE",
    "RECORD_MODE",
    "SHADOW_BRANCH_PREFIX",
    "FLORFILE",
    "REPLAY_PARALLEL",
]
