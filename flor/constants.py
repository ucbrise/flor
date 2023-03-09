from dataclasses import dataclass
from enum import Enum
from pathlib import Path

REPLAY_MODE = Enum("REPLAY_MODE", "weak strong")
SHADOW_BRANCH_PREFIX = "flor.shadow"
LOG_RECORDS = Path(".flor") / "log_records.csv"
REPLAY_JSON = Path(".flor") / ".replay.json"
SECONDS_JSON = Path(".flor") / "seconds.json"

DATA_PREP = (
    "projid",
    "runid",
    "tstamp",
    "vid",
)
OUTR_LOOP = DATA_PREP + ("epoch",)
INNR_LOOP = OUTR_LOOP + ("step",)

DESERIALIZATION_COEFF = 3.573e-08


@dataclass
class REPLAY_PARALLEL:
    pid: int
    ngpus: int
