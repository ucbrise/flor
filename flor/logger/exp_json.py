from flor.constants import REPLAY_JSON
from flor import flags
from typing import Optional, Dict, Any

import json
from pathlib import Path

replay_d: Optional[Dict[str, Any]] = None
record_d = {}


def deferred_init():
    global replay_d
    with open(REPLAY_JSON, "r", encoding="utf-8") as f:
        replay_d = json.load(f)


def exists():
    return Path(REPLAY_JSON).exists()


def put(name, value, ow=True):
    assert ow or name not in record_d
    record_d[name] = value


def get(name):
    if flags.REPLAY:
        assert replay_d is not None
        return replay_d.get(name, None)
    else:
        return record_d.get(name, None)


def flush():
    with open(REPLAY_JSON, "w", encoding="utf-8") as f:
        json.dump(record_d, f, ensure_ascii=False, indent=4)
