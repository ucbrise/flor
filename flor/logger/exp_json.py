from flor.constants import REPLAY_JSON, SECONDS_JSON
from flor import flags
from flor.state import State
from typing import Optional, Dict, Any

import json, os
from pathlib import Path
import time

replay_d: Optional[Dict[str, Any]] = None
record_d = {}


def _get_path(p=REPLAY_JSON):
    assert State.common_dir is not None
    return os.path.join(os.path.dirname(State.common_dir), p)


def deferred_init():
    global replay_d
    if State.common_dir is not None:
        with open(_get_path(), "r", encoding="utf-8") as f:
            replay_d = json.load(f)


def exists():
    if State.common_dir is not None:
        path = Path(_get_path())
        return path.exists()
    else:
        return False


def put(name, value, ow=True):
    assert ow or name not in record_d
    record_d[name] = value


def get(name):
    """Get from previous run"""
    assert replay_d is not None
    return replay_d.get(name, None)


def flush():
    if flags.NAME and not flags.REPLAY:
        assert State.common_dir is not None
        record_d['CLI'] = vars(flags.parser.nsp)
        with open(_get_path(), "w", encoding="utf-8") as f:
            json.dump(record_d, f, ensure_ascii=False, indent=4)

        assert State.import_time is not None
        State.seconds["PREP"] = (
            State.seconds["PREP"]
            if State.seconds["PREP"] is not None
            else time.time() - State.import_time
        )
        assert State.seconds["PREP"] is not None
        with open(_get_path(SECONDS_JSON), "w", encoding="utf-8") as f:
            json.dump(State.seconds, f, indent=4)
