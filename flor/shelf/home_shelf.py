import uuid
from datetime import datetime
from pathlib import Path, PurePath
from typing import Optional, Union

from flor import flags
from flor.state import State
from flor.logger import exp_json
from flor.skipblock import SkipBlock

home: Path = Path.home()

florin: Path = home / ".flor"
florin.mkdir(exist_ok=True)

job: Optional[Path] = None
data: Optional[Path] = None
timestamp: Optional[str] = None


def mk_job(name: str):
    global timestamp, job, data
    assert isinstance(name, str)
    timestamp = datetime.now().isoformat()
    exp_json.put("TSTAMP", str(PurePath(timestamp).with_suffix(".json")))
    State.timestamp = timestamp
    job = florin / name
    job.mkdir(exist_ok=True)
    data = job / "data"
    data.mkdir(exist_ok=True)
    (job / "csv").mkdir(exist_ok=True)


def set_job(name: str):
    global timestamp, job, data
    assert isinstance(name, str)
    timestamp = exp_json.get("TSTAMP")
    State.timestamp = timestamp
    job = florin / name
    data = job / "data"


def get_index() -> Optional[Path]:
    if job is not None and timestamp is not None:
        if flags.REPLAY:
            projid = exp_json.get("PROJID")
            tstamp = exp_json.get("TSTAMP")
            assert projid is not None and tstamp is not None
            return florin / PurePath(projid) / PurePath(tstamp)
        else:
            return job / PurePath(timestamp).with_suffix(".json")
    else:
        return None


def get_job():
    assert job is not None
    return job


def get_latest() -> Optional[Path]:
    return job / PurePath("latest").with_suffix(".json") if job is not None else None


def get_pkl_ref() -> Optional[Path]:
    return (
        data / PurePath(uuid.uuid4().hex).with_suffix(".pkl")
        if data is not None
        else None
    )


def get_csv_ref(name, tstamp) -> Optional[Path]:
    return (
        job
        / PurePath("csv")
        / PurePath(f"{name}_{flags.NAME}_{tstamp}").with_suffix(".csv")
        if job is not None
        else None
    )


def verify(path: Union[PurePath, str]) -> bool:
    assert flags.NAME is not None
    resolved_path = florin / flags.NAME / path
    return resolved_path.exists()


def close():
    if not flags.REPLAY and flags.NAME:
        if len(SkipBlock.logger.buffer) > 0 or SkipBlock.logger.flush_count:
            path = get_index()
            assert path is not None
            if len(SkipBlock.logger.buffer) > 0:
                SkipBlock.logger.flush(is_final=True)
            latest = get_latest()
            assert latest is not None
            assert path.exists()
            if latest.exists():
                latest.unlink()
            latest.symlink_to(path)
            return path
