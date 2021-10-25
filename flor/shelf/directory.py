import uuid
from datetime import datetime
from pathlib import Path, PurePath
from typing import Optional, Union

from flor import flags

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
    job = florin / name
    job.mkdir(exist_ok=True)
    data = job / "data"
    data.mkdir(exist_ok=True)


def get_index() -> Optional[Path]:
    if job is not None and timestamp is not None:
        if flags.REPLAY:
            assert flags.INDEX is not None
            return job / flags.INDEX.with_suffix(".json")
        else:
            return job / PurePath(timestamp).with_suffix(".json")
    else:
        return None


def get_latest() -> Optional[Path]:
    return job / PurePath("latest").with_suffix(".json") if job is not None else None


def get_pkl_ref() -> Optional[Path]:
    return (
        data / PurePath(uuid.uuid4().hex).with_suffix(".pkl")
        if data is not None
        else None
    )


def verify(path: Union[PurePath, str]) -> bool:
    assert flags.NAME is not None
    resolved_path = florin / flags.NAME / path
    return resolved_path.exists()
