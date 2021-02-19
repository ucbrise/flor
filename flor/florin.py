from . import flags

import uuid
from pathlib import Path, PurePath
from datetime import datetime

home = Path.home()
florin = home / '.flor'
florin.mkdir(exist_ok=True)

job = None
data = None
timestamp = None


def mk_job(name: str):
    global timestamp, job, data
    assert isinstance(name, str)
    timestamp = datetime.now().isoformat()
    job = florin / name
    job.mkdir(exist_ok=True)
    data = job / PurePath('data')
    data.mkdir(exist_ok=True)


def get_index():
    return job / PurePath(
        flags.INDEX if flags.REPLAY else timestamp
    ).with_suffix('.json')


def get_latest():
    return job / PurePath('latest').with_suffix('.json')


def get_pkl_ref() -> PurePath:
    while True:
        candidate = data / PurePath(uuid.uuid4().hex).with_suffix('.pkl')
        if not candidate.exists():
            return candidate

