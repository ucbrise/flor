from .constants import *
from . import cli
from . import utils
from . import versions

from typing import Any, Iterable, Iterator, TypeVar, Optional, Union
from contextlib import contextmanager

from tqdm import tqdm
import json
import atexit

import time
from datetime import datetime

T = TypeVar("T")

output_buffer = []

layers = {}
checkpoints = []

skip_cleanup = True


def log(name, value):
    global skip_cleanup
    skip_cleanup = False
    serializable_value = value if utils.is_jsonable(value) else str(value)
    if layers:
        output_buffer.append(utils.add2copy(layers, name, serializable_value))
        msg = f"{', '.join([f'{k}: {v}' for k,v in layers.items()])}, {name}: {str(serializable_value)}"
    else:
        output_buffer.append({name: serializable_value})
        msg = f"{name}: {str(serializable_value)}"
    tqdm.write(msg)
    return value


def arg(name: str, default: Optional[Any] = None) -> Any:
    if cli.in_replay_mode():
        # GIT
        pass
    elif name in cli.flags.hyperparameters:
        # CLI
        v = cli.flags.hyperparameters[name]
        if default is not None:
            v = utils.duck_cast(v, default)
            log(name, v)
            return v
        log(name, v)
        return v
    elif default is not None:
        # default
        log(name, default)
        return default
    else:
        raise


@contextmanager
def checkpointing(**kwargs):
    # set up the context
    checkpoints.extend(list(kwargs.items()))
    yield
    # tear down the context if needed
    checkpoints.clear()


def loop(name: str, iterator: Iterable[T]) -> Iterator[T]:
    pos = len(layers)
    if pos == 0:
        output_buffer.append(
            utils.add2copy(
                layers, f"enter::{name}", datetime.now().isoformat(timespec="seconds")
            )
        )
    layers[name] = 0
    for each in tqdm(iterator, position=pos, leave=(True if pos == 0 else False)):
        layers[name] += 1
        start_t = time.perf_counter()
        yield each
        elapsed_t = time.perf_counter() - start_t
        if pos == 0:
            output_buffer.append(utils.add2copy(layers, "auto::secs", elapsed_t))
            if is_due_chkpt(elapsed_t):
                chkpt()
    del layers[name]

    if pos == 0:
        output_buffer.append(
            utils.add2copy(
                layers, f"exit::{name}", datetime.now().isoformat(timespec="seconds")
            )
        )


def is_due_chkpt(elapsed_t):
    return True


def chkpt():
    for name, obj in checkpoints:
        pass


@atexit.register
def cleanup():
    if skip_cleanup:
        return
    if not cli.in_replay_mode():
        # RECORD
        branch = versions.current_branch()
        if branch is not None:
            output_buffer.append(
                {
                    "PROJID": PROJID,
                    "BRANCH": branch,
                    "TSTAMP": TIMESTAMP,
                    "FILENAME": SCRIPTNAME,
                }
            )
            with open(".flor.json", "w") as f:
                json.dump(output_buffer, f, indent=2)
            versions.git_commit(f"FLOR::Auto-commit::{TIMESTAMP}")
    else:
        # REPLAY
        pass


__all__ = ["log", "arg", "checkpointing", "loop"]
