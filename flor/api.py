import os
from pathlib import Path
from .constants import *
from .clock import Clock
from . import cli
from . import utils
from . import versions
from . import obj_store
from . import database

from typing import Any, Iterable, Iterator, TypeVar, Optional, Union
from contextlib import contextmanager

from tqdm import tqdm
import json
import atexit
import sqlite3

import time
from datetime import datetime

T = TypeVar("T")

output_buffer = []

layers = {}
checkpoints = []

skip_cleanup = True


def log(name, value):
    if skip_cleanup:
        _deferred_init()

    serializable_value = value if utils.is_jsonable(value) else str(value)
    output_buffer.append(
        utils.add2copy(
            utils.add2copy(layers, "value_name", name), "value", serializable_value
        )
    )
    tqdm.write(utils.to_string(layers, name, serializable_value))

    return value


def arg(name: str, default: Optional[Any] = None) -> Any:
    if cli.in_replay_mode():
        # GIT
        assert name in cli.flags.hyperparameters
        historical_v = cli.flags.hyperparameters[name]
        log(name, historical_v)
        return historical_v
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
    output_buffer.append(
        utils.add2copy(
            utils.add2copy(layers, "value_name", f"enter::{name}"),
            "value",
            datetime.now().isoformat(timespec="seconds"),
        )
    )
    layers[name] = 0
    for each in tqdm(
        enumerate(slice(name, iterator)),
        position=pos,
        leave=(True if pos == 0 else False),
    ):
        layers[name] = list(enumerate(iterator)).index(each) + 1 if pos == 0 else layers[name] + 1  # type: ignore
        start_t = time.perf_counter()
        if pos == 0 and cli.in_replay_mode():
            load_chkpt()
        yield each[1]  # type: ignore
        elapsed_t = time.perf_counter() - start_t
        if pos == 0:
            output_buffer.append(
                utils.add2copy(
                    utils.add2copy(layers, "value_name", "auto::secs"),
                    "value",
                    elapsed_t,
                )
            )
            if is_due_chkpt(elapsed_t):
                chkpt()
    del layers[name]
    output_buffer.append(
        utils.add2copy(
            utils.add2copy(layers, "value_name", f"exit::{name}"),
            "value",
            datetime.now().isoformat(timespec="seconds"),
        )
    )


def commit():
    # Add logging statements on REPLAY
    conn = sqlite3.connect(os.path.join(HOMEDIR, Path(PROJID).with_suffix(".db")))
    cursor = conn.cursor()
    database.create_tables(cursor)

    if not cli.in_replay_mode():
        # RECORD
        branch = versions.current_branch()
        if branch is not None:
            output_buffer.append(
                {
                    "PROJID": PROJID,
                    "TSTAMP": Clock.get_time(),
                    "FILENAME": SCRIPTNAME,
                }
            )
            database.unpack(output_buffer, cursor)
            with open(".flor.json", "w") as f:
                json.dump(output_buffer, f, indent=2)
            versions.git_commit(f"FLOR::Auto-commit::{Clock.get_time()}")
    else:
        output_buffer.append(
            {"PROJID": PROJID, "TSTAMP": cli.flags.old_tstamp, "FILENAME": SCRIPTNAME}
        )
        database.unpack(output_buffer, cursor)
    conn.commit()
    conn.close()
    output_buffer.clear()
    Clock.set_new_time()


@atexit.register
def cleanup():
    if skip_cleanup:
        return
    commit()


def _deferred_init():
    global skip_cleanup
    if skip_cleanup:
        skip_cleanup = False
        if not cli.in_replay_mode():
            assert (
                versions.current_branch() is not None
            ), "Running from a detached HEAD?"
            versions.to_shadow()


def is_due_chkpt(elapsed_t):
    return not cli.in_replay_mode()


def chkpt():
    for name, obj in checkpoints:
        obj_store.serialize(layers, name, obj)


def load_chkpt():
    for name, obj in checkpoints:
        obj_store.deserialize(layers, name, obj)


def slice(name, iterator):
    if not cli.in_replay_mode():
        return iterator
    assert cli.flags.queryparameters is not None
    original = list(iterator)

    if not cli.flags.queryparameters:
        return [
            original[-1],
        ]

    qop = (cli.flags.queryparameters).get(name, 0)
    if qop == 1:
        return iterator

    new_slice = []
    if qop == 0:
        return new_slice

    assert isinstance(qop, (list, tuple))
    for i in qop:
        new_slice.append(original[int(i) - 1])
    return new_slice


__all__ = ["log", "arg", "checkpointing", "loop", "commit"]
