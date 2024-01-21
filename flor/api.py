import os
from copy import deepcopy
from pathlib import Path
from .constants import *
from .clock import Clock
from .orm import Loop, Log, to_json
from . import cli
from . import utils
from . import versions
from . import obj_store
from . import database

from typing import Any, Iterable, Iterator, TypeVar, Optional, Union
from contextlib import contextmanager

from tqdm import tqdm
import atexit
import sqlite3

T = TypeVar("T")

output_buffer = []

layers = {}
loop_ctx = None
entries = {}

checkpoints = []
checkpointing_clock = Clock()

skip_cleanup = True


def log(name, value):
    if skip_cleanup:
        _deferred_init()

    serializable_value = value if utils.is_jsonable(value) else str(value)
    output_buffer.append(
        Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            loop_ctx if loop_ctx is None else deepcopy(loop_ctx),
            name,
            serializable_value,
            1,
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
    # Add prefix time delta to output_buffer
    output_buffer.append(
        Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            loop_ctx if loop_ctx is None else deepcopy(loop_ctx),
            "delta::prefix",
            checkpointing_clock.get_delta(),
            3,
        )
    )
    # set up the context
    checkpoints.extend(list(kwargs.items()))
    yield
    # tear down the context if needed
    checkpoints.clear()
    checkpointing_clock.set_start_time()


def loop(name: str, iterator: Iterable[T]) -> Iterator[T]:
    global loop_ctx
    clock = Clock()
    clock.set_start_time()
    pos = len(layers)
    layers[name] = 0
    ent_n = entries.get(name, 1)
    entries[name] = ent_n + 1
    loop_ctx = Loop(
        loop_ctx if loop_ctx is None else deepcopy(loop_ctx), name, ent_n, layers[name]
    )
    for each in tqdm(
        enumerate(slice(name, iterator))
        if not cli.in_replay_mode()
        else slice(name, iterator),
        position=pos,
        leave=(True if pos == 0 else False),
    ):
        layers[name] = list(enumerate(iterator)).index(each) + 1 if pos == 0 else layers[name] + 1  # type: ignore
        loop_ctx.iteration = layers[name]
        if pos == 0 and cli.in_replay_mode():
            load_ckpt()
        yield each[1]  # type: ignore
        if pos == 0 and not cli.in_replay_mode():
            ckpt()
    loop_ctx = loop_ctx.parent
    output_buffer.append(
        Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            loop_ctx if loop_ctx is None else deepcopy(loop_ctx),
            "delta::loop",
            clock.get_delta(),
            3,
        )
    )
    del layers[name]


def commit():
    # Add suffix time delta to output_buffer
    output_buffer.append(
        Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            loop_ctx if loop_ctx is None else deepcopy(loop_ctx),
            "delta::suffix",
            checkpointing_clock.get_delta(),
            3,
        )
    )
    # Add logging statements on REPLAY
    conn = sqlite3.connect(os.path.join(HOMEDIR, Path(PROJID).with_suffix(".db")))
    cursor = conn.cursor()
    database.create_tables(cursor)

    if not cli.in_replay_mode():
        # RECORD
        branch = versions.current_branch()
        if branch is not None:
            database.unpack(output_buffer, cursor)
            to_json(output_buffer)
            versions.git_commit(f"FLOR::Auto-commit::{Clock.get_datetime()}")
    else:
        database.unpack(output_buffer, cursor)
    conn.commit()
    conn.close()
    output_buffer.clear()
    Clock.set_new_datetime()
    checkpointing_clock.s_time = None
    global skip_cleanup
    skip_cleanup = True


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


def ckpt():
    for name, obj in checkpoints:
        obj_store.serialize(layers, name, obj)


def load_ckpt():
    for name, obj in checkpoints:
        obj_store.deserialize(layers, name, obj)


def slice(name, iterator):
    if not cli.in_replay_mode():
        return iterator
    original = list(iterator)

    qop = (cli.flags.queryparameters).get(name, 0)
    if qop == 1:
        return enumerate(iterator)

    new_slice = []
    if qop == 0:
        new_slice.append((len(original) - 1, original[-1]))
        return new_slice

    assert isinstance(qop, (list, tuple))
    for i in qop:
        new_slice.append((i, original[int(i)]))
    return new_slice


__all__ = ["log", "arg", "checkpointing", "loop", "commit"]
