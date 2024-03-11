import os
from copy import deepcopy
from pathlib import Path
from .constants import *
from .clock import Clock
from . import orm
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
context = None

checkpoints = []
checkpointing_clock = Clock()

skip_cleanup = True


def log(name, value):
    if skip_cleanup:
        _deferred_init()

    serializable_value = value if utils.is_jsonable(value) else str(value)
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            context if context is None else deepcopy(context),
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
    try:
        # Add prefix time delta to output_buffer
        output_buffer.append(
            orm.Log(
                PROJID,
                Clock.get_datetime(),
                SCRIPTNAME,
                context if context is None else deepcopy(context),
                "delta::prefix",
                checkpointing_clock.get_delta(),
                3,
            )
        )
        # set up the context
        checkpoints.extend(list(kwargs.items()))
        # TODO: while there are func skipblocks to load
        yield
    except Exception as e:
        # Here you can handle the exception if needed, or log it, etc.
        print(f"An error occurred: {e}")
        # Optionally re-raise the exception if you want the error to propagate
        raise
    finally:
        # This code runs whether an exception occurred or not
        checkpoints.clear()
        checkpointing_clock.set_start_time()


@contextmanager
def iteration(name: str, idx: Optional[int], value: Optional[str]):
    global context
    clock = Clock()
    clock.set_start_time()
    layers[name] = (
        int(idx) if idx is not None else None,
        str(value) if value is not None else None,
    )
    parent_context = context
    if cli.in_replay_mode():
        # TODO: load the end-state checkpoint
        load_ckpt()
        raise
    else:
        context = orm.Loop(
            orm.generate_64bit_id(),
            deepcopy(parent_context) if parent_context is not None else None,
            name,
            layers[name][0],
            layers[name][1],
        )
        yield
        ckpt()
    context = parent_context
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            deepcopy(context) if context is not None else None,
            "delta::iteration",
            clock.get_delta(),
            3,
        )
    )
    del layers[name]


def loop(name: str, iterator: Iterable[T]) -> Iterator[T]:
    global context
    clock = Clock()
    clock.set_start_time()
    pos = len(layers)
    layers[name] = (0, None)
    parent_context = context
    for each in tqdm(
        (
            enumerate(slice(name, iterator))
            if not cli.in_replay_mode()
            else slice(name, iterator)
        ),
        position=pos,
        leave=(True if pos == 0 else False),
    ):
        layers[name] = (
            int(each[0]) + 1,
            str(each[1]) if utils.is_jsonable(each[1]) else None,
        )
        context = orm.Loop(
            orm.generate_64bit_id(),
            deepcopy(parent_context) if parent_context is not None else None,
            name,
            layers[name][0],
            layers[name][1],
        )
        if pos == 0 and cli.in_replay_mode():
            load_ckpt()
        yield each[1]  # type: ignore
        if pos == 0 and not cli.in_replay_mode():
            ckpt()
    context = parent_context
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            deepcopy(context) if context is not None else None,
            "delta::loop",
            clock.get_delta(),
            3,
        )
    )
    del layers[name]


def commit():
    global skip_cleanup
    # Add suffix time delta to output_buffer
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            None if context is None else deepcopy(context),
            "delta::suffix",
            checkpointing_clock.get_delta(),
            3,
        )
    )
    # Add logging statements on REPLAY
    conn, cursor = database.conn_and_cursor()
    if not cli.in_replay_mode():
        # RECORD
        branch = versions.current_branch()
        if branch is not None:
            orm.to_json(output_buffer)
            database.unpack(output_buffer, cursor)
            versions.git_commit(f"FLOR::Auto-commit::{Clock.get_datetime()}")
    else:
        database.unpack(output_buffer, cursor)
    conn.commit()
    cursor = conn.cursor()
    database.deduplicate_table(cursor, "loops")
    conn.commit()
    conn.close()
    output_buffer.clear()
    Clock.set_new_datetime()
    checkpointing_clock.s_time = None
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

    qop = (
        (cli.flags.queryparameters).get(name, 0)
        if cli.flags.queryparameters is not None
        else 0
    )
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


__all__ = ["log", "arg", "checkpointing", "loop", "iteration", "commit"]
