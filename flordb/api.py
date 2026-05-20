import os
import shlex
import sys
from pathlib import Path
from .constants import *
from .clock import Clock
from . import orm
from . import cli
from . import utils
from . import versions
from . import obj_store
from . import database

from typing import Any, Iterable, Iterator, List, TypeVar, Optional
from contextlib import contextmanager

from tqdm import tqdm
import atexit

CMD_FILE = os.path.join(CURRDIR, ".flor.cmd")

T = TypeVar("T")

output_buffer = []
run_args: dict = {}

layers = {}
context: List[orm.Segment] = []

checkpoints = []
checkpointing_clock = Clock()

skip_cleanup = True


def _ctx_snapshot() -> Optional[List[orm.Segment]]:
    return list(context) if context else None


def log(name, value):
    if skip_cleanup:
        _deferred_init()

    serializable_value = value if utils.is_jsonable(value) else str(value)
    tqdm.write(utils.to_string(layers, name, serializable_value))

    if cli.in_replay_mode() and name not in cli.flags.queryparameters["VARS"]:
        # Check that name is in logging statement propagation list
        return value

    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            _ctx_snapshot(),
            name,
            serializable_value,
            1,
        )
    )

    return value


def arg(name: str, default: Optional[Any] = None) -> Any:
    if cli.in_replay_mode():
        # GIT
        assert name in cli.flags.hyperparameters
        historical_v = cli.flags.hyperparameters[name]
        log(name, historical_v)
        run_args[name] = historical_v
        return historical_v
    elif name in cli.flags.hyperparameters:
        # CLI
        v = cli.flags.hyperparameters[name]
        if default is not None:
            v = utils.duck_cast(v, default)
            log(name, v)
            run_args[name] = v
            return v
        log(name, v)
        run_args[name] = v
        return v
    elif default is not None:
        # default
        log(name, default)
        run_args[name] = default
        return default
    else:
        raise


@contextmanager
def checkpointing(**kwargs):
    try:
        output_buffer.append(
            orm.Log(
                PROJID,
                Clock.get_datetime(),
                SCRIPTNAME,
                _ctx_snapshot(),
                "delta::prefix",
                checkpointing_clock.get_delta(),
                3,
            )
        )
        checkpoints.extend(list(kwargs.items()))
        yield
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        checkpoints.clear()
        checkpointing_clock.set_start_time()


@contextmanager
def iteration(name: str, idx: Optional[int], value: Optional[str]):
    clock = Clock()
    clock.set_start_time()
    layers[name] = (
        int(idx) if idx is not None else None,
        str(value) if value is not None else None,
    )
    if cli.in_replay_mode():
        # TODO: load the end-state checkpoint
        load_ckpt()
        raise
    context.append(orm.Segment(name, layers[name][0], layers[name][1]))
    try:
        yield
        ckpt()
    finally:
        context.pop()
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            _ctx_snapshot(),
            "delta::iteration",
            clock.get_delta(),
            3,
        )
    )
    del layers[name]


def loop(name: str, iterator: Iterable[T]) -> Iterator[T]:
    clock = Clock()
    clock.set_start_time()
    pos = len(layers)
    layers[name] = (0, None)
    context.append(orm.Segment(name, 0, None))
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
        context[-1] = orm.Segment(name, layers[name][0], layers[name][1])
        if pos == 0 and cli.in_replay_mode():
            load_ckpt()
        yield each[1]  # type: ignore
        if pos == 0 and not cli.in_replay_mode():
            ckpt()
    context.pop()
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            _ctx_snapshot(),
            "delta::loop",
            clock.get_delta(),
            3,
        )
    )
    del layers[name]


def commit():
    global skip_cleanup
    tstamp = Clock.get_datetime()
    output_buffer.append(
        orm.Log(
            PROJID,
            tstamp,
            SCRIPTNAME,
            _ctx_snapshot(),
            "delta::suffix",
            checkpointing_clock.get_delta(),
            3,
        )
    )
    conn, cursor = database.conn_and_cursor()
    if not cli.in_replay_mode():
        # RECORD
        branch = versions.current_branch()
        if branch is not None:
            orm.to_jsonl(output_buffer, tstamp)
            database.unpack(output_buffer, cursor)
            _write_cmd_file(tstamp)
            versions.git_commit(_build_commit_message(tstamp, run_args))
    else:
        database.unpack(output_buffer, cursor)
    conn.commit()
    conn.close()
    output_buffer.clear()
    run_args.clear()
    Clock.set_new_datetime()
    checkpointing_clock.s_time = None
    skip_cleanup = True


def _build_commit_message(tstamp: str, args: dict) -> str:
    subject = f"{versions.AUTO_COMMIT_SUBJECT_PREFIX}{tstamp}"
    if not args:
        return subject
    body = "\n".join(f"{k}={v}" for k, v in args.items())
    return f"{subject}\n\n{body}"


def _write_cmd_file(tstamp: str):
    cmd = " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv])
    with open(CMD_FILE, "w") as f:
        f.write(f"{tstamp}\n{cmd}\n")


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
            versions.ensure_gitignored(".flor/")
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
    if qop == 1 or not cli.flags.queryparameters["WEV"]:
        return enumerate(iterator)

    new_slice = []
    if qop == 0:
        new_slice.append((len(original) - 1, original[-1]))
        return new_slice

    assert isinstance(qop, (list, tuple))
    qop = [i for i in qop if i.isnumeric()]
    for i in qop:
        new_slice.append((i, original[int(i)]))
    return new_slice


__all__ = [
    "log",
    "arg",
    "checkpointing",
    "loop",
    "iteration",
    "commit",
    "output_buffer",
]
