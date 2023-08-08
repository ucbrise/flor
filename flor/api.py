from .constants import *
from . import cli
from . import utils
from . import versions

from typing import Any, Iterable, Iterator, TypeVar, Optional, Union
from contextlib import contextmanager

from tqdm import tqdm
import json
import atexit

import io

T = TypeVar("T")

output_buffer = io.StringIO(newline="\n")

layers = {}
checkpoints = []


def log(name, value):
    serializable_value = value if utils.is_jsonable(value) else str(value)
    stack = list(layers.items())
    if stack:
        msg = f"{str(stack)} {name}: {str(serializable_value)}"
    else:
        msg = f"{name}: {str(serializable_value)}"
    output_buffer.write(msg)
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
def checkpointing(*args):
    # set up the context
    checkpoints.extend(list(args))

    yield  # The code within the 'with' block will be executed here.

    # tear down the context if needed
    checkpoints[:] = []


def layer(name: str, iterator: Iterable[T]) -> Iterator[T]:
    pos = len(layers)
    layers[name] = 0
    for each in tqdm(iterator, position=pos, leave=(True if pos == 0 else False)):
        layers[name] += 1
        yield each
    del layers[name]


@atexit.register
def cleanup():
    if not cli.in_replay_mode():
        branch = versions.current_branch()
        with open(".flor.txt", "w") as f:
            f.write(output_buffer.getvalue())
        if branch is not None:
            msg = f"PROJID: {PROJID}, BRANCH: {branch}, TSTAMP: {TIMESTAMP}"
            print(msg)
            versions.git_commit()


__all__ = ["log", "arg", "checkpointing", "layer"]
