from .constants import *
from . import cli
from . import utils

from typing import Any, Iterable, Iterator, TypeVar, Optional, Union
from contextlib import contextmanager


T = TypeVar("T")


layers = []
checkpoints = []


def log(name, value):
    serializable_value = value if utils.is_jsonable(value) else str(value)
    print(name, serializable_value)
    # if State.loop_nesting_level:
    #     log_records.put(name, serializable_value)
    # else:
    #     exp_json.put(name, serializable_value, ow=False)
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
    layers[:] = []
    checkpoints[:] = []


def layer(name: str, iterator: Iterable[T]) -> Iterator[T]:
    layers.append(name)
    for each in iterator:
        yield each


__all__ = ["log", "arg", "checkpointing", "layer"]
