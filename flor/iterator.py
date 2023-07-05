from typing import Iterable, List, Union

from flor import flags
from flor.shelf import home_shelf as shelf, cwd_shelf
from flor.skipblock import SkipBlock

from flor.constants import *

from flor.state import State


runtime_initialized = False


def it(value: Union[Iterable, bool]):
    """
    Main loop wrapper
    :param value:
        Iterable when iterating over a for loop
        Bool when looping with while
    """
    assert isinstance(value, (Iterable, bool))
    if flags.NAME is None:
        if isinstance(value, bool):
            return value
        else:
            assert isinstance(value, Iterable)
            for each in value:
                yield each
            return

    _deferred_init()

    if not flags.REPLAY:
        # Record mode
        if isinstance(value, bool):
            return value
        else:
            for each in value:
                yield each
    else:
        # Replay mode
        segment = SkipBlock.journal.get_segment_window()
        for capsule in segment:
            flags.RESUMING = capsule.init_only
            if isinstance(value, bool):
                yield True
            else:
                assert isinstance(value, Iterable)
                if flags.RESUMING:
                    if capsule.epoch is None:
                        continue
                    else:
                        State.epoch = int(capsule.epoch)
                        assert hasattr(
                            value, "__getitem__"
                        ), "TODO: Implement next() calls to consume iterator"
                        yield value[capsule.epoch]  # type: ignore
                else:
                    assert capsule.epoch is not None
                    State.epoch = value[capsule.epoch]  # type: ignore
                    assert hasattr(
                        value, "__getitem__"
                    ), "TODO: Implement next() calls to consume iterator"
                    yield value[capsule.epoch]  # type: ignore


def _deferred_init(_nil=[]):
    """
    At most once execution
    """
    global runtime_initialized
    if not runtime_initialized:
        if flags.NAME is not None:
            assert (
                cwd_shelf.in_shadow_branch()
            ), f"Please run FLOR from a shadow branch (branch name: `{SHADOW_BRANCH_PREFIX}.[...]`)\n"
            SkipBlock.bind()
            if flags.REPLAY:
                SkipBlock.journal.read()
        if flags.NAME and not flags.REPLAY:
            index_path = shelf.get_index()
            SkipBlock.logger.set_path(index_path)

        runtime_initialized = True


__all__ = ["it"]
