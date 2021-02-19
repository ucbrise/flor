from . import flags
from . import florin
from . import file
from . import needle
from .skip_block import SkipBlock

from typing import Iterable, Union


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

    init()

    if not flags.REPLAY:
        # Record mode
        if isinstance(value, bool):
            if not value:
                file.close()
            return value
        else:
            for each in value:
                yield each
            file.close()
    else:
        # Replay mode
        segment = needle.get_segment(file.TREE.sparse_checkpoints, file.TREE.iterations_count)
        for capsule in segment:
            flags.RESUMING = capsule.init_only
            if isinstance(value, bool):
                yield True
            else:
                assert isinstance(value, Iterable)
                if flags.RESUMING:
                    if capsule.epoch is needle.NO_INIT:
                        continue
                    else:
                        # TODO: ...
                        assert hasattr(value, '__getitem__'), "TODO: Implement next() calls to consume iterator"
                        yield value[capsule.epoch]
                else:
                    assert capsule.epoch is not needle.NO_INIT
                    # TODO: ...
                    assert hasattr(value, '__getitem__'), "TODO: Implement next() calls to consume iterator"
                    yield value[capsule.epoch]


def init(_nil=[]):
    """
    At most once execution
    """
    if not _nil:
        assert flags.NAME is not None
        florin.mk_job(flags.NAME)
        if flags.REPLAY:
            file.read()
        SkipBlock.bind()
        _nil.append(True)


__all__ = ['it', 'SkipBlock']