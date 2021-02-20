from flor import flags
from flor import journal
from flor import shelf
from .skipblock import SkipBlock

from typing import Union, Iterable


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
            if not value:
                journal.close()
            return value
        else:
            for each in value:
                yield each
            journal.close()
    else:
        # Replay mode
        segment = journal.get_segment()
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
                        # TODO: ...
                        assert hasattr(value, '__getitem__'), "TODO: Implement next() calls to consume iterator"
                        yield value[capsule.epoch]
                else:
                    assert capsule.epoch is not None
                    # TODO: ...
                    assert hasattr(value, '__getitem__'), "TODO: Implement next() calls to consume iterator"
                    yield value[capsule.epoch]


def _deferred_init(_nil=[]):
    """
    At most once execution
    """
    if not _nil:
        assert flags.NAME is not None
        shelf.mk_job(flags.NAME)
        if flags.REPLAY:
            journal.read()
        SkipBlock.bind()
        _nil.append(True)


__all__ = ['it']
