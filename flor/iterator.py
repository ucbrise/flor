from . import flags
from . import shelf
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
                SkipBlock.logger.append(SkipBlock.journal.as_tree().get_eof())
                SkipBlock.logger.close()
            return value
        else:
            for each in value:
                yield each
            SkipBlock.logger.append(SkipBlock.journal.as_tree().get_eof())
            SkipBlock.logger.close()
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
        SkipBlock.bind()
        if flags.REPLAY:
            SkipBlock.journal.read()
        else:
            SkipBlock.logger.set_path(shelf.get_index())
            assert SkipBlock.logger.path is not None
        _nil.append(True)

__all__ = ['it']
