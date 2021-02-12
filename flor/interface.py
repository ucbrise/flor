import flags
import florin
from typing import Iterable, Union
from skip_block import SkipBlock


def it(value: Union[Iterable, bool]):
    """
    Main loop wrapper
    :param value:
        Iterable when iterating over a for loop
        Bool when looping with while
    """
    if flags.NAME is None:
        return value
    assert isinstance(value, (Iterable, bool))

    init()

    if not flags.REPLAY:
        if isinstance(value, bool):
            if not value:
                SkipBlock.LogFile.close()
            return value
        else:
            for each in value:
                yield each
            SkipBlock.LogFile.close()
    else:
        if isinstance(value, bool):
            if not value:
                # Clean up
                ...
            return value
        else:
            for each in value:
                yield each
            # Clean up


def init(_nil=[]):
    """
    At most once execution
    """
    if not _nil:
        assert flags.NAME is not None
        florin.mk_job(flags.NAME)
        SkipBlock.bind()
        _nil.append(True)


__all__ = ['it', 'SkipBlock']
