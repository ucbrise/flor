from typing import Iterable, Union
from skip_block import LogFile


def it(value: Union[Iterable, bool]):
    """
    Main loop wrapper
    :param value:
        Iterable when iterating over a for loop
        Bool when looping with while
    """
    assert isinstance(value, (Iterable, bool))
    if isinstance(value, bool):
        if not value and 'EXEC':
            LogFile.close()
        return value
    else:
        for each in value:
            yield each
        if 'EXEC':
            LogFile.close()
