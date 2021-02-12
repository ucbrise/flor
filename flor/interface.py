import flags
import florin
import file
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
                file.close()
            return value
        else:
            for each in value:
                yield each
            file.close()
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
        if flags.REPLAY:
            file.read()
            file.parse()
        SkipBlock.bind()
        _nil.append(True)


def get_partitions(num_epochs, num_partitions, pretraining, period):
    # Roundrobin allocation with pipelining
    assert num_partitions <= num_epochs
    if pretraining:
        del period
        partitions = [[] for _ in range(num_partitions)]
        for epoch in range(num_epochs):
            partitions[epoch % num_partitions].append(-1)
        i = 0
        for j in range(num_partitions):
            for k in range(len(partitions[j])):
                partitions[j][k] = i
                i += 1
        assert i == num_epochs
        for part in partitions:
            for each in part:
                assert each >= 0
        assert partitions[-1][-1] == num_epochs - 1
        return partitions
    else:
        range_regions = []
        i = 0
        while i*period < num_epochs:
            start = i*period
            stop = min((i+1)*period, num_epochs)
            range_regions.append(range(start, stop))
            i += 1
        partitions = [[] for _ in range(num_partitions)]
        for range_element in range(len(range_regions)):
            #roundrobin work allocation, early epochs first
            partitions[range_element % num_partitions].append(-1)
        for j in range(num_partitions):
            for k in range(len(partitions[j])):
                partitions[j][k] = range_regions.pop(0)
        assert len(range_regions) == 0
        partitions = [range(rs[0].start, rs[-1].stop) if rs else [] for rs in partitions]
        if num_partitions < num_epochs:
            return partitions
        else:
            # For when you sample a Fine-tuning run with sparse checkpoints
            return [range(p.start, s+1) for p in partitions for s in p]


__all__ = ['it', 'SkipBlock']