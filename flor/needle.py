import flags
from record import EOF
from typing import List, Union


STRONG_EPOCH = 0


class WeakEpoch:
    def __init__(self, iterations_count: int, sparse_checkpoints: List[int]):
        self.iterations_count = iterations_count
        self.sparse_checkpoints = sparse_checkpoints

    def sparse(self) -> int:
        temp = self.iterations_count
        self.iterations_count = len(self.sparse_checkpoints)
        idx = self.dense() + 1
        self.iterations_count = temp
        return self.sparse_checkpoints[idx]

    def dense(self) -> int:
        segments: List[List[Union[int, None]]] = [[] for _ in range(flags.PID[1])]
        for epoch in range(self.iterations_count):
            segments[epoch % flags.PID[1]].append(None)

        i = 0
        for j in range(flags.PID[1]):
            for k in range(len(segments[j])):
                segments[j][k] = i
                i += 1
        assert i == self.iterations_count
        assert segments[-1][-1] == self.iterations_count - 1

        our_segment = segments[flags.PID[0] - 1]
        our_first_epoch = our_segment[0]
        return our_first_epoch - 1


def seek(log_record: EOF) -> int:
    sparse_checkpoints = log_record.sparse_checkpoints
    assert isinstance(sparse_checkpoints, List)
    iterations_count = log_record.iterations_count
    assert isinstance(iterations_count, int)
    weak_epoch = WeakEpoch(iterations_count, sparse_checkpoints)

    if not sparse_checkpoints:
        # All epochs are valid entries
        if flags.MODE == flags.WEAK:
            return weak_epoch.dense()
        else:
            return STRONG_EPOCH
    else:
        # Only a subset of epochs are valid entries
        # Only weak initialization is possible
        assert flags.MODE == flags.WEAK
        return weak_epoch.sparse()

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
