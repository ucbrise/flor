import flags
from record import EOF

pid = flags.PID

current = start = 0


class WeakEpoch:
    def __init__(self):
        ...

    def sparse(self):
        ...

    def dense(self):
        segments = [[] for _ in range(flags.PID[1])]
        for epoch in range(iterations_count):
            segments[epoch % flags.PID[1]].append(-1)
        i = 0
        for j in range(flags.PID[1]):
            for k in range(len(segments[j])):
                segments[j][k] = i
                i += 1
        assert i == iterations_count
        for segment in segments:
            for epoch in segment:
                assert epoch >= 0
        assert segments[-1][-1] == iterations_count - 1
        return segments[flags.PID[0] - 1][0]


def seek(log_record: EOF):
    sparse_checkpoints = log_record.sparse_checkpoints
    assert isinstance(sparse_checkpoints, bool)
    iterations_count = log_record.iterations_count
    assert isinstance(iterations_count, int)

    if not sparse_checkpoints:
        # All epochs are valid entries
        ...
    else:
        # Only a subset of epochs are valid entries
        ...

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
