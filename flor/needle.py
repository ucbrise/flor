import flags
from record import EOF
from typing import List, Union


NO_INIT = None


class WeakEpoch:
    def __init__(self, iterations_count: int, sparse_checkpoints: List[int]):
        self.iterations_count = iterations_count
        self.sparse_checkpoints = [NO_INIT,] + sparse_checkpoints

    def get_segment(self, idx) -> List[int]:
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
        return segments[idx]

    def sparse(self) -> int:
        temp = self.iterations_count
        self.iterations_count = len(self.sparse_checkpoints)
        our_segment = self.get_segment(flags.PID[0] - 1)
        self.iterations_count = temp
        assert our_segment, "TODO: Handle case when user allocs more partitions than there is work."
        return self.sparse_checkpoints[our_segment[0]]

    def dense(self) -> int:
        our_segment = self.get_segment(flags.PID[0] - 1)
        assert our_segment, "TODO: Handle case when user allocs more partitions than there is work."
        pred_epoch = our_segment[0] - 1 if our_segment[0] else NO_INIT
        return pred_epoch


def seek(log_record: EOF) -> Union[int, None]:
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
            return NO_INIT
    else:
        # Only a subset of epochs are valid entries
        # Only weak initialization is possible
        assert flags.MODE == flags.WEAK
        return weak_epoch.sparse()
