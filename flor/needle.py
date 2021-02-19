from . import flags

from typing import List, Union


NO_INIT = None


class Capsule:
    def __init__(self, init_only: bool, epoch: Union[int, None]):
        self.init_only = init_only
        self.epoch = epoch


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

    def sparse(self, hi=False) -> int:
        temp = self.iterations_count
        self.iterations_count = len(self.sparse_checkpoints)
        our_segment = self.get_segment(flags.PID[0] - 1)
        self.iterations_count = temp
        # TODO: ...
        assert our_segment, "TODO: Handle case when user allocs more partitions than there is work."
        if not hi:
            return self.sparse_checkpoints[our_segment[0]]
        else:
            if flags.PID[0] == flags.PID[1]:
                # This is the last segment
                return self.iterations_count
            else:
                # There exists a greater segment
                return self.sparse_checkpoints[our_segment[-1] + 1]

    def dense(self) -> int:
        our_segment = self.get_segment(flags.PID[0] - 1)
        # TODO: ...
        assert our_segment, "TODO: Handle case when user allocs more partitions than there is work."
        pred_epoch = our_segment[0] - 1 if our_segment[0] else NO_INIT
        return pred_epoch


def seek(sparse_checkpoints: List[int], iterations_count: int) -> Union[int, None]:
    assert isinstance(sparse_checkpoints, List)
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


def get_segment(sparse_checkpoints: List[int], iterations_count: int) -> List[Capsule]:
    """
    The first element in the range corresponds to the predecessor and is only for initialization purposes
    """
    assert isinstance(sparse_checkpoints, List)
    assert isinstance(iterations_count, int)
    weak_epoch = WeakEpoch(iterations_count, sparse_checkpoints)

    if not sparse_checkpoints:
        # All epochs are valid entries
        our_segment = weak_epoch.get_segment(flags.PID[0] - 1)
        # TODO: ...
        assert our_segment, "TODO: Handle case when user allocs more partitions than there is work."
        if flags.MODE == flags.WEAK:
            # Asks to initialize predecessor
            return [Capsule(True, weak_epoch.dense()), ] + [Capsule(False, e) for e in our_segment]
        else:
            return [Capsule(True, e) for e in range(our_segment[0])] + [Capsule(False, e) for e in our_segment]
    else:
        # Only a subset of epochs are valid entries
        # Only weak initialization is possible
        assert flags.MODE == flags.WEAK
        lo = weak_epoch.sparse()
        hi = weak_epoch.sparse(hi=True)
        assert hi is not NO_INIT
        return [Capsule(True, lo), ] + \
               [Capsule(False, e) for e in range(lo + 1 if lo is not None else 0, hi)]
