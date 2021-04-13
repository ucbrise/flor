from flor import flags
from typing import List, Union

NO_INIT = None


class Window:
    def __init__(self, iterations_count: int, sparse_checkpoints: List[int]):
        self.iterations_count = iterations_count
        self.extended_sparse_checkpoints = [NO_INIT, ] + sparse_checkpoints

    def get_work_segment(self):
        if self._is_sparse():
            # Only a subset of epochs are valid entries
            # Only weak initialization is possible
            assert flags.MODE == flags.WEAK
            lo = self._sparse()
            hi = self._sparse(hi=True)
            assert hi is not None
            return [Capsule(True, lo), ] + \
                   [Capsule(False, e) for e in range(lo + 1 if lo is not None else 0, hi)]
        else:
            # All epochs are valid entries
            our_segment = self._get_segment_helper(self.iterations_count)[flags.PID[0] - 1]
            # TODO: ...
            assert our_segment, "TODO: Handle case when user allocs more partitions than there is work."
            if flags.MODE == flags.WEAK:
                # Asks to initialize predecessor
                return [Capsule(True, self._dense()), ] + [Capsule(False, e) for e in our_segment]
            else:
                return [Capsule(True, e) for e in range(our_segment[0])] + [Capsule(False, e) for e in our_segment]

    def get_resume_epoch(self):
        """
        i : the epoch from which to resume (weak initialization)
        """
        if flags.MODE == flags.STRONG:
            return None
        if self._is_sparse():
            return self._sparse()
        else:
            return self._dense()

    def _is_sparse(self):
        is_sparse = len(self.extended_sparse_checkpoints) > 1
        assert is_sparse or len(self.extended_sparse_checkpoints) == 1 and self.extended_sparse_checkpoints[0] is NO_INIT
        return is_sparse

    def _get_segment_helper(self, num_resume_points):
        segments: List[List[Union[int, None]]] = [[] for _ in range(flags.PID[1])]
        for pt in range(num_resume_points):
            segments[pt % flags.PID[1]].append(None)
        i = 0
        for j in range(flags.PID[1]):
            for k in range(len(segments[j])):
                segments[j][k] = i
                i += 1
        assert i == num_resume_points
        return segments

    def _sparse(self, hi=False) -> Union[int, None]:
        our_segment = self._get_segment_helper(len(self.extended_sparse_checkpoints))[flags.PID[0] - 1]
        # TODO: ...
        assert our_segment, "TODO: Handle case when user allocs more partitions than there is work."
        # Re-index the segment with respect to self.sparse_checkpoints
        if not hi:
            return self.extended_sparse_checkpoints[our_segment[0]]
        else:
            if flags.PID[0] == flags.PID[1]:
                # This is the last segment
                return self.iterations_count
            else:
                # There exists a greater segment
                return self.extended_sparse_checkpoints[our_segment[-1] + 1]

    def _dense(self) -> Union[int, None]:
        our_segment = self._get_segment_helper(self.iterations_count)[flags.PID[0] - 1]
        # TODO: ...
        assert our_segment, "TODO: Handle case when user allocs more partitions than there is work."
        pred_epoch = our_segment[0] - 1 if our_segment[0] else NO_INIT
        return pred_epoch


class Capsule:
    def __init__(self, init_only: bool, epoch: Union[int, None]):
        self.init_only = init_only
        self.epoch = epoch
