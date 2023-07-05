from flor import flags
from flor.constants import *
from flor.logger import exp_json

from typing import List, Union, Optional

NO_INIT = None


class Window:
    def __init__(self, iterations_count: int, sparse_checkpoints: List[int]):
        self.iterations_count = iterations_count
        self.extended_sparse_checkpoints: List[Optional[int]] = list(sparse_checkpoints)
        self.extended_sparse_checkpoints.insert(0, NO_INIT)

    def get_work_segment(self):
        if self._is_sparse():
            # Only a subset of epochs are valid entries
            # Only weak initialization is possible
            assert flags.MODE is REPLAY_MODE.weak
            lo = self._sparse()
            hi = self._sparse(hi=True)
            assert hi is not None
            return [
                Capsule(True, lo),
            ] + [Capsule(False, e) for e in range(lo + 1 if lo is not None else 0, hi)]
        else:
            # All epochs are valid entries
            our_segment = (
                self._get_segment_helper(self.iterations_count)[flags.PID.pid - 1]
                if flags.PID.pid > 0
                else []
            )
            if flags.MODE is REPLAY_MODE.weak:
                # Asks to initialize predecessor
                return [
                    Capsule(True, self._dense()),
                ] + [Capsule(False, e) for e in our_segment]
            else:
                idx = our_segment[0]
                assert idx is not None
                return [Capsule(True, e) for e in range(idx)] + [
                    Capsule(False, e) for e in our_segment
                ]

    def get_resume_epoch(self):
        """
        i : the epoch from which to resume (weak initialization)
        """
        if flags.MODE is REPLAY_MODE.strong:
            return None
        if self._is_sparse():
            return self._sparse()
        else:
            return self._dense()

    def _is_sparse(self):
        is_sparse = len(self.extended_sparse_checkpoints) > 1
        assert (
            is_sparse
            or len(self.extended_sparse_checkpoints) == 1
            and self.extended_sparse_checkpoints[0] is NO_INIT
        )
        return is_sparse

    def _get_segment_helper(self, num_resume_points):
        segments: List[List[Optional[int]]] = [[] for _ in range(flags.PID.ngpus)]
        for pt in range(num_resume_points):
            segments[pt % flags.PID.ngpus].append(None)
        i = 0
        for j in range(flags.PID.ngpus):
            for k in range(len(segments[j])):
                segments[j][k] = i
                i += 1
        assert i == num_resume_points
        return segments

    def _sparse(self, hi=False) -> Optional[int]:
        our_segment = self._get_segment_helper(len(self.extended_sparse_checkpoints))[
            flags.PID.pid - 1
        ]
        # TODO: ...
        assert (
            our_segment
        ), "TODO: Handle case when user allocs more partitions than there is work."
        # Re-index the segment with respect to self.sparse_checkpoints
        if not hi:
            assert our_segment[0] is not None
            return self.extended_sparse_checkpoints[our_segment[0]]
        else:
            if flags.PID.pid == flags.PID.ngpus:
                # This is the last segment
                return self.iterations_count
            else:
                # There exists a greater segment
                idx = our_segment[-1]
                assert idx is not None
                return self.extended_sparse_checkpoints[idx + 1]

    def _dense(self) -> Optional[int]:
        if flags.PID.pid == 0:
            assert isinstance(self.iterations_count, int) and self.iterations_count > 0
            return self.iterations_count - 1
        our_segment = self._get_segment_helper(self.iterations_count)[flags.PID.pid - 1]
        # TODO: ...
        assert (
            our_segment
        ), "TODO: Handle case when user allocs more partitions than there is work."
        pred_epoch = our_segment[0] - 1 if our_segment[0] else NO_INIT
        return pred_epoch


class Capsule:
    def __init__(self, init_only: bool, epoch: Optional[int]):
        self.init_only = init_only
        self.epoch = epoch

    def __str__(self) -> str:
        return f"init_only: {self.init_only}, epoch: {self.epoch}"
