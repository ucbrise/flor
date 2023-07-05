import json
from typing import List, Union

from flor import flags
from flor.state import State
from flor.shelf import home_shelf
from flor.constants import *
from flor.logger import exp_json

from .entry import *
from .tree import Tree
from .tree.window import Capsule


def read_entries() -> List[Union[d_entry, md_entry]]:
    assert flags.REPLAY
    index = home_shelf.get_index()
    assert index is not None, "FileNotFound. Missing checkpoint directory, or job name"

    with open(index, "r") as f:
        return [make_entry(json.loads(line.strip())) for line in f]


class Journal:
    def __init__(self):
        self.tree = Tree()  # type: ignore
        self.sub_tree = None
        self.entries = None

    def read(self):
        self.entries = read_entries()
        self.tree.parse(self.entries)

    def get_segment_window(self) -> List[Capsule]:
        assert flags.PID.ngpus <= self.tree.iterations_count
        if self.tree.sparse_checkpoints:
            assert (
                flags.PID.ngpus <= len(self.tree.sparse_checkpoints) + 1
            ), f"Not enough checkpoints. Max degree of parallelism: {len(self.tree.sparse_checkpoints) + 1}"
        if flags.MODE is REPLAY_MODE.weak and flags.PID.pid != 1:
            self._advance_head()
            assert self.sub_tree is not None
            return self.sub_tree.get_segment()
        return self.tree.get_segment()

    def get_iterations_count(self):
        return self.as_tree().iterations_count

    def as_tree(self) -> Tree:
        if (
            not flags.REPLAY
            or flags.MODE is REPLAY_MODE.strong
            or (flags.PID.pid == 1 and flags.PID.ngpus == 1)
        ):
            return self.tree
        else:
            assert self.sub_tree is not None
            return self.sub_tree

    def get_eof(self, commit_sha: str):
        tree = self.as_tree()
        return EOF(tree.sparse_checkpoints, tree.iterations_count, commit_sha)

    def _advance_head(self):
        """
        Used for checkpoint resume,
        ignores journal entries that precede the first epoch of work
        """
        self.sub_tree = Tree()  # type: ignore
        epoch_to_init: Union[int, None] = self.tree.get_resume_epoch()
        if epoch_to_init is not None:
            assert self.tree.root is not None
            target = self.tree[self.tree.root.static_key].blocks[epoch_to_init]
            State.target_block = target  # type: ignore
            feeding = False
            assert self.entries is not None
            for journal_entry in self.entries:
                if (
                    not feeding
                    and journal_entry.is_left()
                    and journal_entry.sk == target.static_key
                    and journal_entry.gk == target.global_key
                ):
                    feeding = True
                if feeding:
                    self.sub_tree.feed_entry(journal_entry)


__all__ = ["Journal"]
