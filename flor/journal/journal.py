from typing import List, Union

from flor import flags
from flor.constants import *
from flor.journal.entry import EOF
from flor.tree import Tree
from flor.tree.window import Capsule

from . import file


class Journal:
    def __init__(self):
        self.tree = Tree()
        self.sub_tree = None
        self.entries = None

    def read(self):
        self.entries = file.read()
        self.tree.parse(self.entries)

    def get_segment_window(self) -> List[Capsule]:
        assert flags.PID.ngpus <= self.tree.iterations_count
        if self.tree.sparse_checkpoints:
            assert (
                flags.PID.ngpus <= len(self.tree.sparse_checkpoints) + 1
            ), f"Not enough checkpoints. Max degree of parallelism: {len(self.tree.sparse_checkpoints) + 1}"
        if flags.MODE is REPLAY_MODE.weak and flags.PID.pid > 1:
            self._advance_head()
            assert self.sub_tree is not None
            return self.sub_tree.get_segment()
        return self.tree.get_segment()

    def get_iterations_count(self):
        tree = self.as_tree()
        return tree.iterations_count

    def as_tree(self) -> Tree:
        if not flags.REPLAY or flags.MODE is REPLAY_MODE.strong or flags.PID.pid == 1:
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
        assert self.sub_tree is None, "Need a fresh Tree to feed"
        self.sub_tree = Tree()
        epoch_to_init: Union[int, None] = self.tree.get_resume_epoch()
        if epoch_to_init is not None:
            assert self.tree.root is not None
            target = self.tree[self.tree.root.static_key].blocks[epoch_to_init]
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
