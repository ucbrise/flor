from . import file
from flor import flags
from flor.tree import Tree
from flor.tree.window import Window

from typing import Union

class Journal:
    def __init__(self):
        self.tree = Tree()
        self.sub_tree = None
        self.entries = None

    def read(self):
        self.entries = file.read()
        self.tree.parse(self.entries)

    def get_segment_window(self) -> Window:
        assert flags.PID[1] <= self.tree.iterations_count
        if self.tree.sparse_checkpoints:
            assert flags.PID[1] <= len(self.tree.sparse_checkpoints) + 1, \
                f"Not enough checkpoints. Max degree of parallelism: {len(self.tree.sparse_checkpoints) + 1}"
        if flags.MODE == flags.WEAK and flags.PID[0] > 1:
            self._advance_head()
            return self.sub_tree.get_segment()
        return self.tree.get_segment()

    def as_tree(self) -> Tree:
        if not flags.REPLAY or flags.MODE == flags.STRONG or flags.PID[0] == 1:
            return self.tree
        else:
            assert self.sub_tree is not None
            return self.sub_tree

    def _advance_head(self):
        """
        Used for checkpoint resume,
        ignores journal entries that precede the first epoch of work
        """
        assert self.sub_tree is None, "Need a fresh Tree to feed"
        self.sub_tree = Tree()
        epoch_to_init: Union[int, None] = self.tree.get_resume_epoch()
        if epoch_to_init is not None:
            target = self.tree[self.tree.root.static_key].blocks[epoch_to_init]
            feeding = False
            for journal_entry in self.entries:
                if (not feeding and journal_entry.is_left() and
                        journal_entry.sk == target.static_key and
                        journal_entry.gk == target.global_key):
                    feeding = True
                if feeding:
                    self.sub_tree.feed_entry(journal_entry)
