from . import file
from .block import Tree
from .entry import DataRef, DataVal, Bracket, EOF
from flor import flags

from typing import Union

tree = Tree()
sub_tree = Tree()


def read():
    file.read()
    tree.parse(file.entries)


def feed(journal_entry: Union[DataRef, DataVal, Bracket, EOF]):
    file.feed(journal_entry)
    tree.feed_entry(journal_entry)


def write():
    file.write()


def close():
    file.close(tree.get_eof())


def get_segment():
    assert flags.PID[1] <= tree.iterations_count
    if tree.sparse_checkpoints:
        assert flags.PID[1] <= len(tree.sparse_checkpoints) + 1, \
            f"Not enough checkpoints. Max degree of parallelism: {len(tree.sparse_checkpoints) + 1}"
    if flags.MODE == flags.WEAK and flags.PID[0] > 1:
        advance_head()
        return sub_tree.get_segment()
    return tree.get_segment()


def as_tree():
    if not flags.REPLAY or flags.MODE == flags.STRONG or flags.PID[0] == 1:
        return tree
    else:
        return sub_tree


def add_sparse_checkpoint():
    tree.add_sparse_checkpoint()


def advance_head():
    """
    Used for checkpoint resume,
    ignores journal entries that precede the first epoch of work
    """
    assert sub_tree.root is None, "Need a fresh Tree to feed"
    epoch_to_init: Union[int, None] = tree.get_resume_epoch()
    if epoch_to_init is not None:
        target = tree[tree.root.static_key].blocks[epoch_to_init]
        feeding = False
        for journal_entry in file.entries:
            if (not feeding and journal_entry.is_left() and
                    journal_entry.sk == target.static_key and
                    journal_entry.gk == target.global_key):
                feeding = True
            if feeding:
                sub_tree.feed_entry(journal_entry)