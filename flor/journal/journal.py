from . import file
from . import needle
from .block import Tree

from typing import Union

tree = Tree()
sub_tree = Tree()


def read():
    file.read()
    tree.parse(file.entries)


def feed(journal_entry):
    file.feed(journal_entry)
    tree.feed_entry(journal_entry)


def write():
    file.write()


def close():
    file.close(tree.get_eof())


def advance_head():
    """
    Used for checkpoint resume,
    ignores journal entries that precede the first epoch of work
    """
    assert sub_tree.root is None, "Need a fresh Tree to feed"
    epoch_to_init: Union[int, None] = needle.seek(tree.sparse_checkpoints, tree.iterations_count)
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