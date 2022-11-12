from .block import Block
from .group import BlockGroup
from .window import Window
from ..entry import *

from collections import OrderedDict as ODict
from typing import List, Optional, Union, Dict


class Tree:
    def __init__(self, log_entry: Optional[Bracket] = None):
        self.hash: Dict[str, BlockGroup] = ODict()
        self.block = None

        if log_entry is not None:
            assert log_entry.is_left()
            self.block = Block(log_entry)
            self._hash(self.block)

        self.root = self.block

        self.sparse_checkpoints: List[int] = []
        self.iterations_count = 0

    def __getitem__(self, item) -> BlockGroup:
        return self.hash[item]

    def _hash(self, block: Block):
        if block.static_key in self.hash:
            self.hash[block.static_key].add_block(block)
        else:
            self.hash[block.static_key] = BlockGroup(block)

    def add_sparse_checkpoint(self):
        # print(f"SPARSE CHECKPOINT at {self.iterations_count - 1}")
        self.sparse_checkpoints.append(self.iterations_count - 1)

    def feed_entry(self, log_entry: Union[d_entry, md_entry]):
        if self.root is None:
            assert self.block is None
            assert log_entry.is_left()
            assert isinstance(log_entry, Bracket)
            self.block = Block(log_entry)
            self.root = self.block
            self._hash(self.block)
        elif log_entry.is_left():
            assert self.block is not None
            if self.block.right_fed:
                assert isinstance(log_entry, Bracket)
                successor = Block(log_entry, self.block.parent)
                self.block.successor = successor
                self.block = successor
                self._hash(successor)
            else:
                assert isinstance(log_entry, Bracket)
                child = Block(log_entry, self.block)
                self.block.child = child
                self.block = child
                self._hash(child)
        elif log_entry.is_right():
            assert self.block is not None
            assert not isinstance(log_entry, EOF)
            if self.block.belongs_in_block(log_entry):
                self.block.feed_entry(log_entry)
            else:
                assert (
                    self.block.parent is not None
                    and self.block.parent.belongs_in_block(log_entry)
                )
                self.block = self.block.parent
                self.block.feed_entry(log_entry)
        else:
            assert isinstance(log_entry, EOF)
            self.sparse_checkpoints = log_entry.sparse_checkpoints
            self.iterations_count = log_entry.iterations_count

        if log_entry.is_left() and log_entry.sk == self.root.static_key:
            self.iterations_count += 1

    def parse(self, records: List[Union[d_entry, md_entry]]):
        assert self.block is None, "Need a new Tree for parsing"
        for rec in records:
            self.feed_entry(rec)

    def get_resume_epoch(self):
        """
        Gets the predecessor epoch for weak initialization,
            if such epoch exists, else None.
        """
        return Window(self.iterations_count, self.sparse_checkpoints).get_resume_epoch()

    def get_segment(self):
        """
        Gets the relevant work segment, including tagged initialization epochs
        """
        return Window(self.iterations_count, self.sparse_checkpoints).get_work_segment()
