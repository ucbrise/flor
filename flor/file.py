from . import florin
from . import needle
from .journal.entry import DataVal, DataRef, Bracket, EOF, make_entry

import json
import pathlib
from typing import Union, List
from collections import OrderedDict


class Block:
    def __init__(self, log_record: Bracket, parent=None):
        assert log_record.is_left()

        self.child: Union['Block', None] = None
        self.parent: Union['Block', None] = parent
        self.successor: Union['Block', None] = None

        self.static_key = log_record.sk
        self.global_key = log_record.gk
        self.data_records: List[Union[DataVal, DataRef]] = []
        self.right_fed = False
        self.force_mat = False

    def belongs_in_block(self, data_record: Union[DataVal, DataRef, Bracket]):
        assert data_record.is_right()
        return (
            data_record.sk == self.static_key and
            data_record.gk == self.global_key)

    def feed_record(self, data_record: Union[DataVal, DataRef, Bracket]):
        assert self.belongs_in_block(data_record)
        self.right_fed = True
        if not isinstance(data_record, Bracket):
            # If it's not a RBRACKET then it's a data record
            self.data_records.append(data_record)

    def force_mat_successors(self):
        assert self.parent is None
        first_static_key = self.static_key
        block = self.successor
        while block is not None and block.static_key != first_static_key:
            block.force_mat = True
            block = block.successor


class BlockGroup:
    def __init__(self, first: Block):
        self.blocks = [first, ]

        self.materialization_time = None
        self.computation_time = None
        self.executions_count = 0
        self.materializations_count = 0

    def tick_execution(self, t):
        self.executions_count += 1
        if self.computation_time is None:
            self.computation_time = t

    def should_time_mat(self):
        assert self.executions_count > 0
        return self.executions_count == 1

    def tick_materialization(self):
        self.materializations_count += 1

    def set_mat_time(self, t):
        if self.materialization_time is None:
            self.materialization_time = t

    def add_block(self, block: Block):
        self.blocks.append(block)

    def peek_block(self) -> Block:
        return self.blocks[-1]


class Tree:
    def __init__(self, log_record: Bracket = None):
        """
        LBRACKET creates new Tree
        """
        self.hash: OrderedDict[str, BlockGroup] = OrderedDict()
        self.block = None

        if log_record is not None:
            assert log_record.is_left()
            self.block = Block(log_record)
            self._hash(self.block)

        self.root = self.block

        self.sparse_checkpoints: List[int] = []
        self.iterations_count = 0

    def _hash(self, block: Block):
        if block.static_key in self.hash:
            self.hash[block.static_key].add_block(block)
        else:
            self.hash[block.static_key] = BlockGroup(block)

    def add_sparse_checkpoint(self):
        # print(f"SPARSE CHECKPOINT at {self.iterations_count - 1}")
        self.sparse_checkpoints.append(self.iterations_count - 1)

    def feed_record(self, log_record: Union[DataRef, DataVal, Bracket, EOF]):
        if self.root is None:
            assert self.block is None
            assert log_record.is_left()
            self.block = Block(log_record)
            self.root = self.block
            self._hash(self.block)
        elif log_record.is_left():
            if self.block.right_fed:
                successor = Block(log_record, self.block.parent)
                self.block.successor = successor
                self.block = successor
                self._hash(successor)
            else:
                child = Block(log_record, self.block)
                self.block.child = child
                self.block = child
                self._hash(child)
        elif log_record.is_right():
            if self.block.belongs_in_block(log_record):
                self.block.feed_record(log_record)
            else:
                assert self.block.parent is not None \
                       and self.block.parent.belongs_in_block(log_record)
                self.block = self.block.parent
                self.block.feed_record(log_record)
        else:
            assert isinstance(log_record, EOF)
            self.sparse_checkpoints = log_record.sparse_checkpoints
            self.iterations_count = log_record.iterations_count

        if log_record.is_left() and log_record.sk == self.root.static_key:
            self.iterations_count += 1


def read():
    global TREE
    with open(florin.get_index(), 'r') as f:
        for line in f:
            log_record = make_entry(json.loads(line.strip()))
            feed_record(log_record)
    epoch_to_init: Union[int, None] = needle.seek(TREE.sparse_checkpoints, TREE.iterations_count)
    if epoch_to_init is not None:
        target: Block = TREE.hash[TREE.root.static_key].blocks[epoch_to_init]
        TREE = Tree()
        feeding = False
        for log_record in records:
            if (not feeding and log_record.is_left() and
                    log_record.sk == target.static_key and
                    log_record.gk == target.global_key):
                feeding = True
            if feeding:
                TREE.feed_record(log_record)


def feed_record(log_record: Union[DataRef, DataVal, Bracket, EOF]):
    records.append(log_record)
    TREE.feed_record(log_record)


def write():
    with open(florin.get_index(), 'w') as f:
        for log_record in records:
            if isinstance(log_record, DataRef):
                log_record.set_ref_and_dump(florin.get_pkl_ref())
            f.write(json.dumps(log_record.jsonify()) + pathlib.os.linesep)
    records[:] = []


def close():
    feed_record(EOF(TREE.sparse_checkpoints, TREE.iterations_count))
    write()
    merge()


def merge():
    """
    Stitch together parallel-written files
    """
    if florin.get_latest().exists():
        florin.get_latest().unlink()
    florin.get_latest().symlink_to(florin.get_index())


records: List[Union[DataRef, DataVal, Bracket, EOF]] = []
TREE = Tree()
