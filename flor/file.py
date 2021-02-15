import florin
from record import *
from typing import Union, List
from collections import OrderedDict

import json
import os


class Block:
    def __init__(self, log_record: Bracket, parent=None):
        assert log_record.is_left()

        self.child: 'Block' = None
        self.parent: 'Block' = parent
        self.successor: 'Block' = None

        self.static_key = log_record.sk
        self.global_key = log_record.gk
        self.data_records: List[Union[DataVal, DataRef]] = []
        self.right_fed = False

        # Adaptive checkpointing
        self.materialization_time = None
        self.computation_time = None
        self.executions_count = 0
        self.materializations_count = 0

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

    def force_mat_successors(self):
        assert self.parent is None
        first_static_key = self.static_key
        block = self.successor
        while block is not None and block.static_key != first_static_key:
            block.force_mat = True
            block = block.successor


class Tree:
    def __init__(self, log_record: Bracket = None):
        """
        LBRACKET creates new Tree
        """
        self.hash: OrderedDict[str, OrderedDict[int, Block]] = OrderedDict()
        self.block = None

        if log_record is not None:
            assert log_record.is_left()
            self.block = Block(log_record)
            self._hash(self.block)

        self.root = self.block

        self.sparse_checkpoints = False
        self.iterations_count = 0

    def _hash(self, block: Block):
        if block.static_key in self.hash:
            self.hash[block.static_key][block.global_key] = block
        else:
            d = OrderedDict()
            d[block.global_key] = block
            self.hash[block.static_key] = d

    @staticmethod
    def _depth_first_walk(block: Block):
        while True:
            if block.child is not None:
                Tree._depth_first_walk(block.child)
            yield block
            if block.successor is None:
                break
            block = block.successor

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

        if log_record.is_left() and log_record.sk == self.root.static_key:
            self.iterations_count += 1


def read():
    with open(florin.get_index(), 'r') as f:
        for line in f:
            log_record = make_record(json.loads(line.strip()))
            records.append(log_record)


def feed_record(log_record: Union[DataRef, DataVal, Bracket, EOF]):
    records.append(log_record)
    TREE.feed_record(log_record)


def write():
    with open(florin.get_index(), 'w') as f:
        for log_record in records:
            if isinstance(log_record, DataRef):
                log_record.set_ref_and_dump(florin.get_pkl_ref())
            f.write(json.dumps(log_record.jsonify()) + os.linesep)
    records[:] = []


def parse():
    for log_record in records:
        TREE.feed_record(log_record)


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
