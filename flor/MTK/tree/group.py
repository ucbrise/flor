from .block import Block
from typing import List


class BlockGroup:
    def __init__(self, first: Block):
        self.blocks: List[Block] = [
            first,
        ]

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
