from flor.journal.entry import *
from typing import Union, List, get_args


class Block:
    def __init__(self, log_record: md_entry, parent=None):
        assert log_record.is_left()

        self.child: Union["Block", None] = None
        self.parent: Union["Block", None] = parent
        self.successor: Union["Block", None] = None

        self.static_key = log_record.sk
        self.global_key = log_record.gk
        self.data_records: List[d_entry] = []
        self.right_fed = False
        self.force_mat = False

    def belongs_in_block(self, data_record: Union[d_entry, md_entry]):
        assert data_record.is_right()
        return data_record.sk == self.static_key and data_record.gk == self.global_key

    def feed_entry(self, data_record: Union[d_entry, md_entry]):
        assert self.belongs_in_block(data_record)
        self.right_fed = True
        if not isinstance(data_record, get_args(md_entry)):
            self.data_records.append(data_record)  # type: ignore

    def force_mat_successors(self):
        assert self.parent is None
        first_static_key = self.static_key
        block = self.successor
        while block is not None and block.static_key != first_static_key:
            block.force_mat = True
            block = block.successor
