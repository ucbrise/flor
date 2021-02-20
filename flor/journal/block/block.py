from ..entry import DataVal, DataRef, Bracket
from typing import Union, List


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

    def feed_entry(self, data_record: Union[DataVal, DataRef, Bracket]):
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
