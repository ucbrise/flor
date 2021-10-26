from .seemblock import SeemBlock

from flor import flags
from flor import shelf
from flor.journal.entry import DataRef, DataVal, Bracket, LBRACKET, RBRACKET

import time
from typing import Dict, List, Union


class WriteBlock(SeemBlock):
    scaling_factor = 1.38
    dynamic_identifiers: Dict[str, int] = dict()
    pda: List[Bracket] = []

    @staticmethod
    def step_into(block_name: str, probed=None):
        assert isinstance(block_name, str)
        dynamic_id = WriteBlock.dynamic_identifiers.get(block_name, 0)
        WriteBlock.dynamic_identifiers[block_name] = dynamic_id + 1

        lbracket = Bracket(
            block_name, dynamic_id, LBRACKET, predicate=True, timestamp=time.time()
        )
        WriteBlock.journal.as_tree().feed_entry(lbracket)

        WriteBlock.logger.append(lbracket)
        WriteBlock.pda.append(lbracket)
        return lbracket.predicate

    @staticmethod
    def end(*args, values=None):
        lbracket = WriteBlock.pda.pop()
        assert lbracket.timestamp is not None
        block_group = WriteBlock.journal.as_tree()[lbracket.sk]
        block = block_group.peek_block()
        assert block.global_key == lbracket.gk
        block_group.tick_execution(time.time() - lbracket.timestamp)
        if not args:
            rbracket = Bracket(lbracket.sk, lbracket.gk, RBRACKET)
            WriteBlock.journal.as_tree().feed_entry(rbracket)
            WriteBlock.logger.append(rbracket)
            block_group.set_mat_time(0)
        else:
            data_records = []

            start_time = time.time()
            for arg in args:
                data_record = val_to_record(arg, lbracket)
                data_records.append(data_record)
                block_group.should_time_mat() and data_record.would_mat()
            block_group.set_mat_time(time.time() - start_time)

            if WriteBlock._should_materialize(block_group):
                for data_record in data_records:
                    WriteBlock.journal.as_tree().feed_entry(data_record)
                    WriteBlock.logger.append(data_record)
                block_group.tick_materialization()
            else:
                rbracket = Bracket(lbracket.sk, lbracket.gk, RBRACKET)
                WriteBlock.journal.as_tree().feed_entry(rbracket)
                WriteBlock.logger.append(rbracket)

    @staticmethod
    def _should_materialize(block_group):
        assert block_group.materialization_time is not None
        assert block_group.computation_time is not None

        block = block_group.peek_block()

        # Must align successor checkpoints for periodic checkpointing
        if block.force_mat:
            return True

        # First consider atomic case (always/never)
        ratio = block_group.materialization_time / block_group.computation_time
        threshold = min(1 / (1 + WriteBlock.scaling_factor), flags.EPSILON)
        if ratio < threshold:
            return True

        # Then account for parallelism speedup
        if block.parent is None:
            threshold *= block_group.executions_count / (
                block_group.materializations_count + 1
            )
            if ratio < threshold:
                WriteBlock.journal.as_tree().add_sparse_checkpoint()
                block.force_mat_successors()
                return True

        return False


def val_to_record(arg, lbracket: Bracket) -> Union[DataRef, DataVal]:
    if hasattr(arg, "state_dict"):
        arg = getattr(arg, "state_dict")()

    if type(arg) in [type(None), int, float, bool, str]:
        return DataVal(lbracket.sk, lbracket.gk, arg)
    else:
        return DataRef(lbracket.sk, lbracket.gk, arg)
