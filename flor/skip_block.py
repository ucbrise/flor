from . import flags
from . import file
from .copy import deepcopy
from .journal.entry import DataRef, DataVal, Bracket, LBRACKET, RBRACKET

import time
from typing import List, Union
from abc import ABC, abstractmethod


class SeemBlock(ABC):
    @staticmethod
    @abstractmethod
    def step_into(block_name: str, probed=False):
        ...

    @staticmethod
    @abstractmethod
    def end(*args, values=None):
        ...


class WriteBlock(SeemBlock):
    scaling_factor = 1.38
    dynamic_identifiers = dict()
    pda: List[Bracket] = []

    @staticmethod
    def step_into(block_name: str, probed=False):
        assert isinstance(block_name, str)
        dynamic_id = WriteBlock.dynamic_identifiers.get(block_name, 0)
        WriteBlock.dynamic_identifiers[block_name] = dynamic_id + 1

        lbracket = Bracket(block_name, dynamic_id, LBRACKET,
                           predicate=True, timestamp=time.time())
        file.feed_record(lbracket)
        WriteBlock.pda.append(lbracket)
        return lbracket.predicate

    @staticmethod
    def end(*args, values=None):
        lbracket = WriteBlock.pda.pop()
        block_group = file.TREE.hash[lbracket.sk]
        block = block_group.peek_block()
        assert block.global_key == lbracket.gk
        block_group.tick_execution(lbracket.timestamp - time.time())
        if not args:
            rbracket = Bracket(lbracket.sk, lbracket.gk, RBRACKET)
            file.feed_record(rbracket)
            block_group.set_mat_time(0)
        else:
            data_records = []

            start_time = time.time()
            for arg in args:
                data_record = val_to_record(arg, lbracket)
                data_records.append(data_record)
                block_group.should_time_mat() and data_record.would_mat()
            block_group.set_mat_time(start_time - time.time())

            if WriteBlock._should_materialize(block_group):
                for data_record in data_records:
                    file.feed_record(data_record)
                block_group.tick_materialization()
            else:
                rbracket = Bracket(lbracket.sk, lbracket.gk, RBRACKET)
                file.feed_record(rbracket)

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
            threshold *= block_group.executions_count / (block_group.materializations_count + 1)
            if ratio < threshold:
                file.TREE.add_sparse_checkpoint()
                block.force_mat_successors()
                return True

        return False


def val_to_record(arg, lbracket: Bracket) -> Union[DataRef, DataVal]:
    if hasattr(arg, 'state_dict'):
        arg = getattr(arg, 'state_dict')()

    if type(arg) in [type(None), int, float, bool, str]:
        return DataVal(lbracket.sk, lbracket.gk, arg)
    else:
        return DataRef(lbracket.sk, lbracket.gk, deepcopy(arg))


class ReadBlock(SeemBlock):
    dynamic_identifiers = dict()
    pda: List[Bracket] = []

    @staticmethod
    def step_into(block_name: str, probed=False):
        assert isinstance(block_name, str)
        dynamic_id = ReadBlock.dynamic_identifiers.get(block_name, 0)
        ReadBlock.dynamic_identifiers[block_name] = dynamic_id + 1

        lbracket = Bracket(block_name, dynamic_id, LBRACKET, predicate=not flags.RESUMING and probed)
        ReadBlock.pda.append(lbracket)
        return lbracket.predicate

    @staticmethod
    def end(*args, values=None):
        lbracket = ReadBlock.pda.pop()
        block = file.TREE.hash[lbracket.sk].blocks[lbracket.gk]
        if not lbracket.predicate:
            for data_record, arg in zip(block.data_records, args):
                data_record.make_val()
                value_serialized = data_record.value
                if hasattr(arg, 'load_state_dict'):
                    # PyTorch support
                    arg.load_state_dict(value_serialized)
                elif isinstance(arg, list):
                    assert isinstance(value_serialized, list)
                    arg[:] = value_serialized
                elif isinstance(arg, dict):
                    assert isinstance(value_serialized, dict)
                    arg.update(value_serialized)
                elif hasattr(arg, '__dict__'):
                    if isinstance(value_serialized, dict):
                        arg.__dict__.update(arg)
                    else:
                        assert type(arg) == type(value_serialized)
                        arg.__dict__.update(value_serialized.__dict__)
                else:
                    # TODO: ...
                    raise RuntimeError("TODO: add hooks for user-defined de-serialization")
        # TODO: ...
        assert values is None, "TODO: Add support for literals/atomics"


class SkipBlock(SeemBlock):
    @staticmethod
    def step_into(block_name: str, probed=False):
        if flags.NAME is not None:
            raise RuntimeError("SkipBlock missing dynamic linking")
        return True

    @staticmethod
    def end(*args, values=None):
        if flags.NAME is not None:
            raise RuntimeError("SkipBlock missing dynamic linking")

    @staticmethod
    def bind():
        block = ReadBlock if flags.REPLAY else WriteBlock
        SkipBlock.step_into = block.step_into
        SkipBlock.end = block.end


__all__ = ['SkipBlock']
