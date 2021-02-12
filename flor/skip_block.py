"""
TODO: Add values (literals/atomics) support to SkipBlock.end()
"""
import flags
import florin
from record import DataRef, DataVal, Bracket, LBRACKET, RBRACKET
from copy import deepcopy
from file import File

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
    dynamic_identifiers = dict()
    pda: List[Bracket] = []

    @staticmethod
    def step_into(block_name: str, probed=False):
        assert isinstance(block_name, str)
        dynamic_id = WriteBlock.dynamic_identifiers.get(block_name, 0)
        WriteBlock.dynamic_identifiers[block_name] = dynamic_id + 1

        lbracket = Bracket(block_name, dynamic_id, LBRACKET,
                           predicate=True, timestamp=time.time())
        SkipBlock.LogFile.buffer(lbracket)
        WriteBlock.pda.append(lbracket)
        return lbracket.predicate

    @staticmethod
    def end(*args, values=None):
        lbracket = WriteBlock.pda.pop()
        running_time = lbracket.timestamp - time.time()
        if not args:
            rbracket = Bracket(lbracket.sk, lbracket.gk, RBRACKET)
            SkipBlock.LogFile.buffer(rbracket)
        else:
            for arg in args:
                data_record = val_to_record(arg, lbracket)
                SkipBlock.LogFile.buffer(data_record)


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
    tree = None

    @staticmethod
    def step_into(block_name: str, probed=False):
        assert isinstance(block_name, str)
        dynamic_id = ReadBlock.dynamic_identifiers.get(block_name, 0)
        ReadBlock.dynamic_identifiers[block_name] = dynamic_id + 1

        lbracket = Bracket(block_name, dynamic_id, LBRACKET, predicate=probed)
        ReadBlock.pda.append(lbracket)
        return lbracket.predicate

    @staticmethod
    def end(*args, values=None):
        lbracket = ReadBlock.pda.pop()
        block = ReadBlock.tree.hash[lbracket.sk][lbracket.gk]
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
        if values is None:
            return


class SkipBlock(SeemBlock):
    LogFile = None

    @staticmethod
    def step_into(block_name: str, probed=False):
        raise RuntimeError("SkipBlock missing dynamic linking")

    @staticmethod
    def end(*args, values=None):
        raise RuntimeError("SkipBlock missing dynamic linking")

    @staticmethod
    def bind():
        SkipBlock.LogFile = File(florin.get_index())
        if flags.REPLAY:
            SkipBlock.step_into = ReadBlock.step_into
            SkipBlock.end = ReadBlock.end
            SkipBlock.LogFile.read()
            ReadBlock.tree = SkipBlock.LogFile.parse()
        else:
            SkipBlock.step_into = WriteBlock.step_into
            SkipBlock.end = WriteBlock.end


__all__ = ['SkipBlock']
