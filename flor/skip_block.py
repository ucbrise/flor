from record import DataRef, DataVal, Bracket, LBRACKET, RBRACKET
from copy import deepcopy
from file import File

from typing import List, Union
from abc import ABC, abstractmethod


class SeemBlock(ABC):
    @staticmethod
    @abstractmethod
    def step_into(block_name: str, probed=False):
        ...

    @staticmethod
    @abstractmethod
    def end(*args):
        ...


class WriteBlock(SeemBlock):
    dynamic_identifiers = dict()
    pda: List[Bracket] = []

    @staticmethod
    def step_into(block_name: str, probed=False):
        assert isinstance(block_name, str)
        dynamic_id = WriteBlock.dynamic_identifiers.get(block_name, 0)
        WriteBlock.dynamic_identifiers[block_name] = dynamic_id + 1

        lbracket = Bracket(block_name, dynamic_id, LBRACKET)
        LogFile.buffer(lbracket)
        WriteBlock.pda.append(lbracket)

    @staticmethod
    def end(*args):
        lbracket = WriteBlock.pda.pop()
        if not args:
            rbracket = Bracket(lbracket.sk, lbracket.gk, RBRACKET)
            LogFile.buffer(rbracket)
        else:
            for arg in args:
                data_record = val_to_record(arg, lbracket)
                LogFile.buffer(data_record)


def val_to_record(arg, lbracket: Bracket) -> Union[DataRef, DataVal]:
    if hasattr(arg, 'state_dict'):
        arg = getattr(arg, 'state_dict')()

    if type(arg) in [type(None), int, float, bool, str]:
        return DataVal(lbracket.sk, lbracket.gk, arg)
    else:
        return DataRef(lbracket.sk, lbracket.gk, deepcopy(arg))


class ReadBlock(SeemBlock):
    @staticmethod
    def step_into(block_name: str, probed=False):
        ...

    @staticmethod
    def end(*args):
        ...


SkipBlock = WriteBlock if '...' else ReadBlock
LogFile: File = File('...')

__all__ = ['SkipBlock', 'LogFile']
