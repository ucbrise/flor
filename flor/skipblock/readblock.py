from .seemblock import SeemBlock

from flor import flags
from flor.journal.entry import Bracket, LBRACKET

from typing import List


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
        block = ReadBlock.journal.as_tree()[lbracket.sk].blocks[lbracket.gk]
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
