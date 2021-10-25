from .seemblock import SeemBlock
from .readblock import ReadBlock
from .writeblock import WriteBlock

from flor import flags


class SkipBlock(SeemBlock):
    @staticmethod
    def step_into(block_name: str, probed=None):
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
        SkipBlock.step_into = block.step_into  # type: ignore
        SkipBlock.end = block.end
