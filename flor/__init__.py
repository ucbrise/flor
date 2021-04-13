from . import flags
from .iterator import it
from .skipblock import SkipBlock
from .logger import Logger

flags.Parser.parse_name()
flags.Parser.parse_replay()


__all__ = ['flags', 'it', 'SkipBlock', 'Logger']

