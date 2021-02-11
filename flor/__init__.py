import flags
from interface import it, SkipBlock
from pathlib import Path


flags.Parser.parse_name()
flags.Parser.parse_replay()


__all__ = ['flags', 'it', 'SkipBlock']

