from pathlib import Path
import flags

_p = Path.home()
_p = _p / '.flor'
_p.mkdir(exist_ok=True)
del _p, Path

flags.Parser.parse_name()
flags.Parser.parse_replay()


__all__ = ['flags']

