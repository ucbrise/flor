from . import flags
from .kits import MTK, DPK, load_kvs

flags.Parser.parse()

__all__ = ["MTK", "DPK", "flags", "load_kvs"]
