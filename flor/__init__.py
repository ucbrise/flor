from . import flags
from .utils import cross_prod
from .batch import batch

flags.parse()

__all__ = ["flags", "batch"]