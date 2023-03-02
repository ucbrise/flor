from . import flags
from .logger import log, pinned
from .kits import MTK
from .query import *
from .hlast import apply

from .constants import *

flags.Parser.parse()
