from .constants import *
from . import cli

import sys

if "ipykernel" not in sys.modules:
    cli.parse_args()

from .api import *
from .repl import query, pivot, replay
from . import utils