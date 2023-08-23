from .constants import *
from .api import *
from . import cli

import sys

if "ipykernel" not in sys.modules:
    cli.parse_args()

from . import database
from .hlast import backprop
