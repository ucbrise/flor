from .constants import *
from .api import *
from . import cli
from . import versions

cli.parse_args()

if cli.flags.args is not None:
    if cli.flags.args.flor_command is None:
        if versions.current_branch() is not None:
            versions.to_shadow()
