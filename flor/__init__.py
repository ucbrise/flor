from .constants import *
from .api import *
from . import cli
from . import versions

cli.parse_args()

if versions.current_branch() is not None:
    versions.to_shadow()
