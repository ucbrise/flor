from .bracket import Bracket
from .eof import EOF
from typing import Union

md_entry = Union[Bracket, EOF]

__all__ = ["Bracket", "EOF", "md_entry"]
