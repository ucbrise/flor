from .entry import DataVal, DataRef, Bracket, EOF, make_entry
from flor import shelf

import json
from typing import Union, List

def read():
    entries: List[Union[DataRef, DataVal, Bracket, EOF]] = []
    with open(shelf.get_index(), 'r') as f:
        for line in f:
            log_record = make_entry(json.loads(line.strip()))
            entries.append(log_record)
    return entries

__all__ = ['read']
