from .entry import DataVal, DataRef, Bracket, EOF, make_entry
from flor import shelf

import json
from typing import Union, List


def read() -> List[Union[DataRef, DataVal, Bracket, EOF]]:
    entries: List[Union[DataRef, DataVal, Bracket, EOF]] = []
    index = shelf.get_index()
    if index is not None:
        with open(index, "r") as f:
            for line in f:
                log_record = make_entry(json.loads(line.strip()))
                entries.append(log_record)
        return entries
    raise RuntimeError("Shelf not initialized. Did you call shelf.mk_job?")


__all__ = ["read"]
