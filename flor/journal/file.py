from ..shelf import florin
from .entry import DataVal, DataRef, Bracket, EOF, make_entry

import json
import pathlib
from typing import Union, List

entries: List[Union[DataRef, DataVal, Bracket, EOF]] = []


def read():
    with open(florin.get_index(), 'r') as f:
        for line in f:
            log_record = make_entry(json.loads(line.strip()))
            entries.append(log_record)


def feed(journal_entry: Union[DataRef, DataVal, Bracket, EOF]):
    entries.append(journal_entry)


def write():
    with open(florin.get_index(), 'w') as f:
        for log_record in entries:
            if isinstance(log_record, DataRef):
                log_record.set_ref_and_dump(florin.get_pkl_ref())
            f.write(json.dumps(log_record.jsonify()) + pathlib.os.linesep)
    entries[:] = []


def merge():
    """
    Stitch together parallel-written files
    """
    if florin.get_latest().exists():
        florin.get_latest().unlink()
    florin.get_latest().symlink_to(florin.get_index())


def close(eof_entry: EOF):
    feed(eof_entry)
    write()
    merge()


__all__ = ['read', 'feed', 'write', 'close', 'entries']
