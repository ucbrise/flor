import json
import argparse
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple

from dataclasses import dataclass
from .versions import current_branch, to_shadow
import sys


@dataclass
class Flags:
    hyperparameters: Dict[str, str]
    queryparameters: Optional[Dict[str, str]]
    old_tstamp: Optional[str]
    args: Optional[Any]
    columns: Optional[Tuple[str]]


flags = Flags({}, None, None, None, None)


def parse_replay_flor(arg):
    parts = arg.split()
    return {
        p.split("=")[0]: eval(p.split("=")[1])
        if "::" not in p
        else str(p.split("=")[1])
        for p in parts
    }


def parse_columns(column_string):
    return [str(each) for each in column_string.split()]


def parse_args():
    parser = argparse.ArgumentParser(description="FlorDB CLI")

    # Existing code
    parser.add_argument(
        "--replay_flor",
        nargs="*",
        type=parse_replay_flor,
        help="Key-value pair arguments corresponding to `flor.loop` name and access method",
    )
    parser.add_argument(
        "--kwargs",
        nargs="*",
        type=str,
        help="Additional key-value pair arguments for hyper-parameters",
    )

    # Flor module subparser
    flor_parser = parser.add_subparsers(dest="flor_command")

    # Unpack command
    unpack_parser = flor_parser.add_parser("unpack")
    stat_parser = flor_parser.add_parser("stat")

    # Replay command
    replay_parser = flor_parser.add_parser(
        "replay", help="Replay with specified VARS and optional QUERY"
    )
    replay_parser.add_argument(
        "VARS",
        type=lambda s: s.split(","),
        help="List of logged variables, or linenos, comma-separated",
    )
    replay_parser.add_argument(
        "where_clause", nargs="?", type=str, help="Optional SQL WHERE clause"
    )

    # Query command
    query_parser = flor_parser.add_parser("query")
    query_parser.add_argument(
        "q", type=str, help="SQL query to execute on the database"
    )

    # Pivot command
    pivot_parser = flor_parser.add_parser("dataframe")
    pivot_parser.add_argument(
        "columns",
        nargs="?",
        type=lambda s: s.split(","),
        help="List of logged variables, comma-separated",
    )


    flor_commands = [
        "--kwargs",
        "--replay_flor",
        "unpack",
        "replay",
        "query",
        "dataframe",
        "stat"
    ]

    # Check if any of the flor_commands is in the arguments
    if any(command in sys.argv for command in flor_commands):
        args = parser.parse_args()
        flags.args = args
    else:
        flags.args = None

    if flags.args is not None and flags.args.replay_flor is not None:
        flags.queryparameters = {}
        for d in flags.args.replay_flor:
            flags.queryparameters.update(d)
        replay_initialize()

    # Process the key-value pair arguments
    if flags.args is not None and flags.args.kwargs is not None:
        if not flags.args.kwargs:
            raise RuntimeError("--kwargs called but no arguments added")
        for kwarg in flags.args.kwargs:
            key, value = kwarg.split("=")
            flags.hyperparameters[key] = value

    if flags.args is not None and flags.args.flor_command == "dataframe":
        flags.columns = flags.args.columns

    return flags


def in_replay_mode():
    return flags.queryparameters is not None


def replay_initialize():
    assert (
        flags.args is not None and flags.args.kwargs is None
    ), "Cannot set --kwargs in replay, would rewrite history"
    # update flags.hyperparameters
    with open(".flor.json", "r") as f:
        data = json.load(f)
    for obj in data:
        if obj["loop"] is None and obj["type"] == 1:
            d = {obj["name"]: obj["value"]}
            flags.hyperparameters.update(d)
    flags.old_tstamp = data[0]["tstamp"]
