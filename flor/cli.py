import json
import argparse
from argparse import Namespace
from typing import Any, Dict, Optional

from dataclasses import dataclass
from .versions import current_branch, to_shadow


@dataclass
class Flags:
    hyperparameters: Dict[str, str]
    queryparameters: Optional[Dict[str, str]]
    old_tstamp: Optional[str]
    args: Optional[Any]


flags = Flags({}, None, None, None)


def parse_replay_flor(arg):
    parts = arg.split()
    return {
        p.split("=")[0]: eval(p.split("=")[1])
        if "::" not in p
        else str(p.split("=")[1])
        for p in parts
    }


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

    # Apply command
    apply_parser = flor_parser.add_parser("apply")
    apply_parser.add_argument(
        "dp_list", nargs="*", help="The variable-length list of dp_str and dp values"
    )
    apply_parser.add_argument("train_file", help="The train.py file")

    # Query command
    query_parser = flor_parser.add_parser("query")
    query_parser.add_argument(
        "q", type=str, help="SQL query to execute on the database"
    )

    # Existing code
    args = parser.parse_args()
    flags.args = args

    if args.replay_flor is not None:
        flags.queryparameters = {}
        for d in args.replay_flor:
            flags.queryparameters.update(d)
        replay_initialize()

    # Process the key-value pair arguments
    if args.kwargs is not None:
        if not args.kwargs:
            raise RuntimeError("--kwargs called but no arguments added")
        for kwarg in args.kwargs:
            key, value = kwarg.split("=")
            flags.hyperparameters[key] = value

    return flags


def in_replay_mode():
    if cond := flags.queryparameters is not None:
        pass
    return cond


def replay_initialize():
    assert (
        flags.args is not None and flags.args.kwargs is None
    ), "Cannot set --kwargs in replay, would rewrite history"
    # TODO: Validate flor.queryparameters

    # update flags.hyperparameters
    with open(".flor.json", "r") as f:
        data = json.load(f)
    for obj in data:
        if len(obj) == 2:
            d = {obj["value_name"]: obj["value"]}
            flags.hyperparameters.update(d)
    flags.old_tstamp = data[-1]["TSTAMP"]
