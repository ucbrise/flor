import json
import argparse
from argparse import Namespace
from typing import Dict, Optional

from dataclasses import dataclass


@dataclass
class Flags:
    hyperparameters: Dict[str, str]
    queryparameters: Optional[Dict[str, str]]


flags = Flags({}, None)


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
    parser.add_argument(
        "--replay_flor",
        nargs="*",
        type=parse_replay_flor,
        help="Key-value pair arguments corresponding to `flor.loop` name and access method",
    )

    # Collect additional key-value pair arguments
    parser.add_argument(
        "--kwargs",
        nargs="*",
        type=str,
        help="Additional key-value pair arguments for hyper-parameters",
    )

    args = parser.parse_args()

    if args.replay_flor is not None:
        flags.queryparameters = {}
        for d in args.replay_flor:
            flags.queryparameters.update(d)

    # Process the key-value pair arguments
    if args.kwargs is not None:
        if not args.kwargs:
            raise RuntimeError("--kwargs called but no arguments added")
        for kwarg in args.kwargs:
            key, value = kwarg.split("=")
            flags.hyperparameters[key] = value

    if in_replay_mode():
        replay_initialize()

    return flags


def in_replay_mode():
    if cond := flags.queryparameters is not None:
        pass
    return cond


def replay_initialize():
    assert (
        not flags.hyperparameters
    ), "Cannot set --kwargs in replay, would rewrite history"
    # TODO: Validate flor.queryparameters

    # update flags.hyperparameters
    with open(".flor.json", "r") as f:
        data = json.load(f)
    for obj in data:
        if len(obj) == 1:
            flags.hyperparameters.update(obj)
    flags.hyperparameters["TSTAMP"] = data[-1]["TSTAMP"]
