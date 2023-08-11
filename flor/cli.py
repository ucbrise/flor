import argparse
from argparse import Namespace
from typing import Dict, Optional

from dataclasses import dataclass


@dataclass
class Flags:
    hyperparameters: Dict[str, str]
    queryparameters: Dict[str, str]


flags = Flags({}, {})


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

    flags.queryparameters = args.replay_flor

    # Process the key-value pair arguments
    if args.kwargs is not None:
        for kwarg in args.kwargs:
            key, value = kwarg.split("=")
            flags.hyperparameters[key] = value
        else:
            raise RuntimeError("--kwargs called but no arguments added")

    if in_replay_mode():
        replay_initialize()

    return flags


def in_replay_mode():
    if cond := flags.queryparameters is not None:
        print("FLOR REPLAY MODE", str(flags.queryparameters))
    else:
        print("FLOR RECORD MODE")
    return cond


def replay_initialize():
    pass
