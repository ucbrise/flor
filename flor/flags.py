from flor.shelf import home_shelf, cwd_shelf
from flor.constants import *
from flor.state import State
from flor.logger import exp_json
from flor.query.database import start_db

import sys
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import PurePath, Path
from time import time

from argparse import ArgumentParser, HelpFormatter, Namespace


NAME: Optional[str] = None
REPLAY: bool = False
INDEX: Optional[PurePath] = None
MODE: Optional[REPLAY_MODE] = None
PID: REPLAY_PARALLEL = REPLAY_PARALLEL(1, 1)
EPSILON: float = 1 / 15
RESUMING: bool = False

DATALOGGING = True

"""
--flor NAME [EPSILON]
--replay_flor [weak | strong] [i/n]
"""


def set_REPLAY(
    name: str,
    index: Optional[str] = None,
    mode: Optional[str] = None,
    pid: Optional[Tuple[int, int]] = None,
):
    """
    When set: enables FLOR REPLAY
    """
    global NAME, REPLAY, INDEX, MODE, PID
    NAME = name
    REPLAY = True
    MODE = REPLAY_MODE.weak
    if index is not None:
        assert isinstance(index, str)
        assert PurePath(index).suffix == ".json"
        assert home_shelf.verify(PurePath(index).name)
        INDEX = Path(index)
    if mode is not None:
        MODE = REPLAY_MODE[mode]
    if pid is not None:
        assert isinstance(pid, tuple) and len(pid) == 2
        p, n = pid
        assert isinstance(p, int) and isinstance(n, int)
        assert p >= 0
        assert n >= 1
        assert p <= n
        PID = REPLAY_PARALLEL(*pid)


class CLI_Args(ArgumentParser):
    def __init__(self) -> None:
        super().__init__(
            prog=None,
            usage=None,
            description=None,
            epilog="Arguments parsed by Flor",
            parents=[],
            formatter_class=HelpFormatter,
            prefix_chars="-",
            fromfile_prefix_chars=None,
            argument_default=None,
            conflict_handler="error",
            add_help=False,
            allow_abbrev=False,
            exit_on_error=True,
        )
        self.nsp = Namespace()

    def parse_record(self):
        assert (
            "--replay_flor" not in sys.argv
        ), "Pick at most one of `--flor` or `--replay_flor` but not both"
        assert (
            cwd_shelf.in_shadow_branch()
        ), "Please invoke --flor from a `flor.shadow` branch."
        global NAME, EPSILON, DATALOGGING
        flor_flags = []
        feeding = False
        for _ in range(len(sys.argv)):
            arg = sys.argv.pop(0)

            if arg == "--flor":
                feeding = True
            elif arg[0] == "-":
                feeding = False

            if feeding:
                flor_flags.append(arg)
            else:
                sys.argv.append(arg)

        assert len(flor_flags) <= 3
        if flor_flags:
            assert flor_flags.pop(0) == "--flor"
            assert (
                flor_flags or exp_json.exists()
            ), "Missing NAME argument in --flor NAME"
            if exp_json.exists():
                exp_json.deferred_init()
            for flag in flor_flags:
                if flag[0:2] == "0.":
                    EPSILON = float(flag)
                else:
                    NAME = flag
            if NAME is None:
                assert exp_json.exists()
                NAME = exp_json.get("NAME")  # take from past
        assert NAME is not None
        if exp_json.exists() and exp_json.get("NAME") == NAME:
            # IF previous name is same as this name
            DATALOGGING = False
        exp_json.put("NAME", NAME)
        home_shelf.mk_job(cwd_shelf.get_projid())

    def parse_replay(self):
        assert (
            "--flor" not in sys.argv
        ), "Pick at most one of `--flor` or `--replay_flor` but not both"
        assert (
            cwd_shelf.in_shadow_branch()
        ), "Please invoke --replay_flor from a `flor.shadow` branch."
        try:
            assert exp_json.exists()
            exp_json.deferred_init()
        except FileNotFoundError:
            print("No replay file, did you record first?")
            raise
        assert exp_json.get("NAME"), f"check your `{LOG_RECORDS}` file. Missing name."
        flor_flags = []
        feeding = False
        for _ in range(len(sys.argv)):
            arg = sys.argv.pop(0)

            if arg == "--replay_flor":
                feeding = True
            elif arg[0] == "-":
                feeding = False

            if feeding:
                flor_flags.append(arg)
            else:
                sys.argv.append(arg)

        assert len(flor_flags) <= 3
        if flor_flags:
            assert flor_flags.pop(0) == "--replay_flor"
            mode = None
            pid = None
            for flor_flag in flor_flags:
                if flor_flag in [each.name for each in REPLAY_MODE]:
                    mode = flor_flag
                elif len(flor_flag.split("/")) == 2:
                    p, n = flor_flag.split("/")
                    pid = int(p), int(n)
                else:
                    raise RuntimeError(
                        "Invalid argument passed to --replay_flor"
                        + "[weak | strong] [I/N]"
                    )
            projid = cwd_shelf.get_projid()
            set_REPLAY(
                str(exp_json.get("NAME")),
                index=exp_json.get("MEMO"),
                mode=mode,
                pid=pid,
            )
            home_shelf.set_job(projid)
            start_db(projid)

    def parse_remaining(self):
        skip = True
        for flg in [each for each in sys.argv if each[0:2] == "--"]:
            skip = False
            self.add_argument(flg)
        if not skip:
            _snapshot = list(sys.argv)
            self.nsp, _ = self.parse_known_args()
            sys.argv[:] = _snapshot
        elif REPLAY:
            # TODO: Replay case, load CLI from .flor/.replay.json
            pass
            


parser = CLI_Args()


def is_interactive():
    try:
        get_ipython()  # type: ignore
        return True
    except:
        return False


def parse():
    State.import_time = time()
    if "--flor" in sys.argv:
        parser.parse_record()
    elif "--replay_flor" in sys.argv:
        parser.parse_replay()
    if not is_interactive():
        parser.parse_remaining()


__all__ = [
    "NAME",
    "REPLAY",
    "INDEX",
    "MODE",
    "PID",
    "EPSILON",
    "DATALOGGING",
    "set_REPLAY",
    "parse",
    "parser",
    "CLI_Args",
]
