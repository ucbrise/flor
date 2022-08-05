import json
from typing import Dict, Optional, Tuple, Union
from flor import shelf

import sys
from pathlib import PurePath, Path
from .constants import *
from .pin import kvs


NAME: Optional[str] = None
REPLAY: bool = False
INDEX: Optional[PurePath] = None
MODE: Optional[Union[REPLAY_MODE, RECORD_MODE]] = None
PID: REPLAY_PARALLEL = REPLAY_PARALLEL(1, 1)
EPSILON: float = 1 / 15
RESUMING: bool = False

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
        index_exists = shelf.verify(PurePath(index).name)
        INDEX = Path.home() / ".flor" / NAME / Path(index).name
        if not index_exists:
            print("MISSING CACHE: Recomputing Checkpoints")
            REPLAY = False
            MODE = RECORD_MODE.chkpt_restore
    if mode is not None:
        MODE = REPLAY_MODE[mode]
    if pid is not None:
        assert isinstance(pid, tuple) and len(pid) == 2
        p, n = pid
        assert isinstance(p, int) and isinstance(n, int)
        assert p >= 1
        assert n >= 1
        assert p <= n
        PID = REPLAY_PARALLEL(*pid)


class Parser:
    """
    --flor NAME [EPSILON]
    --replay_flor [weak | strong] [i/n]
    """

    @staticmethod
    def _parse_name():
        assert (
            "--replay_flor" not in sys.argv
        ), "Pick at most one of `--flor` or `--replay_flor` but not both"
        global NAME, EPSILON
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
                flor_flags or Path(FLORFILE).exists()
            ), "Missing NAME argument in --flor NAME"
            for flag in flor_flags:
                if flag[0:2] == "0.":
                    EPSILON = float(flag)
                else:
                    NAME = flag
            if Path(FLORFILE).exists():
                with open(FLORFILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                assert "NAME" in d
                NAME = d["NAME"] if NAME is None else NAME

    @staticmethod
    def _parse_replay():
        global names_in_replay
        assert (
            "--flor" not in sys.argv
        ), "Pick at most one of `--flor` or `--replay_flor` but not both"
        try:
            with open(FLORFILE, "r", encoding="utf-8") as f:
                d = json.load(f)
        except FileNotFoundError:
            print("No replay file, did you record first?")
            raise
        assert "NAME" in d, "check your `.replay.json` file. Missing name."
        if "KVS" in d:
            kvs.update(
                {
                    k: v
                    for k, v in d["KVS"].items()
                    if "." not in k or k.split(".")[1] == "a"
                }
            )
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
            set_REPLAY(d["NAME"], index=d.get("MEMO", None), mode=mode, pid=pid)

    @staticmethod
    def parse():
        if "--flor" in sys.argv:
            Parser._parse_name()
        elif "--replay_flor" in sys.argv:
            Parser._parse_replay()

    @staticmethod
    def from_string(s: str):
        sys.argv = s.split(" ")
        Parser.parse()


from_string = Parser.from_string

__all__ = [
    "NAME",
    "REPLAY",
    "INDEX",
    "MODE",
    "PID",
    "EPSILON",
    "set_REPLAY",
    "Parser",
    "from_string",
]
