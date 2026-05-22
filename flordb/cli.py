import argparse
import glob
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from .versions import current_branch, to_shadow
from .constants import RUNS_DIR
from . import orm
import sys

from .hlast.visitors import WithExpVisitor, ResumeBlockVisitor
import ast


IterKind = Literal["all", "last", "none", "indices"]


@dataclass(frozen=True)
class IterSpec:
    """How a single flor.loop should be narrowed during replay.

    kind:
      - "all"     -> every iteration
      - "last"    -> only the final iteration (default for unmentioned loops)
      - "none"    -> skip the loop entirely
      - "indices" -> only the iterations listed in `indices`
    """
    kind: IterKind
    indices: Tuple[int, ...] = ()


DEFAULT_ITER_SPEC = IterSpec("last")


def parse_iter_spec(spec: str) -> IterSpec:
    s = spec.strip()
    if s == "all":
        return IterSpec("all")
    if s == "last":
        return IterSpec("last")
    if s in ("none", ""):
        return IterSpec("none")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        idxs = tuple(sorted({int(p) for p in parts}))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--iter spec must be one of: all, last, none, or a comma list of ints; got {spec!r}"
        )
    return IterSpec("indices", idxs)


def parse_iter_arg(s: str) -> Tuple[str, IterSpec]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--iter expects NAME=SPEC (e.g. epoch=0,2 or step=all); got {s!r}"
        )
    name, spec = s.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--iter NAME cannot be empty; got {s!r}")
    return name, parse_iter_spec(spec)


def parse_override_arg(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--override expects KEY=VALUE; got {s!r}"
        )
    k, v = s.split("=", 1)
    k = k.strip()
    if not k:
        raise argparse.ArgumentTypeError(f"--override KEY cannot be empty; got {s!r}")
    return k, v


def parse_apply_vars(s: str) -> List[str]:
    return [tok.strip() for tok in s.split(",") if tok.strip()]


@dataclass
class Flags:
    hyperparameters: Dict[str, str] = field(default_factory=dict)
    # Replay-mode state. All four are populated only when --replay_flor is set.
    replay_flor: bool = False
    apply_vars: Optional[List[str]] = None     # --apply VARS (None = no projection)
    iter_specs: Dict[str, IterSpec] = field(default_factory=dict)  # --iter NAME=SPEC
    overrides: Dict[str, str] = field(default_factory=dict)        # --override K=V
    wev_found: bool = False                    # WithExpVisitor result from the script
    old_tstamp: Optional[str] = None
    resume_spec: Optional["ResumeSpec"] = None
    # CLI plumbing.
    args: Optional[Any] = None
    columns: Optional[Tuple[str, ...]] = None


@dataclass
class ResumeSpec:
    path: str
    lhs_name: str
    applies: list  # list[tuple[str, str]] — (target_name, key)


flags = Flags()


def parse_columns(column_string):
    return [str(each) for each in column_string.split()]


def parse_args():
    parser = argparse.ArgumentParser(description="FlorDB CLI")

    parser.add_argument(
        "--replay_flor",
        action="store_true",
        help="Enable replay mode. Use --apply / --iter / --override to refine.",
    )
    parser.add_argument(
        "--apply",
        type=parse_apply_vars,
        default=None,
        help=(
            "Comma-separated projection of log names (or @LINENO) to recompute. "
            "When omitted, every flor.log fires."
        ),
    )
    parser.add_argument(
        "--iter",
        dest="iter_specs",
        action="append",
        default=[],
        type=parse_iter_arg,
        help=(
            "Per-loop narrowing: --iter NAME=SPEC. "
            "SPEC ∈ {all, last, none, comma-sep int list}. Repeatable."
        ),
    )
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        default=[],
        type=parse_override_arg,
        help=(
            "Replay-time override of an env-shaped flor.arg (e.g. device=cpu). "
            "Cannot override hyperparameters logged by the historical run. Repeatable."
        ),
    )
    parser.add_argument(
        "--kwargs",
        nargs="*",
        type=str,
        help="Forward-run hyperparameter overrides (k=v). Disallowed under --replay_flor.",
    )

    # flordb subcommands
    flor_parser = parser.add_subparsers(dest="flor_command")

    unpack_parser = flor_parser.add_parser("unpack")
    stat_parser = flor_parser.add_parser("stat")

    replay_parser = flor_parser.add_parser(
        "replay", help="Replay with specified VARS over historical runs"
    )
    replay_parser.add_argument(
        "VARS",
        type=lambda s: s.split(","),
        help="List of logged variables (or @LINENO), comma-separated",
    )
    replay_parser.add_argument(
        "where_clause", nargs="?", type=str, help="Optional SQL WHERE clause"
    )
    replay_parser.add_argument(
        "--iter",
        dest="narrow_iters",
        action="append",
        default=[],
        type=parse_iter_arg,
        help=(
            "Override the auto-narrowing the orchestrator would pick. "
            "Repeatable: --iter epoch=2 --iter step=none."
        ),
    )

    query_parser = flor_parser.add_parser("query")
    query_parser.add_argument(
        "q", type=str, help="SQL query to execute on the database"
    )

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
        "--apply",
        "--iter",
        "--override",
        "unpack",
        "replay",
        "query",
        "dataframe",
        "stat",
    ]

    if any(command in sys.argv for command in flor_commands):
        args = parser.parse_args()
        flags.args = args
    else:
        flags.args = None

    if flags.args is not None and getattr(flags.args, "replay_flor", False):
        flags.replay_flor = True
        flags.apply_vars = flags.args.apply
        flags.iter_specs = dict(flags.args.iter_specs or [])
        flags.overrides = dict(flags.args.overrides or [])
        replay_initialize()
    else:
        # --apply / --iter / --override without --replay_flor is a user error.
        if flags.args is not None and (
            flags.args.apply or flags.args.iter_specs or flags.args.overrides
        ):
            raise RuntimeError(
                "--apply / --iter / --override require --replay_flor to be set."
            )

    if flags.args is not None and flags.args.kwargs is not None:
        if not flags.args.kwargs:
            raise RuntimeError("--kwargs called but no arguments added")
        for kwarg in flags.args.kwargs:
            key, value = kwarg.split("=")
            flags.hyperparameters[key] = value

    if flags.args is not None and flags.args.flor_command == "dataframe":
        flags.columns = flags.args.columns

    return flags


def in_replay_mode() -> bool:
    return flags.replay_flor


def iter_spec_for(name: str) -> IterSpec:
    """Look up the per-loop spec; unmentioned loops fall back to DEFAULT_ITER_SPEC.

    On first default-resolve for a given loop name, prints a one-line tip so
    users discover the explicit verbs without surprise behavior.
    """
    spec = flags.iter_specs.get(name)
    if spec is not None:
        return spec
    if name not in _defaulted_loops:
        _defaulted_loops.add(name)
        print(
            f"FLOR: --iter {name}=... not given; defaulting to 'last'. "
            f"Use --iter {name}=all (or =none, =0,2,...) to override."
        )
    return DEFAULT_ITER_SPEC


_defaulted_loops: set = set()


# Env-shaped flor.arg names that may be overridden at replay time without
# invalidating results. Keep this small and explicit.
ENV_OVERRIDE_ALLOWLIST: set = {"device", "ckpt_interval_s"}


def replay_initialize():
    if flags.args is not None and flags.args.kwargs:
        raise RuntimeError(
            "Cannot combine --kwargs with --replay_flor; use --override KEY=VALUE "
            "for env-shaped knobs (device, ckpt_interval_s, ...)."
        )
    jsonl_paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*.jsonl")))
    assert jsonl_paths, f"No runs found in {RUNS_DIR}; cannot initialize replay."
    latest = jsonl_paths[-1]
    data = orm.read_jsonl(latest)
    filename = data[0]["filename"]

    with open(filename, "r") as f:
        tree = ast.parse(f.read())
    wev = WithExpVisitor()
    wev.visit(tree)
    flags.wev_found = bool(wev.found)

    rbv = ResumeBlockVisitor()
    rbv.visit(tree)
    if rbv.found:
        flags.resume_spec = ResumeSpec(
            path=rbv.path,  # type: ignore[arg-type]
            lhs_name=rbv.lhs_name,  # type: ignore[arg-type]
            applies=list(rbv.applies),
        )
    elif rbv.unscoped_match:
        print(
            "FLOR: torch.load resume pattern found outside module scope; "
            "auto-restore disabled. Use flor.checkpointing(...) for replay."
        )

    historical_hps: Dict[str, str] = {}
    for obj in data:
        if obj["ctx"] is None and obj["type"] == 1:
            historical_hps[obj["name"]] = obj["value"]

    # Allowlist of env-shaped knobs that are always safe to override during
    # replay. Anything else, if it was logged historically, gets a loud
    # warning -- overriding a real hyperparameter usually invalidates results.
    bad = [
        k for k in flags.overrides
        if k in historical_hps and k not in ENV_OVERRIDE_ALLOWLIST
    ]
    if bad:
        raise RuntimeError(
            f"--override rejected for {bad!r}: these were logged as flor.arg in the "
            f"historical run. Re-run forward with new values, or add the key to "
            f"flordb.cli.ENV_OVERRIDE_ALLOWLIST if it is truly env-shaped."
        )

    flags.hyperparameters.update(historical_hps)
    flags.hyperparameters.update(flags.overrides)
    flags.old_tstamp = data[0]["tstamp"]
