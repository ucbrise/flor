import argparse
import glob
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from .versions import to_shadow
from .constants import RUNS_DIR
from . import orm
import sys

from .hlast.visitors import (
    WithExpVisitor,
    ResumeBlockVisitor,
    RestoreSignalVisitor,
    SaveShapeVisitor,
)
from .capture import flor_print
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
    # Historical flor.arg values from the replayed run's JSONL, JSON-typed.
    # Kept separate from `hyperparameters` (which --override writes strings
    # into) so api.arg can cast an override to the type it is replacing.
    historical_args: Dict[str, Any] = field(default_factory=dict)
    wev_found: bool = False                    # WithExpVisitor result from the script
    old_tstamp: Optional[str] = None
    resume_spec: Optional["ResumeSpec"] = None
    # CLI plumbing.
    args: Optional[Any] = None
    columns: Optional[Tuple[str, ...]] = None


@dataclass
class ResumeSpec:
    path: str
    lhs_name: Optional[str]
    applies: list  # list[tuple[str, Optional[str]]] — (target_name, key)
    # "ast" when ResumeBlockVisitor inferred this from the script's resume
    # block, "save" when SaveShapeVisitor inferred it from the torch.save that
    # writes the file, "explicit" when the user declared it with
    # flor.restore(...). Inference is a guess about what the user's code means;
    # a declaration is not. Only a guess needs the frame lookup, and only a
    # guess can be wrong in a way worth warning about. "ast" additionally needs
    # the neutralization dance -- it is the only source that implies the script
    # loads the file at module scope.
    source: str = "ast"
    # Source line the inference came from, for error messages. Set for "save",
    # where the mapping is a step removed from what the script actually says.
    lineno: Optional[int] = None
    # Live objects keyed by target name, set only for explicit specs. Their
    # presence is what lets flor.restore skip resolving names out of the user's
    # stack frame -- the failure mode that silently skipped renamed targets.
    targets: Optional[dict] = None


flags = Flags()


def parse_columns(column_string):
    return [str(each) for each in column_string.split()]


def _argv_mentions(argv: List[str], commands: List[str]) -> bool:
    """True if argv contains any of `commands`, in bare or `--flag=value` form.

    Bare-token matching alone would let `python train.py --apply=loss` slip
    through unparsed: the script would silently run forward instead of
    reporting that --apply requires --replay_flor.
    """
    for tok in argv:
        if tok in commands:
            return True
        if tok.startswith("-") and "=" in tok and tok.split("=", 1)[0] in commands:
            return True
    return False


def _render_ctx(ctx) -> str:
    """`epoch=0`, or `epoch=0/batch=7` for a nested loop. Empty when unindexed."""
    if not ctx:
        return ""
    return "/".join(
        f"{seg['name']}={seg['iteration']}"
        for seg in ctx
        if isinstance(seg, dict) and seg.get("iteration") is not None
    )


def report_extractions(derived, io_count: int, limit: int, wrote: bool) -> None:
    """Print what extraction found, in the same shape whether or not it wrote.

    Preview and `--extract` differ in one word and one closing line; keeping
    them in one function is what stops the dry run from drifting away from the
    thing it is supposed to be predicting.
    """
    if not derived:
        print(
            f"Nothing extractable from {io_count} captured line(s). "
            f"No metrics {'were added' if wrote else 'would be added'}."
        )
        return

    names = sorted({log.name for log, _ in derived})
    indexes = sorted(
        {
            seg["name"]
            for log, _ in derived
            for seg in (log.ctx or [])
            if isinstance(seg, dict)
        }
    )
    verb = "Extracted" if wrote else "Would extract"
    print(
        f"{verb} {len(derived)} value(s) across {len(names)} metric(s) from "
        f"{io_count} captured line(s): {', '.join(names)}"
    )
    if indexes:
        print(f"Indexed by: {', '.join(indexes)}")
    print()

    shown = derived[:limit]
    ctx_width = max((len(_render_ctx(log.ctx)) for log, _ in shown), default=0)
    name_width = max(len(log.name) for log, _ in shown)
    for log, line in shown:
        ctx = _render_ctx(log.ctx)
        print(
            f"  {ctx:<{ctx_width}}  {log.name:<{name_width}}  "
            f"{log.value:<12} <- {line}"
        )
    if len(derived) > limit:
        print(f"  ... {len(derived) - limit} more")

    if not wrote:
        print("\nNothing was written. Run again with --extract to write them.")
        return

    print(
        f"\nThey are columns now: "
        f"flor.dataframe({', '.join(repr(n) for n in names)}). "
        f"Re-run --extract any time to refresh the reading."
    )
    # The derivation lives in the local cache only, so a teammate reads the
    # same io and gets no columns until they run this themselves.
    print(
        "Written to the local cache. It is not committed: the captured text it "
        "was read from is, so a clone re-derives it with `flor capture "
        "--extract` rather than inheriting it."
    )


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
    # Positional VARS / WHERE are the pre-v4 spelling, kept working. New code
    # should use --apply / --where, which match the script-side surface
    # (`python train.py --replay_flor --apply ... --iter ... --override ...`).
    replay_parser.add_argument(
        "VARS",
        nargs="?",
        type=parse_apply_vars,
        help="Deprecated positional form of --apply.",
    )
    replay_parser.add_argument(
        "where_clause",
        nargs="?",
        type=str,
        help="Deprecated positional form of --where.",
    )
    replay_parser.add_argument(
        "--apply",
        dest="replay_apply",
        type=parse_apply_vars,
        default=None,
        help=(
            "Comma-separated log names (or @LINENO) to recompute over historical "
            "runs, e.g. --apply loss,val_acc."
        ),
    )
    replay_parser.add_argument(
        "--where",
        dest="replay_where",
        type=str,
        default=None,
        help="Optional SQL WHERE clause used for column discovery.",
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
    # Distinct dest from the top-level --override: this one is a passthrough
    # to the replayed child process, not a flag on this (orchestrator) run, so
    # it must not trip the "requires --replay_flor" check below.
    replay_parser.add_argument(
        "--override",
        dest="replay_overrides",
        action="append",
        default=[],
        type=parse_override_arg,
        help=(
            "Forwarded to each replayed run as --override KEY=VALUE "
            "(e.g. device=cpu). Repeatable."
        ),
    )

    capture_parser = flor_parser.add_parser(
        "capture", help="Inspect automatically captured print / logging output"
    )
    capture_parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Show the metrics --extract would pull out of the io already "
            "captured, and the line each came from. Writes nothing."
        ),
    )
    capture_parser.add_argument(
        "--extract",
        action="store_true",
        help=(
            "Read metrics out of captured text and add them to the metric "
            "table, so they appear in flor.dataframe(). Re-runs cleanly: each "
            "invocation replaces the previous extraction."
        ),
    )
    capture_parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Rows to show (default 40).",
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
        "capture",
    ]

    if _argv_mentions(sys.argv[1:], flor_commands):
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


def resolve_replay_args(args) -> Tuple[List[str], Optional[str]]:
    """Reconcile the two spellings of `flor replay`'s inputs.

    New:    flor replay --apply loss,val_acc --where "epoch > 2"
    Legacy: flor replay loss,val_acc "epoch > 2"

    Mixing them is rejected rather than guessed at: with --apply given, a bare
    positional would silently land in the VARS slot, so `flor replay --apply
    loss "epoch > 2"` would replay a variable named `epoch > 2`.
    """
    apply_vars = args.replay_apply
    where = args.replay_where
    if apply_vars is None:
        apply_vars = args.VARS
        if where is None:
            where = args.where_clause
    elif args.VARS:
        raise RuntimeError(
            f"flor replay: --apply was given, so the positional form is not "
            f"accepted (got {' '.join(args.VARS)!r}). Pass a filter as "
            f"--where instead."
        )
    if not apply_vars:
        raise RuntimeError(
            "flor replay: nothing to apply. Name the hindsight variables to "
            "recompute, e.g. `flor replay --apply loss,val_acc`."
        )
    return list(apply_vars), where


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
        flor_print(
            f"FLOR: --iter {name}=... not given; defaulting to 'last'. "
            f"Use --iter {name}=all (or =none, =0,2,...) to override."
        )
    return DEFAULT_ITER_SPEC


_defaulted_loops: set = set()


# Env-shaped flor.arg names that may be overridden at replay time without
# invalidating results. Keep this small and explicit.
ENV_OVERRIDE_ALLOWLIST: set = {"device", "ckpt_interval_s"}


def _describe_shape(path: str, applies: list) -> str:
    """How an inferred mapping reads in a message.

    Flat: `'ckpt.pth' holds net's state`.
    Keyed: `'ckpt.pth' maps 'model' -> net, 'optimizer' -> opt`.
    """
    if len(applies) == 1 and applies[0][1] is None:
        return f"{path!r} holds {applies[0][0]}'s state"
    pairs = ", ".join(f"{key!r} -> {name}" for name, key in applies)
    return f"{path!r} maps {pairs}"


def _why_the_save_site_was_no_help(ssv: SaveShapeVisitor) -> str:
    """The clause explaining what stopped save-site inference, for the warning."""
    if ssv.multi_path:
        return (
            " Its torch.save calls write more than one file, and replay "
            "addresses one file per run."
        )
    if ssv.conflicting_shape:
        return (
            f" Its torch.save calls disagree about what {ssv.path!r} holds, so "
            "only whichever ran last is true and flor can't tell which."
        )
    if ssv.unscoped_match:
        return (
            " Its torch.save sits in a helper function, whose locals name "
            "objects replay has no frame to reach."
        )
    return ""


def _infer_from_save_site(
    tree: ast.AST, filename: str, rsv: RestoreSignalVisitor
) -> SaveShapeVisitor:
    """Last resort: read the mapping off the torch.save that writes the file.

    Weaker evidence than a resume block, and wrong in a way the restore path
    cannot catch on its own. A run that saves `best_model` names an object that
    resolves, takes its state cleanly, and succeeds -- while the object it
    actually trains keeps whatever the fast-forward left in it, and the loop
    logs metrics off that as history. No exception is coming, so the mapping is
    announced instead of applied silently.
    """
    ssv = SaveShapeVisitor()
    ssv.visit(tree)
    if not ssv.found:
        return ssv
    if rsv.declared or rsv.enrolled:
        # The script already says how to restore, and says it at runtime:
        # flor.restore replaces whatever was inferred the moment it executes,
        # and flor.checkpointing owns its objects through a path of its own. A
        # guess read off the save site would be overwritten or redundant, and
        # announcing one would tell a user who did declare that flor guessed.
        return SaveShapeVisitor()
    flags.resume_spec = ResumeSpec(
        path=ssv.path,  # type: ignore[arg-type]
        lhs_name=None,
        applies=list(ssv.applies),
        source="save",
        lineno=ssv.lineno,
    )
    flor_print(
        f"FLOR: {filename} saves a checkpoint but never loads one back, so "
        f"flor read the layout off the torch.save at line {ssv.lineno}: "
        f"{_describe_shape(ssv.path, ssv.applies)}. Replay restores that every "
        f"iteration. Saving and restoring are not the same statement -- a run "
        f"that saves a copy (best_model, an EMA) rather than the object it "
        f"trains restores into the wrong one, and succeeds at it. If that is "
        f"this run, name the real target: "
        f"flor.restore({ssv.path!r}, <name>=<obj>, ...)."
    )
    return ssv


def _warn_if_nothing_restores(
    filename: str, rsv: RestoreSignalVisitor, ssv: SaveShapeVisitor
) -> None:
    """Say so when a script checkpoints but tells replay nothing about loading.

    The forward run happily mirrors every torch.save; it is only replay that
    needs to know which object each mirror belongs in. When neither inference
    nor a declaration supplies that, replay used to proceed in silence and
    reconstruct nothing -- reporting recomputed-from-the-wrong-state numbers as
    historical fact. The whole failure is invisible from the outside, so the
    warning is the only thing standing between the user and bad results.
    """
    if not rsv.torch_save:
        # Nothing was checkpointed through the piggy-back path, so there is
        # nothing for a resume block to restore. Enrollment-only scripts and
        # scripts with no checkpoints at all land here.
        return
    if rsv.enrolled or rsv.declared:
        return
    flor_print(
        f"FLOR: {filename} calls torch.save but declares no way to load it "
        f"back: no flor.checkpointing(...), no flor.restore(...), no "
        f"module-scope `obj.load_state_dict(torch.load(<literal path>))` (or "
        f"its keyed form), and nothing flor could read off the save itself."
        f"{_why_the_save_site_was_no_help(ssv)} Replay will "
        f"recompute from whatever state the script happens to build, which is "
        f"not the state the forward run had. Add "
        f"flor.restore(<path>, <name>=<obj>, ...) after you construct the "
        f"model to make replay faithful."
    )


def _infer_resume_spec(tree: ast.AST, filename: str) -> None:
    """Settle how replay will put mirrors back, in descending order of evidence.

    The resume block is the script saying outright which object takes which
    slice. The save site says the same thing about the file, but not about
    which object should end up holding it -- so it is consulted only when
    there is no resume block to read, and it announces what it concluded.
    """
    rbv = ResumeBlockVisitor()
    rbv.visit(tree)
    if rbv.found:
        flags.resume_spec = ResumeSpec(
            path=rbv.path,  # type: ignore[arg-type]
            lhs_name=rbv.lhs_name,
            applies=list(rbv.applies),
        )
        return
    if rbv.multi_path:
        # The script's own resume block reads two files, so its state really is
        # split across them. The save site can only agree with that or
        # contradict it; neither makes one file enough.
        flor_print(
            "FLOR: this script's resume block loads more than one checkpoint "
            "file; auto-restore disabled, because replay addresses one file per "
            "run. Save the pieces into a single checkpoint and declare it with "
            "flor.restore(<path>, <name>=<obj>, ...), or enroll the objects "
            "with flor.checkpointing(...)."
        )
        return
    rsv = RestoreSignalVisitor()
    rsv.visit(tree)
    ssv = _infer_from_save_site(tree, filename, rsv)
    if flags.resume_spec is not None:
        return
    if rbv.unscoped_match:
        flor_print(
            "FLOR: torch.load resume pattern found outside module scope, and "
            "the torch.save that writes the file was no clearer; auto-restore "
            "disabled. Declare it with flor.restore(<path>, <name>=<obj>, ...) "
            "so replay can address it."
        )
    _warn_if_nothing_restores(filename, rsv, ssv)


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

    _infer_resume_spec(tree, filename)

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

    flags.historical_args = dict(historical_hps)
    flags.hyperparameters.update(historical_hps)
    flags.hyperparameters.update(flags.overrides)
    flags.old_tstamp = data[0]["tstamp"]
