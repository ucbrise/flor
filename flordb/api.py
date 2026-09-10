import inspect
import json
import os
import shlex
import shutil
import statistics
import sys
import time
from pathlib import Path
from .constants import *
from .clock import Clock
from . import orm
from . import cli
from . import utils
from . import versions
from . import database
from . import capture
from . import checkpoint_io

from typing import Any, Iterable, Iterator, List, TypeVar, Optional, NoReturn
from contextlib import contextmanager

from tqdm import tqdm
import atexit

CMD_FILE = os.path.join(CURRDIR, ".flor.cmd") # type: ignore

T = TypeVar("T")

output_buffer = []
run_args: dict = {}

layers = {}
context: List[orm.Segment] = []

# Setup/teardown profiling anchors. `_setup_emitted` flips on first outermost
# flor.loop / flor.iteration entry (emitting time::setup once, measured from
# script start). `_last_main_exit_time` is updated at each outermost loop /
# iteration exit; commit() then emits time::teardown = perf_counter() - that.
_setup_emitted: bool = False
_last_main_exit_time: Optional[float] = None

# Set while replay runs an iteration the user didn't select. The body still
# runs -- later iterations depend on the state it leaves -- but records nothing.
_suppress_logs: bool = False

# Set when _neutralized_resume_state has turned the script's resume block into
# a no-op, so replay starts from the same initialization the forward run did.
_resume_neutralized: bool = False

# Set when the script read its resume block's file during setup, before any
# loop. Only then can end-of-run state be sitting on top of its initialization,
# which is what _refuse_unneutralized_resume checks for.
_resume_load_seen: bool = False

# flor.arg names the replayed run never logged, which fell back to their
# declared default. Tracked so the warning prints once per name per session.
_replay_defaulted_args: set = set()

# Retired calls already reported as no-ops this session.
_retired_warned: set = set()

# True in a training script; False in a notebook, `python -c`, or flor's own
# CLI (set by flordb/__init__). Only a script's run records where its training
# started -- a notebook reading checkpoints is not starting a run.
_script_run: bool = False

# (via, key, call) of each start this run has recorded. Explicit checkpoint
# loads carry an occurrence number; repeated torch loads share a file's start.
_starts_recorded: set = set()

# (recorded, current) tstamp pairs already reported as drifted this session.
_start_notes: set = set()

skip_cleanup = True


def _retired(name: str, message: str) -> None:
    """Say, once per session, that a retired call no longer does anything.

    The calls still have to run: replay checks out and executes historical
    versions of the script, which were written against the old API.
    """
    if name in _retired_warned:
        return
    _retired_warned.add(name)
    capture.flor_print(f"FLOR: {name}(...) no longer has an effect. {message}")


def set_ckpt_interval(seconds: float) -> None:
    _retired(
        "flor.set_ckpt_interval",
        "FlorDB keeps one copy of each file the script saves per run, not one "
        "per iteration, so there is no interval to set. You can remove the call.",
    )


def set_capture(
    enabled: Optional[bool] = None,
    *,
    max_line: Optional[int] = None,
    max_records: Optional[int] = None,
    extract: Optional[bool] = None,
) -> None:
    """Tune automatic capture of print / logging output.

    enabled      -- record io at all (also settable with FLOR_CAPTURE=0)
    max_line     -- longest line recorded verbatim
    max_records  -- per-run ceiling on captured lines
    """
    if enabled is not None:
        capture.config.enabled = bool(enabled)
    if max_line is not None:
        capture.config.max_line = int(max_line)
    if max_records is not None:
        capture.config.max_records = int(max_records)
    if extract is not None:
        # Kept as an accepted keyword so scripts carrying it still run. It no
        # longer does anything: extraction moved off the write path, where it
        # cost a re-run, to `python -m flordb capture --extract`, which derives
        # the same metrics from io already on disk.
        capture.flor_print(
            "FLOR: flor.set_capture(extract=...) no longer has an effect. "
            "Extract metrics from captured text with "
            "`python -m flordb capture --extract` (preview first with "
            "`--preview`); it reads runs you have already recorded."
        )


def _emit_setup_once():
    global _setup_emitted
    if _setup_emitted:
        return
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            _ctx_snapshot(),
            "time::setup",
            Clock().get_delta(),
            VALUE_TYPE_TIME,
        )
    )
    _setup_emitted = True


def _mark_main_segment_end():
    global _last_main_exit_time
    _last_main_exit_time = time.perf_counter()


def _emit_iter_summary(deltas: List[float]) -> None:
    # Replaces the per-iter time::iter stream with a distributional summary
    # anchored on the loop's parent ctx (or None at the outermost level).
    # `time::iter` carries the mean so `flor.dataframe("time::iter")` keeps
    # behaving like a single time-valued column; std and n live on companion
    # value names for callers that want the spread or the iteration count.
    if not deltas:
        return
    n = len(deltas)
    mean = statistics.fmean(deltas)
    std = statistics.stdev(deltas) if n >= 2 else 0.0
    ts = Clock.get_datetime()
    ctx = _ctx_snapshot()
    for value_name, value in (
        ("time::iter", mean),
        ("time::iter::std", std),
        ("time::iter::n", n),
    ):
        output_buffer.append(
            orm.Log(PROJID, ts, SCRIPTNAME, ctx, value_name, value, VALUE_TYPE_TIME)
        )


def _ctx_snapshot() -> Optional[List[orm.Segment]]:
    return list(context) if context else None


def _recording(name: str, bypass_projection: bool = False) -> bool:
    """Whether a record under `name` should be buffered right now.

    Two gates, shared by flor.log and by captured io so both narrow the same
    way:

      * the --apply projection (replay only) -- only the named values are
        emitted;
      * _suppress_logs -- replay runs an iteration the user didn't select:
        it advances state but records nothing.
    """
    if cli.flags.replay_start_failed:
        return False
    if (
        cli.in_replay_mode()
        and cli.flags.apply_vars is not None
        and name not in cli.flags.apply_vars
        and not bypass_projection
    ):
        return False
    return not _suppress_logs


def log(name, value, _bypass_projection: bool = False):
    if skip_cleanup:
        _deferred_init()

    serializable_value = value if utils.is_jsonable(value) else str(value)
    # Muted: this is flor echoing the value it is already recording. Letting
    # capture see it would store every metric twice -- once structured here,
    # once as a line of text.
    with capture.muted():
        tqdm.write(utils.to_string(layers, name, serializable_value))

    if not _recording(name, _bypass_projection):
        return value

    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            _ctx_snapshot(),
            name,
            serializable_value,
            VALUE_TYPE_LOG,
        )
    )

    return value


def _ctx_key(ctx):
    if not ctx:
        return None
    return tuple((s.name, s.iteration, s.value) for s in ctx)


# Last captured io record, as (channel, ctx, text). Consecutive repeats of the
# same line within one loop iteration collapse to a single row; the same line
# in the *next* iteration has a different ctx, so it survives.
_last_io_record: Optional[tuple] = None


_init_failed: bool = False


def _register_run():
    """Make sure the run will be committed at exit.

    In the zero-code-change case -- a script whose only flor reference is
    `import flordb` -- nothing ever calls log / arg / loop / iteration, so
    captured io is the only thing that can flip `skip_cleanup` and get the run
    written. Without this, the whole run is silently dropped at exit.

    Reported rather than raised: this runs underneath a `print`, inside the
    tee's catch-all, so an exception here would be swallowed and the user would
    be left wondering where their run went.
    """
    global _init_failed
    if not skip_cleanup or _init_failed:
        return
    try:
        _deferred_init()
    except Exception as e:
        _init_failed = True
        capture.flor_print(
            f"FLOR: captured this script's output, but the run cannot be "
            f"recorded: {e}"
        )


def _emit_io(channel: str, text: str) -> None:
    """Sink for flordb.capture -- one call per captured line.

    Captured io is `VALUE_TYPE_IO`, which keeps it out of `flor.dataframe()`
    unless asked for by name, and it rides the same output_buffer as everything
    else, so JSONL writing and forward/replay source tagging come for free.
    """
    global _last_io_record
    ctx = _ctx_snapshot()
    buffered = False

    if _recording(channel):
        key = (channel, _ctx_key(ctx), text)
        if key != _last_io_record:
            _last_io_record = key
            output_buffer.append(
                orm.Log(
                    PROJID,
                    Clock.get_datetime(),
                    SCRIPTNAME,
                    ctx,
                    channel,
                    text,
                    VALUE_TYPE_IO,
                )
            )
            buffered = True

    # Extraction is deliberately absent here. It is a guess about text, and a
    # guess does not belong in the run's JSONL, which is the immutable record
    # of what happened. It runs instead over rows already stored, on demand:
    # `python -m flordb capture --extract`.

    if buffered:
        _register_run()


def arg(name: str, default: Optional[Any] = None) -> Any:
    if cli.in_replay_mode():
        # GIT
        if name not in cli.flags.hyperparameters:
            # A flor.arg added to the script since the run being replayed:
            # there is no recorded value to reproduce. Fall back to the
            # declared default so hindsight logging still works, but say so --
            # for this key the replay is not a faithful reproduction.
            if default is None:
                raise RuntimeError(
                    f"FLOR: flor.arg({name!r}) was not logged by the run being "
                    f"replayed, and has no default to fall back on. Give it a "
                    f"default, or pass --override {name}=<value>."
                )
            if name not in _replay_defaulted_args:
                _replay_defaulted_args.add(name)
                capture.flor_print(
                    f"FLOR: flor.arg({name!r}) is absent from the replayed run; "
                    f"using default {default!r}. Pass --override {name}=<value> "
                    f"to replay it with a different value."
                )
            log(name, default, _bypass_projection=True)
            run_args[name] = default
            return default
        historical_v = cli.flags.hyperparameters[name]
        if name in cli.flags.overrides:
            # Historical values come back JSON-typed from the run's JSONL, but
            # --override arrives as a CLI string. Cast it to the type it is
            # replacing (or, for keys with no history, the declared default).
            proto = cli.flags.historical_args.get(name, default)
            if proto is not None:
                historical_v = utils.duck_cast(historical_v, proto)
        # Args are run configuration, not observations: they must survive an
        # --apply projection or the replay rows can't be joined against the
        # hyperparameters that produced them.
        log(name, historical_v, _bypass_projection=True)
        run_args[name] = historical_v
        return historical_v
    elif name in cli.flags.hyperparameters:
        # CLI
        v = cli.flags.hyperparameters[name]
        if default is not None:
            v = utils.duck_cast(v, default)
            log(name, v)
            run_args[name] = v
            return v
        log(name, v)
        run_args[name] = v
        return v
    elif default is not None:
        # default
        log(name, default)
        run_args[name] = default
        return default
    else:
        raise RuntimeError(
            f"FLOR: flor.arg({name!r}) has no default and no value was supplied. "
            f"Give it one -- flor.arg({name!r}, <value>) -- or pass "
            f"--kwargs {name}=<value> on the command line."
        )


@contextmanager
def checkpointing(**kwargs):
    """Retired: FlorDB keeps a copy of each file the script torch.saves.

    Still a working context manager, so scripts written against it -- among
    them the historical versions replay executes -- run unchanged.
    """
    _deferred_init()
    _retired(
        "flor.checkpointing",
        "FlorDB keeps one copy per run of each file the script saves with "
        "torch.save; save these objects that way instead. The block still runs "
        "as a plain `with`.",
    )
    yield


def restore(path, *target, **keyed) -> None:
    """Retired: replay recomputes from iteration 0 and restores nothing."""
    _retired(
        "flor.restore",
        "Replay always recomputes from iteration 0, so there is no checkpoint "
        "to restore into. You can remove the call.",
    )


def _iteration_requested(name: str, idx: Optional[int]) -> bool:
    """Whether this flor.iteration should emit logs during replay.

    flor.iteration doesn't own its iteration space -- the user's own loop (or
    one process per iteration) supplies `idx` -- so flor can neither enumerate
    the iterations up front nor skip the body of a `with` block. Narrowing is
    therefore expressed as log suppression, the same mechanism flor.loop uses
    for the iterations replay runs without being asked about them: the body
    runs (state has to advance), but nothing is recorded.
    """
    spec = cli.flags.iter_specs.get(name)
    if spec is None or spec.kind == "all":
        return True
    if spec.kind == "none":
        return False
    if spec.kind == "last":
        if name not in _unbounded_last_warned:
            _unbounded_last_warned.add(name)
            capture.flor_print(
                f"FLOR: --iter {name}=last is not decidable for flor.iteration "
                f"(flor can't know which iteration is the last one); logging every "
                f"iteration instead. Use --iter {name}=<indices> to filter."
            )
        return True
    return idx in spec.indices


_unbounded_last_warned: set = set()


@contextmanager
def iteration(name: str, idx: Optional[int], value: Optional[str]):
    global _suppress_logs
    _deferred_init()
    pos = len(layers)
    replaying = cli.in_replay_mode()
    if pos == 0:
        if replaying:
            _refuse_unneutralized_resume()
        _emit_setup_once()
    clock = Clock()
    clock.set_start_time()
    layers[name] = (
        int(idx) if idx is not None else None,
        str(value) if value is not None else None,
    )
    context.append(orm.Segment(name, layers[name][0], layers[name][1]))
    outer_suppress = _suppress_logs
    if replaying:
        _suppress_logs = outer_suppress or not _iteration_requested(
            name, layers[name][0]
        )
    try:
        yield
    finally:
        _suppress_logs = outer_suppress
        context.pop()
        if pos == 0:
            _mark_main_segment_end()
        del layers[name]
    # Anchored on the parent ctx (context was popped above), matching how
    # flor.loop anchors its time::iter summary.
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            _ctx_snapshot(),
            "time::iter",
            clock.get_delta(),
            VALUE_TYPE_TIME,
        )
    )


def loop(name: str, iterator: Iterable[T]) -> Iterator[T]:
    global _suppress_logs
    _deferred_init()
    pos = len(layers)
    replaying = cli.in_replay_mode()
    if pos == 0:
        if replaying:
            _refuse_unneutralized_resume()
        _emit_setup_once()
        _suppress_logs = False
    clock = Clock()
    clock.set_start_time()
    layers[name] = (0, None)
    context.append(orm.Segment(name, 0, None))
    outer_suppress = _suppress_logs
    # Replay materializes the iterator: which iterations run, and which of
    # them log, both depend on how many there are. Forward keeps the original
    # lazy iterator semantics.
    iter_source: Any
    silent: set = set()
    if replaying:
        materialized = list(iterator)
        if pos == 0:
            iter_source, silent = _build_outer_replay_plan(name, materialized)
        else:
            iter_source = list(enumerate(materialized))
            silent = _inner_silent(name, len(materialized))
    else:
        iter_source = enumerate(iterator)
    iter_deltas: List[float] = []
    try:
        for each in tqdm(
            iter_source,
            position=pos,
            leave=(True if pos == 0 else False),
            # flor's own progress bar, written past the tee. The user's tqdm
            # bars are still captured (one row for the final rendering); ours
            # would be pure redundancy.
            file=capture.raw_stderr(),
        ):
            layers[name] = (
                int(each[0]),
                str(each[1]) if utils.is_jsonable(each[1]) else None,
            )
            context[-1] = orm.Segment(name, layers[name][0], layers[name][1])
            if replaying:
                _suppress_logs = outer_suppress or int(each[0]) in silent
            iter_clock = Clock()
            iter_clock.set_start_time()
            yield each[1]  # type: ignore
            iter_deltas.append(iter_clock.get_delta())
    finally:
        _suppress_logs = outer_suppress
    context.pop()
    _emit_iter_summary(iter_deltas)
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            _ctx_snapshot(),
            "time::loop",
            clock.get_delta(),
            VALUE_TYPE_TIME,
        )
    )
    if pos == 0:
        _mark_main_segment_end()
    del layers[name]


def commit():
    global skip_cleanup, _setup_emitted, _last_main_exit_time, _last_io_record
    global _init_failed
    if cli.in_replay_mode():
        if cli.flags.replay_start_failed:
            output_buffer.clear()
            skip_cleanup = True
            return
        _require_recorded_starts()
    # Record any trailing output that never got a newline. Done here rather
    # than from its own atexit hook so it is guaranteed to land before the
    # buffer is serialized -- cleanup() below is itself an atexit hook, and
    # hook ordering between two of them is not something to rely on.
    capture.flush()
    tstamp = Clock.get_datetime()
    # time::script is the total wall time of the run -- always emitted, so
    # flat scripts (featurization, mapping, anything that's just a sequence of
    # flor.log calls with no loop) still get a profiling number out of the box.
    output_buffer.append(
        orm.Log(
            PROJID,
            tstamp,
            SCRIPTNAME,
            _ctx_snapshot(),
            "time::script",
            Clock().get_delta(),
            VALUE_TYPE_TIME,
        )
    )
    # time::teardown only makes sense if a loop/iteration boundary anchored
    # a "main work" segment. Skip it on flat scripts to avoid a meaningless 0.
    if _last_main_exit_time is not None:
        output_buffer.append(
            orm.Log(
                PROJID,
                tstamp,
                SCRIPTNAME,
                _ctx_snapshot(),
                "time::teardown",
                time.perf_counter() - _last_main_exit_time,
                VALUE_TYPE_TIME,
            )
        )
    # The sqlite transaction is opened, committed and closed before any git
    # work happens. git_commit shells out to `git add -A`, which can be slow
    # enough to Ctrl-C through; an interrupt raised while this connection sat
    # open mid-write left its RESERVED lock held for the life of the process,
    # so every later commit() died with "database is locked". try/finally so
    # the handle is released no matter what -- including BaseException, which
    # git_commit's own `except Exception` would not have caught anyway.
    git_message = None
    conn, cursor = database.conn_and_cursor()
    try:
        if not cli.in_replay_mode():
            # RECORD
            branch = versions.current_branch()
            if branch is not None:
                orm.to_jsonl(output_buffer, tstamp)
                database.unpack(output_buffer, cursor, source="forward")
                _write_cmd_file(tstamp)
                git_message = _build_commit_message(tstamp, run_args)
        else:
            # Replay rows are scratch -- not written to JSONL or git, and wiped
            # whenever `flor unpack` rebuilds the cache from JSONL truth.
            database.unpack(output_buffer, cursor, source="replay")
        conn.commit()
    finally:
        conn.close()
    output_buffer.clear()
    run_args.clear()
    Clock.set_new_datetime()
    _setup_emitted = False
    _last_main_exit_time = None
    _last_io_record = None
    _init_failed = False
    capture.reset_run_state()
    _starts_recorded.clear()
    cli.flags.checkpoint_calls.clear()
    cli.flags.consumed_starts.clear()
    skip_cleanup = True
    # Last, and outside the buffer reset above: by this point the run is
    # durable in both JSONL and the sqlite cache, so an interrupt here costs
    # the git commit and nothing else. Resetting first is what keeps the
    # retry -- the next cell's callback, or the atexit hook -- from writing
    # the same rows a second time.
    if git_message is not None:
        versions.git_commit(git_message)


def _build_commit_message(tstamp: str, args: dict) -> str:
    subject = f"{versions.AUTO_COMMIT_SUBJECT_PREFIX}{tstamp}"
    if not args:
        return subject
    body = "\n".join(f"{k}={v}" for k, v in args.items())
    return f"{subject}\n\n{body}"


def _write_cmd_file(tstamp: str):
    cmd = " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv])
    with open(CMD_FILE, "w") as f:
        f.write(f"{tstamp}\n{cmd}\n")


@atexit.register
def cleanup():
    if skip_cleanup:
        return
    commit()


def _deferred_init():
    global skip_cleanup
    if skip_cleanup:
        skip_cleanup = False
        if not cli.in_replay_mode():
            assert (
                versions.current_branch() is not None
            ), "Running from a detached HEAD?"
            versions.ensure_gitignored()
            versions.to_shadow()
    _install_torch_hooks()


# ---------------------------------------------------------------------------
# Where training starts
#
# A run may resume from an earlier run's checkpoint, either by loading the file
# that run saved or by asking flor for it. The forward run records which run
# and checkpoint it started from, identified by tstamp and commit, and replay
# loads that same checkpoint instead of starting from initialization.
# ---------------------------------------------------------------------------


def load_checkpoint(tstamp, name=None, *, map_location="cpu", weights_only=True):
    """Return saved state for a run, without executing its training script.

    With no name, require exactly one run checkpoint. Use flor.checkpoints(tstamp)
    to choose among several, or to select an older run's iteration snapshot.
    PyTorch checkpoints return the saved dictionary, not a constructed model;
    instantiate the matching architecture and call model.load_state_dict().
    Older runs' snapshots of other objects return their numpy, pandas, or
    cloudpickle values. Only load checkpoints you trust, especially
    pickle-backed objects or torch checkpoints loaded with weights_only=False.

    Called from a training script, this also records the run the script is
    starting from. Replaying the script's run loads that same checkpoint,
    whatever `tstamp` evaluates to by then. Repeated loads of one name are
    matched in call order. A replay load with no historical match is refused.
    """
    tracking = cli.in_replay_mode() or _script_run
    call = cli.flags.checkpoint_calls.get(name, 0)
    if tracking:
        cli.flags.checkpoint_calls[name] = call + 1
    if cli.in_replay_mode():
        start = _recorded_start("load_checkpoint", name, call)
        if start is None:
            _fail_replay_start(
                f"FLOR: cannot replay flor.load_checkpoint({name!r}), call "
                f"{call + 1}: the historical run recorded no matching load. "
                "The query or resume guard may now select a checkpoint the "
                "original run never loaded. Restore the original control flow "
                "before replaying; refusing to change the run's starting state."
            )
        _note_drift(tstamp, start)
        entry = _start_entry(start)
        state = checkpoint_io._read(entry, map_location, weights_only)
        cli.flags.consumed_starts.add(cli.flags.starts.index(start))
    else:
        entry = checkpoint_io._select(tstamp, name)
        state = checkpoint_io._read(entry, map_location, weights_only)
        if _script_run:
            _record_start("load_checkpoint", name, tstamp, entry["name"], call)
    return state


def _record_start(via: str, key, tstamp, name: str, call=None) -> None:
    """Record that this run's training starts from run `tstamp`'s `name`.

    `via` and `key` say which call loaded it -- a setup torch.load of a
    project-relative path, or flor.load_checkpoint with a name -- so replay can
    answer the same call with the same checkpoint. Explicit loads also record
    their occurrence among calls with that name, including an omitted name.
    """
    if (via, key, call) in _starts_recorded:
        return
    _deferred_init()
    _starts_recorded.add((via, key, call))
    tstamp = checkpoint_io._stamp(tstamp)
    start = {
        "via": via,
        "key": key,
        "tstamp": tstamp,
        "name": name,
        "commit": _commit_of(tstamp),
        "setup": not _setup_emitted,
    }
    if call is not None:
        start["call"] = call
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            None,
            "flor::start",
            json.dumps(start),
            VALUE_TYPE_START,
        )
    )


def _commit_of(tstamp: str) -> Optional[str]:
    """The auto-commit that recorded run `tstamp`, or None if there is none."""
    for ts, hexsha, _ in versions.get_latest_autocommit():
        try:
            if checkpoint_io._stamp(ts) == tstamp:
                return hexsha
        except ValueError:
            continue
    return None


def _recorded_start(via: str, key, call=None) -> Optional[dict]:
    """The start the replayed run recorded for this call, if it recorded one."""
    for start in cli.flags.starts:
        if (
            start.get("via") == via and start.get("key") == key
            # Older records can identify only the first explicit load.
            and (call is None or start.get("call", 0) == call)
        ):
            return start
    return None


def _start_entry(start: dict) -> dict:
    """The checkpoints() row for a recorded start, or a refusal naming the run."""
    try:
        entry = checkpoint_io._select(start["tstamp"], start["name"])
        if not Path(entry["path"]).is_file():
            raise FileNotFoundError(entry["path"])
        return entry
    except FileNotFoundError:
        commit = start.get("commit")
        at = f" (commit {commit[:10]})" if commit else ""
        _fail_replay_start(
            f"FLOR: the run being replayed started from {start['name']!r} of run "
            f"{start['tstamp']}{at}, and that run's copy isn't in .flor/obj_store/ "
            f"here. Copy .flor/obj_store/{start['tstamp']}/ from the machine "
            f"that trained it, then replay again."
        )


def _fail_replay_start(message: str) -> NoReturn:
    """A refused replay must not commit partially recomputed observations."""
    cli.flags.replay_start_failed = True
    output_buffer.clear()
    raise RuntimeError(message) from None


def _require_recorded_starts() -> None:
    """Refuse training if a guard skipped a historical setup load.

    Loading into the targets here would be too late: setup may already have
    computed other values from their state. The original load must execute.
    Older lineage records predate the setup flag and describe setup loads.
    """
    for index, start in enumerate(cli.flags.starts):
        if not start.get("setup", True) or index in cli.flags.consumed_starts:
            continue
        entry = _start_entry(start)
        _fail_replay_start(
            f"FLOR: the historical run loaded {start['name']!r} from run "
            f"{start['tstamp']}, but replay skipped that setup load. "
            "Check the script's resume guard or checkpoint query. "
            f"The recorded copy is at {entry['path']}; make the original "
            "load execute before replaying. Refusing to train from initialization."
        )


def _note_drift(tstamp, start: dict) -> None:
    """Say when the script's query no longer picks the run it started from."""
    try:
        now = checkpoint_io._stamp(tstamp)
    except (TypeError, ValueError):
        now = None
    if now == start["tstamp"] or (start["tstamp"], now) in _start_notes:
        return
    _start_notes.add((start["tstamp"], now))
    capture.flor_print(
        f"FLOR: flor.load_checkpoint now selects run {now}, but the run being "
        f"replayed started from run {start['tstamp']}; replaying from that one."
    )


# ---------------------------------------------------------------------------
# Replay planning
#
# Replay keeps no checkpoints from inside a run, so the state at iteration k
# exists only by computing iterations 0..k. --iter therefore chooses which
# iterations *log*; what runs is whatever those iterations depend on.
# ---------------------------------------------------------------------------


def _requested(name: str, n: int) -> List[int]:
    """Positions among a loop's n iterations that --iter asks replay to log."""
    spec = cli.iter_spec_for(name)
    if spec.kind == "all":
        return list(range(n))
    if spec.kind == "none":
        return []
    if spec.kind == "last":
        return [n - 1] if n else []
    out_of_range = [i for i in spec.indices if not (0 <= i < n)]
    if out_of_range:
        raise RuntimeError(
            f"FLOR: --iter {name}={list(spec.indices)} requests index "
            f"{out_of_range} but the loop only has {n} iteration(s) "
            f"(valid range: 0..{n - 1})."
        )
    return list(spec.indices)


def _build_outer_replay_plan(name: str, materialized: list):
    """Which outermost-loop iterations replay runs, and which of those are silent.

    Returns (iter_source, silent). The plan starts at iteration 0 and stops
    after the last requested one; the iterations it passes through on the way
    run with their logs suppressed.
    """
    if not cli.flags.wev_found:
        # No flor.loop in the script's source -- nothing to narrow.
        return list(enumerate(materialized)), set()
    requested = _requested(name, len(materialized))
    if not requested:
        return [], set()
    through = requested[-1] + 1
    return (
        list(enumerate(materialized[:through])),
        set(range(through)) - set(requested),
    )


def _inner_silent(name: str, n: int) -> set:
    """Iterations of a nested loop that replay runs but records nothing for.

    A nested loop always runs in full: whatever follows it in the enclosing
    iteration depends on the state every one of its steps leaves behind.
    """
    if not cli.flags.wev_found:
        return set()
    return set(range(n)) - set(_requested(name, n))


def _refuse_unneutralized_resume() -> None:
    """Stop a replay whose setup loaded a checkpoint over its initialization.

    Replay recomputes from iteration 0, so the objects entering the loop have
    to be the ones the script's initialization built. A recognized resume
    block's load is answered with the targets' own state; when that couldn't be
    done, the load went through and the objects hold whatever the file held.
    """
    if not _setup_emitted:
        _require_recorded_starts()
    spec = cli.flags.resume_spec
    if spec is None or not _resume_load_seen or _resume_neutralized:
        return
    raise RuntimeError(
        f"FLOR: cannot replay from iteration 0: the resume block in "
        f"{SCRIPTNAME} loaded {spec.path!r} before the loop, and flor could not "
        f"keep it from replacing the script's initialization. Move or delete "
        f"{spec.path!r} and replay again."
    )


# ---------------------------------------------------------------------------
# torch.save / torch.load hooks
#
# Forward: every torch.save to a file path also replaces this run's copy of
# that file under .flor/obj_store/<tstamp>/, so a script needs no flor code to
# have each run's checkpoint kept. Replay: torch.save is skipped, and the
# script's resume block has its torch.load answered with its targets' own
# state, so the recomputation starts from the script's initialization.
# ---------------------------------------------------------------------------

_orig_torch_save = None
_orig_torch_load = None


def _coerce_to_path(path) -> Path:
    # torch.save / torch.load accept str, os.PathLike, or a binary file object.
    # str/PathLike include pathlib.Path, which has a .name (basename) attribute,
    # so we must NOT branch on hasattr(path, "name"); that would discard the
    # directory part. Only file-like objects fall back to .name.
    if isinstance(path, (str, bytes, os.PathLike)):
        return Path(os.fsdecode(path))
    name = getattr(path, "name", None)
    return Path(str(name)) if name else Path("ckpt.pth")


def _flor_torch_save(obj, path, *args, **kwargs):
    assert _orig_torch_save is not None
    if cli.in_replay_mode():
        # The script's file belongs to the latest forward run, and replay is
        # recomputing some other one: leave the file alone and keep no copy.
        return None
    result = _orig_torch_save(obj, path, *args, **kwargs)
    if isinstance(path, (str, bytes, os.PathLike)):
        # Every successful save replaces the copy, so it ends up holding
        # whatever the run saved last, inside the loop or after it.
        _deferred_init()
        saved_path = _coerce_to_path(path)
        checkpoint_io._save(
            os.fsdecode(path),
            lambda destination: shutil.copyfile(saved_path, destination),
            source=saved_path,
        )
    return result


def _neutralized_resume_state(path):
    """The state dict that makes the script's resume block a no-op on replay.

    The resume block (`torch.load("ckpt.pth")` + `load_state_dict`) runs during
    setup and loads whatever is on disk. After a forward run that file holds
    *end-of-run* weights, which would land on top of the fresh initialization
    replay recomputes from.

    Returning each target's *own* current state avoids that:
    `model.load_state_dict(model.state_dict())` leaves the seed-initialized
    weights in place, which is precisely the state the forward run's iteration 0
    started from. Returns None when the block can't be neutralized faithfully;
    the load then goes through, and _refuse_unneutralized_resume stops the
    replay at the loop rather than let it recompute from the wrong state.
    """
    global _resume_neutralized, _resume_load_seen
    spec = cli.flags.resume_spec
    if spec is None:
        return None
    try:
        if _coerce_to_path(path).name != _coerce_to_path(spec.path).name:
            return None
        _resume_load_seen = True
        targets = _resolve_targets(spec)
        if targets is None:
            return None
        state = {}
        flat = None
        for target_name, key in spec.applies:
            snapshot = getattr(targets.get(target_name), "state_dict", None)
            if snapshot is None:
                # A target the resume block writes to but we can't snapshot --
                # a partial no-op would silently half-apply end-of-run state.
                return None
            if key is None:
                # Flat idiom: the file *is* one object's state_dict, so the
                # no-op value is that object's own state, unwrapped.
                if len(spec.applies) != 1:
                    return None
                flat = snapshot()
            else:
                state[key] = snapshot()
        if flat is not None:
            _resume_neutralized = True
            return flat
        if not state:
            return None
    except Exception:
        return None
    _resume_neutralized = True
    return state


def _flor_torch_load(path, *args, **kwargs):
    assert _orig_torch_load is not None
    # Only setup loads concern flor -- the ones before any loop, which is where
    # a script resumes. Once a loop is running, a load is the script's own
    # business, and it gets the file it names.
    if not layers:
        key = (
            checkpoint_io._project_relative(path)
            if isinstance(path, (str, bytes, os.PathLike))
            else None
        )
        if cli.in_replay_mode():
            start = _recorded_start("torch.load", key) if key else None
            if start is not None:
                # The forward run resumed from an earlier run's copy of this
                # file; start the recomputation from that same copy.
                state = _orig_torch_load(_start_entry(start)["path"], *args, **kwargs)
                cli.flags.consumed_starts.add(cli.flags.starts.index(start))
                return state
            neutral = _neutralized_resume_state(path)
            if neutral is not None:
                return neutral
        elif _script_run and key is not None:
            writer = checkpoint_io._writer_of(path, exclude=Clock.get_datetime())
            if writer is not None:
                state = _orig_torch_load(path, *args, **kwargs)
                _record_start("torch.load", key, *writer)
                return state
    return _orig_torch_load(path, *args, **kwargs)


def _install_torch_hooks():
    global _orig_torch_save, _orig_torch_load
    if _orig_torch_save is not None:
        return
    try:
        import torch # type: ignore
    except ImportError:
        return
    _orig_torch_save = torch.save
    _orig_torch_load = torch.load
    torch.save = _flor_torch_save  # type: ignore[assignment]
    torch.load = _flor_torch_load  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Resume-block targets
#
# cli.replay_initialize() finds the script's `torch.load(...)` +
# `X.load_state_dict(loaded[key])` block. Neutralizing it needs the live
# objects the block names: module-scope names are read off the script's frame,
# and a block inside a function has its locals bound while it runs, because it
# may return them to a caller that names them differently.
# ---------------------------------------------------------------------------


def _find_user_frame():
    """Walk outward from the current frame, return the first frame whose
    code filename basename matches SCRIPTNAME. Works whether the resume
    block and flor.loop live at module scope or inside a function in the
    user script.
    """
    f = inspect.currentframe()
    if f is None:
        return None
    f = f.f_back
    while f is not None:
        if os.path.basename(f.f_code.co_filename) == SCRIPTNAME:
            return f
        f = f.f_back
    return None


def _install_resume_binding(spec, filename):
    """Bind a recognized setup block's locals before its frame disappears.

    The script is already executing when it imports flor, so its functions
    cannot be AST-rewritten in place. Trace only the recognized scope until
    its targets exist at the resume block (or the next line after a skipped
    existence guard), then retain the objects and remove our trace. No frames
    are retained and training runs without this tracing overhead.
    """
    _install_torch_hooks()
    filename = os.path.abspath(filename)
    previous = sys.gettrace()
    active = True

    def stop():
        nonlocal active
        active = False
        if sys.gettrace() is trace:
            sys.settrace(previous)

    def trace(frame, event, arg):
        prior_local = previous(frame, event, arg) if previous else None
        if cli.flags.resume_spec is not spec:
            stop()
        if not active:
            return prior_local
        if (
            os.path.abspath(frame.f_code.co_filename) != filename
            or frame.f_code.co_name != spec.scope_name
            or frame.f_code.co_firstlineno != spec.scope_lineno
        ):
            return prior_local

        def bind(frame, event, arg):
            nonlocal prior_local
            if prior_local is not None:
                prior_local = prior_local(frame, event, arg)
            if cli.flags.resume_spec is not spec:
                stop()
            at_block = event == "line" and frame.f_lineno >= spec.lineno
            if active and (at_block or event == "return"):
                scope = dict(frame.f_globals)
                scope.update(frame.f_locals)
                targets = {name: scope.get(name) for name, _ in spec.applies}
                if all(
                    callable(getattr(obj, "load_state_dict", None))
                    for obj in targets.values()
                ):
                    spec.targets = targets
                    stop()
            return bind if active else prior_local

        return bind

    sys.settrace(trace)
    return stop


def _resolve_targets(spec) -> Optional[dict]:
    """The live objects spec.applies names, or None if they can't be reached.

    Function-scoped resume blocks bind their objects while they run. A
    module-scope block's names are resolved against the user's frame.
    """
    if spec.targets is not None:
        return dict(spec.targets)
    if spec.scope_name is not None:
        # Never resolve prep()'s names against unrelated locals in train().
        raise RuntimeError(
            f"FLOR: resume targets in {spec.scope_name}() were not bound before "
            "the replay loop."
        )
    user_frame = _find_user_frame()
    if user_frame is None:
        return None
    scope = dict(user_frame.f_globals)
    scope.update(user_frame.f_locals)
    return {name: scope.get(name) for name, _ in spec.applies}


__all__ = [
    "log",
    "arg",
    "checkpointing",
    "restore",
    "loop",
    "iteration",
    "commit",
    "output_buffer",
    "set_ckpt_interval",
    "set_capture",
    "load_checkpoint",
]
