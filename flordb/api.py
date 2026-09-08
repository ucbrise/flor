import inspect
import os
import shlex
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
from . import obj_store
from . import database
from . import capture

from typing import Any, Iterable, Iterator, List, TypeVar, Optional
from contextlib import contextmanager

from tqdm import tqdm
import atexit

CMD_FILE = os.path.join(CURRDIR, ".flor.cmd") # type: ignore

T = TypeVar("T")

output_buffer = []
run_args: dict = {}

layers = {}
context: List[orm.Segment] = []

checkpoints = []

# Adaptive-checkpoint trigger: at the end of each outermost flor.loop iteration
# (and on any user torch.save call), checkpoint at most once per ckpt_interval_s.
ckpt_interval_s: float = 60.0
_last_ckpt_time: Optional[float] = None

# Setup/teardown profiling anchors. `_setup_emitted` flips on first outermost
# flor.loop / flor.iteration entry (emitting time::setup once, measured from
# script start). `_last_main_exit_time` is updated at each outermost loop /
# iteration exit; commit() then emits time::teardown = perf_counter() - that.
_setup_emitted: bool = False
_last_main_exit_time: Optional[float] = None

# Logical-replay state. When the user requests an iter whose mirror was thrown
# away by ckpt_interval_s throttling, the outer flor.loop falls back to the
# most-recent earlier mirror and fast-forwards through intermediate iters with
# full inner-loop execution and suppressed logs. These flags coordinate that
# across the outer loop, inner slice(), and log().
_logical_replay_active: bool = False
_suppress_logs: bool = False

# Set when _neutralized_resume_state has turned the user's module-scope resume
# block into a no-op, so replaying from iteration 0 is starting from the same
# initialization the forward run did. The plan builder refuses from-zero without
# it whenever the user's checkpoint file is on disk.
_resume_neutralized: bool = False

# The mirror _restore_from_mirror is currently reaching for, set only for the
# duration of its torch.load. While it is set, the load hook must produce that
# file or raise: flor asked for one specific iteration, so quietly substituting
# the user's on-disk checkpoint would answer a different question than the one
# asked.
_restoring_mirror: Optional[Path] = None

# flor.arg names the replayed run never logged, which fell back to their
# declared default. Tracked so the warning prints once per name per session.
_replay_defaulted_args: set = set()

skip_cleanup = True


def set_ckpt_interval(seconds: float) -> None:
    global ckpt_interval_s
    ckpt_interval_s = float(seconds)


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
      * _suppress_logs -- a logical-replay fast-forward iteration advances
        state but records nothing.
    """
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
    # Optional explicit-enrollment helper. The torch.save hook is the default
    # piggy-back path; use this for non-torch objects (sklearn estimators, dicts,
    # etc.) that you want serialized at every adaptive ckpt() trigger.
    # Profiling records (time::setup / time::teardown) are now anchored on
    # outermost flor.loop boundaries, not on this block.
    try:
        checkpoints.extend(list(kwargs.items()))
        yield
    except Exception as e:
        capture.flor_print(f"An error occurred: {e}")
        raise
    finally:
        checkpoints.clear()


def restore(path, *target, **keyed) -> bool:
    """Declare how a torch checkpoint maps back onto live objects.

    The `torch.save` piggy-back path needs no instrumentation on the way out --
    flor mirrors whatever the script saves. Coming back in is the half that
    needs semantics, and flor can only guess at those by pattern-matching the
    script's own resume block. This is how you say it outright when the guess
    is wrong or impossible:

        flor.restore("ckpt.pth", model=model, optimizer=optimizer)
        # -> loaded["model"] into model, loaded["optimizer"] into optimizer

        flor.restore("ckpt.pth", model)
        # -> the whole file into model, for torch.save(model.state_dict(), ...)

    Call it at module scope, right after the objects are built, in place of the
    `torch.load(...)` + `load_state_dict(...)` block it replaces. It does that
    block's job on a forward run: if `path` exists, it loads and applies it, so
    an interrupted run still resumes.

    On replay it applies nothing here and returns False. The per-iteration
    restore inside flor.loop owns the objects then, and loading at module scope
    would put end-of-run state on top of the initialization that replaying from
    iteration 0 depends on -- the same trap `_neutralized_resume_state` exists
    to defuse for inferred blocks, avoided by construction here.

    Returns whether state was applied.
    """
    # Argument validation before any side effect, so a malformed call is a
    # plain TypeError and not a half-initialized run.
    if target and keyed:
        raise TypeError(
            "FLOR: flor.restore takes either one positional target (the whole "
            "file is that object's state) or keyword targets (each keyword is "
            "a key in the saved dict), not both."
        )
    if len(target) > 1:
        raise TypeError(
            f"FLOR: flor.restore takes at most one positional target, got "
            f"{len(target)}. Name them -- flor.restore({path!r}, "
            f"model=model, optimizer=optimizer) -- so each one can be matched "
            f"to its key in the saved dict."
        )
    if not target and not keyed:
        raise TypeError(
            f"FLOR: flor.restore({path!r}) needs at least one target to "
            f"restore into, e.g. flor.restore({path!r}, model=model)."
        )

    if target:
        # No key: the file holds exactly this object's state_dict.
        applies = [("<positional>", None)]
        targets = {"<positional>": target[0]}
    else:
        applies = [(name, name) for name in keyed]
        targets = dict(keyed)

    for name, obj in targets.items():
        if not hasattr(obj, "load_state_dict"):
            raise TypeError(
                f"FLOR: flor.restore target {name!r} is a "
                f"{type(obj).__name__}, which has no load_state_dict. Pass the "
                f"module or optimizer itself, or enroll it with "
                f"flor.checkpointing({name}=...) instead."
            )

    _deferred_init()
    cli.flags.resume_spec = cli.ResumeSpec(
        path=str(path),
        lhs_name=None,
        applies=applies,
        source="explicit",
        targets=targets,
    )
    _install_torch_hooks()

    if cli.in_replay_mode():
        return False
    if not os.path.exists(str(path)):
        return False
    if _orig_torch_load is None:
        raise RuntimeError(
            "FLOR: flor.restore needs torch, which is not importable here."
        )
    # The original, not the hook: this is a forward run reading the user's own
    # file, and there is no loop context to redirect it into anyway.
    loaded = _orig_torch_load(str(path))
    for target_name, key in applies:
        try:
            state = _select_state(loaded, key)
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"FLOR: flor.restore({str(path)!r}) found no {key!r} in the "
                f"saved checkpoint (it holds "
                f"{sorted(loaded) if isinstance(loaded, dict) else type(loaded).__name__}). "
                f"Name the targets after the keys they were saved under."
            ) from e
        targets[target_name].load_state_dict(state)
    return True


def _iteration_requested(name: str, idx: Optional[int]) -> bool:
    """Whether this flor.iteration should emit logs during replay.

    flor.iteration doesn't own its iteration space -- the user's own loop (or
    one process per iteration) supplies `idx` -- so flor can neither enumerate
    the iterations up front nor skip the body of a `with` block. Narrowing is
    therefore expressed as log suppression, the same mechanism flor.loop uses
    for fast-forward iters under logical replay: the body runs (state has to
    advance), but nothing is recorded for iterations the user didn't ask for.
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
    if pos == 0:
        _emit_setup_once()
    clock = Clock()
    clock.set_start_time()
    layers[name] = (
        int(idx) if idx is not None else None,
        str(value) if value is not None else None,
    )
    context.append(orm.Segment(name, layers[name][0], layers[name][1]))
    replaying = cli.in_replay_mode()
    outer_suppress = _suppress_logs
    if replaying:
        # Restore this iteration's historical state the same way the outermost
        # flor.loop does: explicitly enrolled objects first, then the
        # AST-detected torch resume block (outermost scope only -- a nested
        # iteration shares the outer scope's restored state).
        load_ckpt()
        if pos == 0 and cli.flags.resume_spec is not None:
            _restore_from_mirror(name, layers[name][0], layers[name][1])
        _suppress_logs = outer_suppress or not _iteration_requested(
            name, layers[name][0]
        )
    try:
        yield
        # Replay reads from the object store keyed on the *historical* tstamp
        # (obj_store.get_shelf), so an unconditional write here would overwrite
        # the mirrors the replay is reading from. Warming is allowed to fill
        # the gaps the forward run left, and only those.
        warming = replaying and _ckpt_warming_enabled()
        if not replaying or warming:
            ckpt(only_if_absent=warming)
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
    global _last_ckpt_time, _logical_replay_active, _suppress_logs
    _deferred_init()
    pos = len(layers)
    if pos == 0:
        _emit_setup_once()
        # Reset so the first iter's ckpt always fires; later iters get
        # throttled by the time guard.
        _last_ckpt_time = None
        _logical_replay_active = False
        _suppress_logs = False
    clock = Clock()
    clock.set_start_time()
    layers[name] = (0, None)
    context.append(orm.Segment(name, 0, None))
    # On replay we materialize so the planner / restore code can index into a
    # specific iter's value to build the matching obj_store filename. On
    # forward we keep the original lazy iterator semantics.
    logical_silent: set = set()
    logical_mirror_pos: Optional[int] = None
    if cli.in_replay_mode():
        materialized: Optional[list] = list(iterator)
        if pos == 0:
            iter_source: Any
            iter_source, logical_mirror_pos, logical_silent, _logical_replay_active = (
                _build_outer_replay_plan(name, materialized)
            )
        else:
            iter_source = slice(name, materialized)
    else:
        materialized = None
        iter_source = enumerate(iterator)
    first_outer_iter = True
    iter_deltas: List[float] = []
    for each in tqdm(
        iter_source,
        position=pos,
        leave=(True if pos == 0 else False),
        # flor's own progress bar, written past the tee. The user's tqdm bars
        # are still captured (one row for the final rendering); ours would be
        # pure redundancy.
        file=capture.raw_stderr(),
    ):
        layers[name] = (
            int(each[0]),
            str(each[1]) if utils.is_jsonable(each[1]) else None,
        )
        context[-1] = orm.Segment(name, layers[name][0], layers[name][1])
        if pos == 0 and cli.in_replay_mode():
            # Under fast-forward the state comes from the single restore below
            # plus recomputation, so a gap in the shelf is expected, not fatal.
            load_ckpt(missing_ok=_logical_replay_active)
            if materialized is not None:
                if _logical_replay_active:
                    _suppress_logs = int(each[0]) in logical_silent
                    if first_outer_iter and logical_mirror_pos is not None:
                        _restore_at(
                            name, materialized, logical_mirror_pos, enrolled=True
                        )
                else:
                    _suppress_logs = False
                    _restore_at(name, materialized, int(each[0]))
            first_outer_iter = False
        iter_clock = Clock()
        iter_clock.set_start_time()
        yield each[1]  # type: ignore
        iter_deltas.append(iter_clock.get_delta())
        # On replay this shelves only what the forward run left missing, so a
        # fast-forwarded iteration is paid for once rather than on every replay.
        warming = _ckpt_warming_enabled()
        if pos == 0 and (warming or not cli.in_replay_mode()):
            now = time.perf_counter()
            if _last_ckpt_time is None or (now - _last_ckpt_time) >= ckpt_interval_s:
                ckpt(only_if_absent=warming)
                _last_ckpt_time = now
    if pos == 0 and (_ckpt_warming_enabled() or not cli.in_replay_mode()):
        # Force a final checkpoint at outermost loop exit so end-of-run state
        # is always captured, regardless of the time guard.
        ckpt(only_if_absent=_ckpt_warming_enabled())
        _last_ckpt_time = time.perf_counter()
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
        _logical_replay_active = False
        _suppress_logs = False
    del layers[name]


def commit():
    global skip_cleanup, _setup_emitted, _last_main_exit_time, _last_io_record
    global _init_failed
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


# Overrides that cannot change what the recomputed state *is*. ckpt_interval_s
# only sets how often a checkpoint is taken, so a replay carrying it still
# reconstructs the forward run's state faithfully; `device` is on the CLI
# allowlist but cpu/cuda kernels do not agree bit-for-bit, so it is not here.
NUMERICS_NEUTRAL_OVERRIDES = frozenset({"ckpt_interval_s"})


def _ckpt_warming_enabled() -> bool:
    """May this replay shelve mirrors the forward run never left behind?

    A replay recomputes state the forward run held but did not save -- because
    ckpt_interval_s threw it away, or because the repo was cloned with
    runs/*.jsonl and no obj_store at all. Shelving it turns the *next* replay
    of that iteration into a fast-path restore instead of another fast-forward,
    which is what makes a fresh clone slow only once.

    Two rules keep this from corrupting history. Callers must honor the first;
    this predicate is the second:

      - Never overwrite. An existing mirror is forward-run truth and a warmed
        one is a reconstruction, so truth wins every collision.
      - Never warm under an override that could move the numbers. The
        reconstruction is only sound because the replay re-ran the same code
        over the same args; `--override device=cpu` breaks that premise, and
        caching its output as a mirror would quietly poison later replays.
    """
    if not cli.in_replay_mode():
        return False
    return not (set(cli.flags.overrides) - NUMERICS_NEUTRAL_OVERRIDES)


def ckpt(only_if_absent: bool = False):
    for name, obj in checkpoints:
        if only_if_absent and obj_store.has_shelved(layers, name):
            continue
        obj_store.serialize(layers, name, obj)


def load_ckpt(missing_ok: bool = False):
    for name, obj in checkpoints:
        obj_store.deserialize(layers, name, obj, missing_ok=missing_ok)


# ---------------------------------------------------------------------------
# torch.save / torch.load piggy-back hooks
#
# Lets cloned scripts that already call torch.save be checkpointed by flor
# without an explicit `with flor.checkpointing(...):` block. On forward runs,
# any torch.save called inside a flor.loop is mirrored into the project-local
# object store at .flor/obj_store/<run-tstamp>/, using a ctx-aware filename so
# each iteration produces its own snapshot. On replay, torch.load is redirected
# to the matching mirror so the user's own resume-from-checkpoint code restores
# the historical state.
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


def _user_path_stem_ext(path):
    p = _coerce_to_path(path)
    return p.stem or "ckpt", (p.suffix or ".pth")


def _path_in_obj_store(path) -> bool:
    try:
        p = _coerce_to_path(path).resolve()
        return str(p).startswith(str(Path(OBJSTORE_DIR).resolve()))
    except Exception:
        return False


def _flor_torch_save(obj, path, *args, **kwargs):
    global _last_ckpt_time
    assert _orig_torch_save is not None
    result = _orig_torch_save(obj, path, *args, **kwargs)
    if not layers:
        return result
    warming = cli.in_replay_mode() and _ckpt_warming_enabled()
    if cli.in_replay_mode() and not warming:
        return result
    # Don't mirror writes that already land in our own obj_store -- those are
    # ckpt() calls (or other flor-driven saves) and would just produce
    # duplicates with deeper-nested filenames.
    if _path_in_obj_store(path):
        return result
    now = time.perf_counter()
    if _last_ckpt_time is not None and (now - _last_ckpt_time) < ckpt_interval_s:
        return result
    try:
        stem, ext = _user_path_stem_ext(path)
        flor_path = obj_store.get_shelf() / utils.to_filename(layers, stem, ext)
        # Warming fills gaps; it never rewrites a mirror the forward run left.
        if warming and flor_path.exists():
            return result
        _orig_torch_save(obj, str(flor_path), *args, **kwargs)
        _last_ckpt_time = now
    except Exception:
        pass
    return result


def _shelf_has_mirrors_for(stem: str, ext: str) -> bool:
    """Whether the shelf holds a mirror for *any* iteration of this file.

    This is how a checkpoint flor is managing gets told apart from an ordinary
    torch.load of, say, a cached tensor: the former has siblings on the shelf,
    the latter has none. Only the former may be treated as an error when the
    iteration being replayed has no mirror of its own.
    """
    try:
        return any(obj_store.get_shelf().glob(f"{stem}*{ext}"))
    except Exception:
        return False


def _shelf_has_any_mirror_for(spec) -> bool:
    """Whether the historical run left any mirror for the resume block's file."""
    try:
        stem, ext = _user_path_stem_ext(spec.path)
    except Exception:
        return False
    return _shelf_has_mirrors_for(stem, ext)


def _neutralized_resume_state(path):
    """The state dict that makes the user's module-scope resume block a no-op.

    The resume block (`torch.load("ckpt.pth")` + `load_state_dict`) runs at
    module scope, where `layers` is still empty -- so the per-iteration redirect
    below cannot reach it, and it loads whatever is on disk. After a forward run
    that file holds *end-of-run* weights, which would land on top of the fresh
    initialization that replaying from iteration 0 depends on.

    When the shelf still has mirrors this doesn't matter: the plan restores one
    per iteration and overwrites the block's effect anyway. When the shelf is
    empty (a fresh clone -- runs/*.jsonl is committed, obj_store is not) there is
    nothing to overwrite it with, so the block has to be neutralized instead.

    Returning each target's *own* current state does exactly that:
    `model.load_state_dict(model.state_dict())` leaves the seed-initialized
    weights in place, which is precisely the state the forward run's iteration 0
    started from. Returns None when the block can't be neutralized faithfully,
    which keeps the caller on the loud-refusal path rather than guessing.
    """
    global _resume_neutralized
    spec = cli.flags.resume_spec
    if spec is None:
        return None
    try:
        if _coerce_to_path(path).name != _coerce_to_path(spec.path).name:
            return None
        if _shelf_has_any_mirror_for(spec):
            return None
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
    if cli.in_replay_mode():
        if layers:
            try:
                stem, ext = _user_path_stem_ext(path)
                flor_path = obj_store.get_shelf() / utils.to_filename(
                    layers, stem, ext
                )
            except Exception:
                if _restoring_mirror is not None:
                    # flor asked for a specific mirror and cannot even name it.
                    raise
                # Some exotic path/file object flor can't address. It is not a
                # checkpoint flor wrote, so the user's own load stands.
                return _orig_torch_load(path, *args, **kwargs)
            if flor_path.exists():
                return _orig_torch_load(str(flor_path), *args, **kwargs)
            if _restoring_mirror is not None or _shelf_has_mirrors_for(stem, ext):
                # Falling through here would load the file at the user's own
                # path, which after a forward run holds *end-of-run* state --
                # silently answering "what did iteration k look like?" with the
                # last iteration's weights, and logging the result as history.
                # The shelf has siblings, so this really is a flor-managed
                # checkpoint with a gap, not an unrelated load.
                raise RuntimeError(
                    f"FLOR: no checkpoint mirror for {_ctx_description()} at "
                    f"{flor_path.name!r} in {obj_store.get_shelf()}, but other "
                    f"iterations of {stem}{ext} are shelved. The forward run "
                    f"most likely throttled this iteration "
                    f"(flor.set_ckpt_interval). Narrow to an iteration that has "
                    f"a mirror, or re-run forward with a smaller interval. "
                    f"Refusing to fall back to {_coerce_to_path(path)!s}, which "
                    f"holds end-of-run state."
                )
        else:
            # Module scope: no loop context yet, so this is the resume block.
            neutral = _neutralized_resume_state(path)
            if neutral is not None:
                return neutral
    return _orig_torch_load(path, *args, **kwargs)


def _install_torch_hooks():
    global _orig_torch_save, _orig_torch_load
    if _orig_torch_save is not None:
        return
    try:
        import torch
    except ImportError:
        return
    _orig_torch_save = torch.save
    _orig_torch_load = torch.load
    torch.save = _flor_torch_save  # type: ignore[assignment]
    torch.load = _flor_torch_load  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# AST-driven auto-restore (no flor.checkpointing(...) required)
#
# When cli.replay_initialize() finds a module-scope `torch.load(...)` +
# `X.load_state_dict(loaded[key])` pattern in the user's script, flor.loop's
# outer replay plan picks which obj_store mirror to splice into the user's
# module/function frame: the iter's own mirror in the fast path, or the most
# recent earlier one when ckpt_interval_s threw the matching mirror away
# (logical replay -- intermediate iters then fast-forward through the body
# with logs suppressed). The user's own resume code (e.g. line 84 of
# v4/train.py) is left intact -- it still runs once before the loop -- but
# its result is overwritten per-iter.
# ---------------------------------------------------------------------------


def _ctx_description() -> str:
    """The loop context as `epoch=2, step=7`, for error messages."""
    parts = []
    for k, (i, v) in layers.items():
        parts.append(f"{k}={v if v is not None else i}")
    return ", ".join(parts) if parts else "<module scope>"


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


def _layer_for(materialized: list, k: int):
    """The `layers` entry the forward run held while executing iter position k.

    Mirror filenames are derived from `layers`, so this has to match what
    flor.loop writes on the forward pass exactly (index k, value stringified
    only when it's jsonable) or the replay looks for a file that isn't there.
    """
    v = materialized[k]
    return k, (str(v) if utils.is_jsonable(v) else None)


@contextmanager
def _layer_swapped(name: str, iteration: Optional[int], value: Optional[str]):
    """Temporarily present `layers` as it looked at a historical iteration.

    Both the mirror-path computation and _flor_torch_load read `layers`, so
    swapping it is how we address a specific iteration's checkpoint without
    changing any user-visible path.
    """
    had = name in layers
    saved = layers.get(name)
    layers[name] = (iteration, value)
    try:
        yield
    finally:
        if had:
            layers[name] = saved  # type: ignore[assignment]
        else:
            layers.pop(name, None)


def _mirror_path_for(name: str, k: int, materialized: list, spec) -> Path:
    """Build the obj_store mirror path for iter position k of `name`."""
    iteration, value = _layer_for(materialized, k)
    with _layer_swapped(name, iteration, value):
        stem, ext = _user_path_stem_ext(spec.path)
        return obj_store.get_shelf() / utils.to_filename(layers, stem, ext)


def _mirror_exists_at(name: str, k: int, materialized: list, spec) -> bool:
    """True when position k has everything a restore there would need.

    Both restore paths have to be satisfied, because both feed the same
    iteration: the AST-detected torch resume block (when `spec` is set) and
    every object enrolled through `flor.checkpointing`. A script using only
    enrollment has `spec is None` and is judged purely on the shelf; a script
    using neither has nothing to restore, so every position trivially qualifies.
    """
    try:
        if spec is not None and not _mirror_path_for(name, k, materialized, spec).exists():
            return False
        if not checkpoints:
            return True
        iteration, value = _layer_for(materialized, k)
        with _layer_swapped(name, iteration, value):
            return all(obj_store.has_shelved(layers, n) for n, _ in checkpoints)
    except Exception:
        return False


def _restore_at(
    name: str, materialized: list, pos: int, enrolled: bool = False
) -> None:
    """Restore historical position `pos` through whichever paths are in play.

    `enrolled` also re-runs the flor.checkpointing restore. Callers set it only
    when anchoring somewhere other than the iteration the loop is currently on
    -- the per-iteration load_ckpt in the loop already covers that case, and
    repeating it here would just deserialize the same shelf entry twice.
    """
    iteration, value = _layer_for(materialized, pos)
    if enrolled and checkpoints:
        with _layer_swapped(name, iteration, value):
            load_ckpt(missing_ok=True)
    if cli.flags.resume_spec is not None:
        _restore_from_mirror(name, iteration, value)


def _find_latest_mirror_at_or_before(
    name: str, position: int, materialized: list, spec
) -> Optional[int]:
    for k in range(position, -1, -1):
        if _mirror_exists_at(name, k, materialized, spec):
            return k
    return None


def _resolve_targets(spec) -> Optional[dict]:
    """The live objects spec.applies names, or None if they can't be reached.

    An explicit spec carries the objects themselves -- flor.restore was handed
    references, so there is nothing to look up and nothing to get wrong. An
    inferred spec has only names harvested from the source, which have to be
    resolved against the user's frame; that is the step that can silently come
    up empty when a name was rebound, shadowed, or moved into a function.
    """
    if spec.source == "explicit":
        return dict(spec.targets or {})
    user_frame = _find_user_frame()
    if user_frame is None:
        return None
    scope = dict(user_frame.f_globals)
    scope.update(user_frame.f_locals)
    return {name: scope.get(name) for name, _ in spec.applies}


def _select_state(loaded, key):
    """The slice of a checkpoint that belongs to one target.

    `key is None` is the flat idiom -- `torch.save(model.state_dict(), path)` --
    where the whole file is one object's state.
    """
    return loaded if key is None else loaded[key]


def _restore_from_mirror(
    name: str, iteration: Optional[int], value: Optional[str]
) -> bool:
    """Splice the obj_store mirror for one historical iteration into the user's
    objects: swap `layers` so _flor_torch_load redirects torch.load(spec.path)
    to the mirror file, then re-run the spec's load_state_dict calls.

    Raises rather than reporting a success it did not achieve. Every skip this
    used to swallow -- a name that no longer resolves, a target with no
    load_state_dict, a key the checkpoint doesn't carry -- leaves the object
    holding state from some other iteration, and the run goes on to log metrics
    off it as though they were historical.
    """
    global _restoring_mirror
    spec = cli.flags.resume_spec
    if spec is None or iteration is None or iteration < 0:
        return False
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    targets = _resolve_targets(spec)
    if targets is None:
        raise RuntimeError(
            f"FLOR: cannot restore {_ctx_description()}: no frame for "
            f"{SCRIPTNAME} on the stack, so the resume block's targets "
            f"({', '.join(n for n, _ in spec.applies)}) can't be reached. "
            f"Declare them with flor.restore({spec.path!r}, <name>=<obj>, ...)."
        )

    with _layer_swapped(name, iteration, value):
        stem, ext = _user_path_stem_ext(spec.path)
        mirror = obj_store.get_shelf() / utils.to_filename(layers, stem, ext)
        _restoring_mirror = mirror
        try:
            loaded = torch.load(spec.path)
        finally:
            _restoring_mirror = None
        applied, failures = [], []
        for target_name, key in spec.applies:
            target = targets.get(target_name)
            if target is None:
                failures.append(f"{target_name}: not found in {spec.source} scope")
                continue
            apply = getattr(target, "load_state_dict", None)
            if apply is None:
                failures.append(
                    f"{target_name}: {type(target).__name__} has no load_state_dict"
                )
                continue
            try:
                apply(_select_state(loaded, key))
            except Exception as e:
                failures.append(f"{target_name}: {type(e).__name__}: {e}")
                continue
            applied.append(target_name)
        if failures:
            hint = (
                ""
                if spec.source == "explicit"
                else f" flor inferred this mapping from {SCRIPTNAME}; if it is "
                f"wrong, declare it instead with "
                f"flor.restore({spec.path!r}, <name>=<obj>, ...)."
            )
            raise RuntimeError(
                f"FLOR: restoring {_ctx_description()} from {mirror.name} "
                f"failed for {len(failures)} of {len(spec.applies)} target(s): "
                f"{'; '.join(failures)}.{hint}"
            )
        return bool(applied)


def _build_outer_replay_plan(name: str, materialized: list):
    """Decide which outer-loop iters to run on replay.

    Returns (iter_source, logical_mirror_pos, silent_set, logical_active).

    Fast path: when every requested iter has its own obj_store mirror, return
    the narrowed slice as-is -- each iter restores from its own mirror.

    Logical replay: when at least one requested iter is missing its mirror
    (typically because ckpt_interval_s throttled the save), expand to a
    contiguous range starting just after the most-recent earlier mirror and
    ending at max(requested). Caller restores from that mirror at the first
    expanded iter, then fast-forwards through the body with inner loops run
    in full. Silent iters (those not in the user's request) have their logs
    suppressed; requested iters log normally.

    Fresh clone: when no mirror exists at or before the earliest requested iter,
    replay the loop from iteration 0. `.flor/runs/*.jsonl` is committed but
    `obj_store/` is not, so a teammate's clone has the observations and none of
    the checkpoints; iteration 0 is still reconstructible because the seed is a
    flor.arg restored from the historical run.
    """
    if not cli.flags.wev_found:
        # No flor.loop / no `with flor.checkpointing(...)` in the script --
        # nothing to narrow, run the loop end-to-end.
        return list(enumerate(materialized)), None, set(), False

    spec = cli.iter_spec_for(name)

    if spec.kind == "all":
        return list(enumerate(materialized)), None, set(), False
    if spec.kind == "none":
        return [], None, set(), False
    if spec.kind == "last":
        last_idx = len(materialized) - 1
        return [(last_idx, materialized[last_idx])], None, set(), False

    # spec.kind == "indices"
    n = len(materialized)
    out_of_range = [i for i in spec.indices if not (0 <= i < n)]
    if out_of_range:
        raise RuntimeError(
            f"FLOR: --iter {name}={list(spec.indices)} requests index "
            f"{out_of_range} but the loop only has {n} iteration(s) "
            f"(valid range: 0..{n - 1}). Re-run forward with more iterations "
            f"or narrow to an in-range index."
        )
    requested = list(spec.indices)
    if not requested:
        return [], None, set(), False

    resume = cli.flags.resume_spec
    if resume is None and not checkpoints:
        # Nothing to restore through either path -- narrowing is the whole plan.
        return [(i, materialized[i]) for i in requested], None, set(), False

    if all(_mirror_exists_at(name, r, materialized, resume) for r in requested):
        return [(i, materialized[i]) for i in requested], None, set(), False

    target_min = requested[0]
    mirror_pos = _find_latest_mirror_at_or_before(
        name, target_min, materialized, resume
    )
    if mirror_pos is None:
        # No mirror at or before the target -- the usual cause is a fresh clone,
        # which carries .flor/runs/*.jsonl (committed) but no obj_store (not).
        # Replaying from iteration 0 is sound only if the model at loop entry is
        # the seed-initialized one, and the seed is a flor.arg restored from the
        # historical run -- provided the script's initialization actually
        # survived to the loop. The user's own resume block runs at module scope,
        # where `layers` is still empty, so if their checkpoint file is on disk
        # it would have loaded *final*-epoch weights over the fresh init.
        # _neutralized_resume_state turns that block into a no-op precisely when
        # the shelf is empty; this checks that it did, rather than assuming, so
        # a resume shape flor can't neutralize refuses instead of guessing.
        if (
            resume is not None
            and resume.source != "explicit"
            and os.path.exists(resume.path)
            and not _resume_neutralized
        ):
            raise RuntimeError(
                f"FLOR: cannot replay {name}={list(requested)}: no checkpoint "
                f"mirror at or before position {target_min}, and {resume.path!r} "
                f"is present, so this script's resume block has already loaded "
                f"end-of-run state over its initialization -- iteration 0 is no "
                f"longer reconstructible. Move or delete {resume.path!r} to "
                f"replay from the top, or re-run forward to rebuild the mirrors."
            )
        capture.flor_print(
            f"FLOR: no checkpoint for {name}={list(requested)} at or before "
            f"position {target_min}; replaying from iteration 0. This recomputes "
            f"the intervening iterations (accurate only if the script seeds "
            f"deterministically) and shelves the checkpoints it passes, so "
            f"subsequent replays start from the nearest one."
        )
        requested_set = set(requested)
        expanded = list(range(0, requested[-1] + 1))
        silent = {i for i in expanded if i not in requested_set}
        return [(i, materialized[i]) for i in expanded], None, silent, True

    requested_set = set(requested)
    expanded = list(range(mirror_pos + 1, requested[-1] + 1))
    silent = {i for i in expanded if i not in requested_set}
    plan = [(i, materialized[i]) for i in expanded]
    return plan, mirror_pos, silent, True


def slice(name, iterator):
    if not cli.in_replay_mode():
        return iterator
    original = list(iterator)

    # During logical replay, every nested loop must run end-to-end so that
    # training (or whatever the iter body does) actually advances the state
    # the outer fast-forward depends on. User-supplied narrowing for inner
    # loops is intentionally overridden in this mode.
    if _logical_replay_active:
        return list(enumerate(original))

    if not cli.flags.wev_found:
        return list(enumerate(original))

    spec = cli.iter_spec_for(name)
    if spec.kind == "all":
        return list(enumerate(original))
    if spec.kind == "none":
        return []
    if spec.kind == "last":
        return [(len(original) - 1, original[-1])]
    # spec.kind == "indices"
    n = len(original)
    out_of_range = [i for i in spec.indices if not (0 <= i < n)]
    if out_of_range:
        raise RuntimeError(
            f"FLOR: --iter {name}={list(spec.indices)} requests index "
            f"{out_of_range} but the loop only has {n} iteration(s) "
            f"(valid range: 0..{n - 1})."
        )
    return [(i, original[i]) for i in spec.indices]


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
    "ckpt_interval_s",
    "set_capture",
]
