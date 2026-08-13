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

CMD_FILE = os.path.join(CURRDIR, ".flor.cmd")

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
    extract      -- promote recognized `k: v` pairs to metric rows. Off by
                    default; preview what it would do with `flor capture
                    --preview` before turning it on.
    """
    if enabled is not None:
        capture.config.enabled = bool(enabled)
    if max_line is not None:
        capture.config.max_line = int(max_line)
    if max_records is not None:
        capture.config.max_records = int(max_records)
    if extract is not None:
        capture.config.extract = bool(extract)


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

    # Gated separately from the raw line: `--apply loss` on a print-only script
    # means the user wants the extracted `loss`, not the text it came from.
    if capture.config.extract:
        for name, value in capture.extract_pairs(text):
            if _recording(name):
                output_buffer.append(
                    orm.Log(
                        PROJID,
                        Clock.get_datetime(),
                        SCRIPTNAME,
                        ctx,
                        name,
                        value,
                        VALUE_TYPE_LOG,
                    )
                )
                buffered = True

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
        if not replaying:
            # Replay reads from the object store keyed on the *historical*
            # tstamp (obj_store.get_shelf); writing there would overwrite the
            # mirrors the replay is reading from.
            ckpt()
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
            load_ckpt()
            if materialized is not None:
                if _logical_replay_active:
                    _suppress_logs = int(each[0]) in logical_silent
                    if first_outer_iter and logical_mirror_pos is not None:
                        _restore_from_mirror(
                            name, *_layer_for(materialized, logical_mirror_pos)
                        )
                else:
                    _suppress_logs = False
                    _restore_from_mirror(
                        name, *_layer_for(materialized, int(each[0]))
                    )
            first_outer_iter = False
        iter_clock = Clock()
        iter_clock.set_start_time()
        yield each[1]  # type: ignore
        iter_deltas.append(iter_clock.get_delta())
        if pos == 0 and not cli.in_replay_mode():
            now = time.perf_counter()
            if _last_ckpt_time is None or (now - _last_ckpt_time) >= ckpt_interval_s:
                ckpt()
                _last_ckpt_time = now
    if pos == 0 and not cli.in_replay_mode():
        # Force a final checkpoint at outermost loop exit so end-of-run state
        # is always captured, regardless of the time guard.
        ckpt()
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
            versions.ensure_gitignored(".flor/")
            versions.to_shadow()
    _install_torch_hooks()


def ckpt():
    for name, obj in checkpoints:
        obj_store.serialize(layers, name, obj)


def load_ckpt():
    for name, obj in checkpoints:
        obj_store.deserialize(layers, name, obj)


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
    if not layers or cli.in_replay_mode():
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
        _orig_torch_save(obj, str(flor_path), *args, **kwargs)
        _last_ckpt_time = now
    except Exception:
        pass
    return result


def _flor_torch_load(path, *args, **kwargs):
    assert _orig_torch_load is not None
    if cli.in_replay_mode() and layers:
        try:
            stem, ext = _user_path_stem_ext(path)
            flor_path = obj_store.get_shelf() / utils.to_filename(layers, stem, ext)
            if flor_path.exists():
                return _orig_torch_load(str(flor_path), *args, **kwargs)
        except Exception:
            pass
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
    try:
        return _mirror_path_for(name, k, materialized, spec).exists()
    except Exception:
        return False


def _find_latest_mirror_at_or_before(
    name: str, position: int, materialized: list, spec
) -> Optional[int]:
    for k in range(position, -1, -1):
        if _mirror_exists_at(name, k, materialized, spec):
            return k
    return None


def _restore_from_mirror(
    name: str, iteration: Optional[int], value: Optional[str]
) -> bool:
    """Splice the obj_store mirror for one historical iteration into the user's
    frame: swap `layers` so _flor_torch_load redirects torch.load(spec.path) to
    the mirror file, then re-run the user's load_state_dict calls.
    """
    spec = cli.flags.resume_spec
    if spec is None or iteration is None or iteration < 0:
        return False
    user_frame = _find_user_frame()
    if user_frame is None:
        return False
    try:
        import torch  # type: ignore
    except ImportError:
        return False

    with _layer_swapped(name, iteration, value):
        loaded = torch.load(spec.path)
        scope = dict(user_frame.f_globals)
        scope.update(user_frame.f_locals)
        for target_name, key in spec.applies:
            target = scope.get(target_name)
            if target is None:
                continue
            apply = getattr(target, "load_state_dict", None)
            if apply is None:
                continue
            try:
                apply(loaded[key])
            except Exception:
                pass
        return True


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

    Aborts loudly if no mirror exists at or before the earliest requested
    iter -- there is no earlier state to start from.
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
    if resume is None:
        return [(i, materialized[i]) for i in requested], None, set(), False

    if all(_mirror_exists_at(name, r, materialized, resume) for r in requested):
        return [(i, materialized[i]) for i in requested], None, set(), False

    target_min = requested[0]
    mirror_pos = _find_latest_mirror_at_or_before(
        name, target_min, materialized, resume
    )
    if mirror_pos is None:
        raise RuntimeError(
            f"FLOR: cannot replay {name}={list(requested)}: no checkpoint mirror "
            f"found at or before position {target_min}. The historical run "
            f"(ckpt_interval_s={ckpt_interval_s}s) may have discarded every "
            f"earlier mirror -- re-run forward with a lower interval to "
            f"enable replay from this point."
        )

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
    "loop",
    "iteration",
    "commit",
    "output_buffer",
    "set_ckpt_interval",
    "ckpt_interval_s",
    "set_capture",
]
