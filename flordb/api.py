import inspect
import os
import shlex
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

skip_cleanup = True


def set_ckpt_interval(seconds: float) -> None:
    global ckpt_interval_s
    ckpt_interval_s = float(seconds)


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
            3,
        )
    )
    _setup_emitted = True


def _mark_main_segment_end():
    global _last_main_exit_time
    _last_main_exit_time = time.perf_counter()


def _ctx_snapshot() -> Optional[List[orm.Segment]]:
    return list(context) if context else None


def log(name, value):
    if skip_cleanup:
        _deferred_init()

    serializable_value = value if utils.is_jsonable(value) else str(value)
    tqdm.write(utils.to_string(layers, name, serializable_value))

    if cli.in_replay_mode() and name not in cli.flags.queryparameters["VARS"]:
        # Check that name is in logging statement propagation list
        return value

    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            _ctx_snapshot(),
            name,
            serializable_value,
            1,
        )
    )

    return value


def arg(name: str, default: Optional[Any] = None) -> Any:
    if cli.in_replay_mode():
        # GIT
        assert name in cli.flags.hyperparameters
        historical_v = cli.flags.hyperparameters[name]
        log(name, historical_v)
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
        raise


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
        print(f"An error occurred: {e}")
        raise
    finally:
        checkpoints.clear()


@contextmanager
def iteration(name: str, idx: Optional[int], value: Optional[str]):
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
    if cli.in_replay_mode():
        # TODO: load the end-state checkpoint
        load_ckpt()
        raise
    context.append(orm.Segment(name, layers[name][0], layers[name][1]))
    try:
        yield
        ckpt()
    finally:
        context.pop()
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            _ctx_snapshot(),
            "time::iter",
            clock.get_delta(),
            3,
        )
    )
    if pos == 0:
        _mark_main_segment_end()
    del layers[name]


def loop(name: str, iterator: Iterable[T]) -> Iterator[T]:
    global _last_ckpt_time
    _deferred_init()
    pos = len(layers)
    if pos == 0:
        _emit_setup_once()
        # Reset so the first iter's ckpt always fires; later iters get
        # throttled by the time guard.
        _last_ckpt_time = None
    clock = Clock()
    clock.set_start_time()
    layers[name] = (0, None)
    context.append(orm.Segment(name, 0, None))
    # On replay we materialize so _auto_restore can index into the previous
    # iter's value for the mirror filename lookup. On forward we keep the
    # original lazy iterator semantics.
    if cli.in_replay_mode():
        materialized: Optional[list] = list(iterator)
        iter_source: Any = slice(name, materialized)
    else:
        materialized = None
        iter_source = enumerate(iterator)
    for each in tqdm(
        iter_source,
        position=pos,
        leave=(True if pos == 0 else False),
    ):
        layers[name] = (
            int(each[0]) + 1,
            str(each[1]) if utils.is_jsonable(each[1]) else None,
        )
        context[-1] = orm.Segment(name, layers[name][0], layers[name][1])
        if pos == 0 and cli.in_replay_mode():
            load_ckpt()
            if materialized is not None:
                _auto_restore(name, int(each[0]), materialized)
        iter_clock = Clock()
        iter_clock.set_start_time()
        yield each[1]  # type: ignore
        output_buffer.append(
            orm.Log(
                PROJID,
                Clock.get_datetime(),
                SCRIPTNAME,
                _ctx_snapshot(),
                "time::iter",
                iter_clock.get_delta(),
                3,
            )
        )
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
    output_buffer.append(
        orm.Log(
            PROJID,
            Clock.get_datetime(),
            SCRIPTNAME,
            _ctx_snapshot(),
            "time::loop",
            clock.get_delta(),
            3,
        )
    )
    if pos == 0:
        _mark_main_segment_end()
    del layers[name]


def commit():
    global skip_cleanup, _setup_emitted, _last_main_exit_time
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
            3,
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
                3,
            )
        )
    conn, cursor = database.conn_and_cursor()
    if not cli.in_replay_mode():
        # RECORD
        branch = versions.current_branch()
        if branch is not None:
            orm.to_jsonl(output_buffer, tstamp)
            database.unpack(output_buffer, cursor)
            _write_cmd_file(tstamp)
            versions.git_commit(_build_commit_message(tstamp, run_args))
    else:
        database.unpack(output_buffer, cursor)
    conn.commit()
    conn.close()
    output_buffer.clear()
    run_args.clear()
    Clock.set_new_datetime()
    _setup_emitted = False
    _last_main_exit_time = None
    skip_cleanup = True


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
# `X.load_state_dict(loaded[key])` pattern in the user's script, flor.loop
# calls _auto_restore at the start of each replayed outer-iter to splice the
# matching obj_store mirror back into the user's module/function frame. The
# user's own resume code (e.g. line 84 of v4/train.py) is left intact -- it
# still runs once before the loop -- but its result is overwritten per-iter.
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


def _auto_restore(name: str, position: int, materialized: list) -> None:
    spec = cli.flags.resume_spec
    if spec is None or position <= 0:
        return
    user_frame = _find_user_frame()
    if user_frame is None:
        return
    try:
        import torch  # type: ignore
    except ImportError:
        return

    saved_layer = layers.get(name)
    try:
        # Walk backward over the original iteration positions to find the
        # most recent mirror that actually exists on disk (handles throttled
        # saves where ckpt_interval_s skipped some iters).
        found_layer = None
        for k in range(position - 1, -1, -1):
            prev_val = materialized[k]
            v = str(prev_val) if utils.is_jsonable(prev_val) else None
            layers[name] = (k + 1, v)
            try:
                stem, ext = _user_path_stem_ext(spec.path)
                candidate = obj_store.get_shelf() / utils.to_filename(
                    layers, stem, ext
                )
                if candidate.exists():
                    found_layer = (k + 1, v)
                    break
            except Exception:
                continue
        if found_layer is None:
            return
        layers[name] = found_layer
        # _flor_torch_load consults `layers` -- the swap above redirects the
        # read to the historical mirror without changing the user-visible
        # path string.
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
    finally:
        if saved_layer is not None:
            layers[name] = saved_layer


def slice(name, iterator):
    if not cli.in_replay_mode():
        return iterator
    original = list(iterator)

    qop = (
        (cli.flags.queryparameters).get(name, 0)
        if cli.flags.queryparameters is not None
        else 0
    )
    if qop == 1 or not cli.flags.queryparameters["WEV"]:
        return enumerate(iterator)

    new_slice = []
    if qop == 0:
        new_slice.append((len(original) - 1, original[-1]))
        return new_slice

    assert isinstance(qop, (list, tuple))
    qop = [i for i in qop if i.isnumeric()]
    for i in qop:
        new_slice.append((i, original[int(i)]))
    return new_slice


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
]
