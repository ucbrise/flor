"""Automatic capture of `print` / `logging` output.

The point of this module is the zero-code-change onboarding promised by the
Least Redundancy Overhaul: a script whose only flor reference is
`import flordb` still gets its io indexed, addressed by the same loop `ctx`
as `flor.log` records.

"Least redundancy" is load-bearing, and it cuts against the naive
implementation twice:

  * flor echoes every `flor.log` value to the terminal itself
    (`tqdm.write` in `api.log`). Teeing stdout without care would record every
    metric a second time, as text.
  * a `logging` call that reaches a `StreamHandler` shows up on stderr too, so
    a tee plus a `logging.Handler` would record it twice.

Both are solved by `muted()`, a thread-local flag that degrades the tee to pure
passthrough. flor's own writes go through `flor_print`, and the whole `logging`
dispatch is wrapped, so a record is captured once -- from the structured side,
where the level and logger name still exist.

This module deliberately knows nothing about `api`, the object store, or the
database. It turns stream writes into `(channel, text)` pairs and hands them to
a sink installed by `api`, which owns ctx, replay gating, and serialization.
"""

import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

STDOUT = "io::stdout"
STDERR = "io::stderr"
LOG_PREFIX = "io::log::"

TRUNCATION_MARKER = " ...[flor: line truncated]"


@dataclass
class Config:
    enabled: bool = True
    # Longest line recorded verbatim. Anything past this is cut, so one runaway
    # print (a serialized model, a stack of tensors) can't dominate the run's
    # JSONL.
    max_line: int = 4096
    # Ceiling on records per run. A per-step print in a long training loop
    # would otherwise produce millions of rows.
    max_records: int = 10000


config = Config()

_sink: Optional[Callable[[str, str], None]] = None
_installed: bool = False
_saved: Optional[tuple] = None
_records_emitted: int = 0
_cap_notified: bool = False

_local = threading.local()


def _muted() -> bool:
    return getattr(_local, "muted", False)


@contextmanager
def muted():
    """Pass writes through to the terminal without recording them.

    Thread-local: a dataloader worker printing while the main thread is inside
    a muted block should still be captured.
    """
    prev = getattr(_local, "muted", False)
    _local.muted = True
    try:
        yield
    finally:
        _local.muted = prev


def flor_print(*args, **kwargs):
    """`print` for flor's own messages -- shown, never recorded."""
    with muted():
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# stream tee
# ---------------------------------------------------------------------------


class _Tee:
    """File-like proxy that forwards to the real stream and buffers lines.

    Writes reach the wrapped stream first and unconditionally: capture is an
    observer, and a bug here must never cost the user their terminal output.
    """

    def __init__(self, wrapped, channel: str):
        self._wrapped = wrapped
        self._channel = channel
        self._buf = ""

    # -- file protocol ------------------------------------------------------

    def write(self, s):
        result = self._wrapped.write(s)
        if s and config.enabled and not _muted():
            try:
                self._absorb(s)
            except Exception:
                # Never let capture break the write path.
                pass
        return result

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        return self._wrapped.flush()

    def isatty(self):
        return self._wrapped.isatty()

    def fileno(self):
        return self._wrapped.fileno()

    def __getattr__(self, name):
        # tqdm, rich, and friends probe for all sorts of stream attributes.
        return getattr(self._wrapped, name)

    # -- capture ------------------------------------------------------------

    def _absorb(self, chunk: str):
        self._buf += chunk
        if "\n" not in self._buf:
            self._buf = _clamp_partial(self._buf)
            return
        *lines, self._buf = self._buf.split("\n")
        self._buf = _clamp_partial(self._buf)
        for line in lines:
            _emit(self._channel, line)

    def drain(self):
        """Record whatever is buffered without a trailing newline."""
        if self._buf:
            line, self._buf = self._buf, ""
            _emit(self._channel, line)

    @property
    def wrapped(self):
        return self._wrapped


def _clamp_partial(buf: str) -> str:
    """Keep an unterminated buffer bounded.

    A progress bar repaints with `\\r` and may never emit a newline, so the
    buffer would otherwise grow for the length of the run.
    """
    if "\r" in buf:
        # Everything before the final carriage return has been overdrawn; only
        # the last repaint is still on screen.
        buf = buf.rsplit("\r", 1)[-1]
    if len(buf) > config.max_line:
        buf = buf[: config.max_line]
    return buf


def _emit(channel: str, line: str):
    global _records_emitted, _cap_notified
    if "\r" in line:
        line = line.rsplit("\r", 1)[-1]
    line = line.rstrip()
    if not line:
        return
    if len(line) > config.max_line:
        line = line[: config.max_line] + TRUNCATION_MARKER
    if _records_emitted >= config.max_records:
        if not _cap_notified:
            _cap_notified = True
            flor_print(
                f"FLOR: captured {config.max_records} io records this run; "
                f"dropping the rest. Raise the ceiling with "
                f"flor.set_capture(max_records=...) or turn capture off with "
                f"flor.set_capture(False)."
            )
        return
    _records_emitted += 1
    if _sink is not None:
        _sink(channel, line)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

_orig_handle = None


def _flor_handle(logger, record):
    """Wrap the whole `logging` dispatch, not just one handler.

    Registering a `logging.Handler` would put us in handler order alongside the
    user's `StreamHandler`, whose stderr write the tee would then record as a
    second row. Wrapping `Logger.handle` mutes the tee for the entire dispatch
    instead, so the record is captured exactly once -- here, where `levelname`
    and the logger name are still available. Level and filter decisions have
    already been made by the caller, so the user's logging configuration is
    respected.
    """
    assert _orig_handle is not None
    if not config.enabled or _muted():
        return _orig_handle(logger, record)
    with muted():
        try:
            message = record.getMessage()
            if record.name and record.name != "root":
                message = f"{record.name}: {message}"
            channel = LOG_PREFIX + str(record.levelname).lower()
        except Exception:
            channel, message = None, None
        if channel is not None and message is not None:
            for line in str(message).splitlines() or [""]:
                _emit(channel, line)
        return _orig_handle(logger, record)


# ---------------------------------------------------------------------------
# structuring
# ---------------------------------------------------------------------------

# Names recognized as a loop index rather than a metric. The vocabulary is
# fixed on purpose: an open-ended rule would read `Loading 5 files` as an index
# named `Loading` and invent a loop the user never wrote. These five cover the
# step-prefix convention as ML logs actually write it.
STEP_NAMES = ("epoch", "step", "iteration", "iter", "batch")

# `epoch 0 | ...`, `Epoch 1/10`, `step: 1200 ...`. Anchored at the start of the
# line, because that position is what makes the number an index and not a
# measurement -- `loss 3` mid-line is a value, `epoch 3 |` leading is a key.
_STEP_RE = re.compile(
    r"^\s*(" + "|".join(STEP_NAMES) + r")"
    r"(?:\s*[:=]\s*|\s+)"
    r"(\d+)"
    r"(?:\s*/\s*\d+)?"   # `1/10` -- the total is context, not data
    r"(?![\w.])",
    re.IGNORECASE,
)

# A key, then `:` or `=`, then a number. The lookarounds are what keep this
# conservative: the left one rejects `http://host:80` (preceded by `/`) and
# `2026-08-13T11:27:06` (preceded by a word char or `-`); the right one rejects
# units and paths, so `acc: 90%` is left alone rather than recorded as 90.
_PAIR_RE = re.compile(
    r"(?<![\w./:@-])"
    r"([A-Za-z_][A-Za-z0-9_.\-]*)"
    r"\s*[:=]\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?![\w./:%-])"
)

MAX_PAIRS_PER_LINE = 8


def extract_pairs(line: str) -> List[Tuple[str, float]]:
    """Recognize `k: v` / `k=v` metric pairs in a line of free text.

    Deliberately conservative -- a false positive invents a column in
    `flor.dataframe` -- and capped per line so a printed JSON blob doesn't
    explode into dozens of metrics.
    """
    out: List[Tuple[str, float]] = []
    seen = set()
    for match in _PAIR_RE.finditer(line):
        key, raw = match.group(1), match.group(2)
        if key in seen:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        seen.add(key)
        out.append((key, value))
        if len(out) >= MAX_PAIRS_PER_LINE:
            break
    return out


@dataclass
class Extracted:
    """What one line of text yields: at most one index, any number of measures.

    The split is the whole point. An index addresses a row (`epoch`), a measure
    fills a cell (`loss`). Recording an index as a measure makes it a peer of
    `loss` in `database.pivot`, which joins per-variable frames on their common
    columns -- with nothing to join on, three epochs and three losses come back
    as nine rows instead of three.
    """

    index: Optional[Tuple[str, int]] = None
    measures: List[Tuple[str, float]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.index is not None or bool(self.measures)


def extract_fields(line: str) -> Extracted:
    """Split a line of free text into a loop index and metric values.

    `epoch 0 | loss: 0.5000` -> index ("epoch", 0), measures [("loss", 0.5)].

    An index is found two ways: the leading step prefix `_STEP_RE` recognizes,
    and a `k: v` pair whose key is a step name carrying a whole number. Both
    resolve to the same slot, so `epoch 3 | ...` and `epoch: 3 | ...` classify
    alike, and a line offering two indexes keeps the leading one.
    """
    index: Optional[Tuple[str, int]] = None

    match = _STEP_RE.match(line)
    if match is not None:
        index = (match.group(1).lower(), int(match.group(2)))

    measures: List[Tuple[str, float]] = []
    for key, value in extract_pairs(line):
        if key.lower() in STEP_NAMES and float(value).is_integer() and value >= 0:
            if index is None:
                index = (key.lower(), int(value))
            elif index[0] == key.lower():
                # `epoch 3` matched the prefix and `epoch: 3` matched the pair
                # rule on the same line; one index, not an index and a metric.
                pass
            else:
                measures.append((key, value))
            continue
        measures.append((key, value))

    return Extracted(index=index, measures=measures)


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def env_disabled() -> bool:
    return os.environ.get("FLOR_CAPTURE", "").strip().lower() in ("0", "false", "no")


def install(sink: Callable[[str, str], None]) -> bool:
    """Tee the standard streams and wrap `logging` dispatch. Idempotent."""
    global _sink, _installed, _saved, _orig_handle
    if _installed:
        return False
    if env_disabled():
        config.enabled = False
        return False
    _sink = sink
    _orig_handle = logging.Logger.handle
    _saved = (sys.stdout, sys.stderr, _orig_handle)
    sys.stdout = _Tee(sys.stdout, STDOUT)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr, STDERR)  # type: ignore[assignment]
    logging.Logger.handle = _flor_handle  # type: ignore[assignment]
    _installed = True
    return True


def flush():
    """Record trailing partial lines. Safe to call repeatedly."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, _Tee):
            stream.drain()


def reset_run_state():
    """Clear the per-run record ceiling. Called when a run commits."""
    global _records_emitted, _cap_notified
    _records_emitted = 0
    _cap_notified = False


def uninstall():
    global _sink, _installed, _saved
    if not _installed:
        return
    flush()
    assert _saved is not None
    sys.stdout, sys.stderr, logging.Logger.handle = _saved  # type: ignore[assignment]
    _saved = None
    _sink = None
    _installed = False


def installed() -> bool:
    return _installed


def raw_stream(stream):
    """The real stream behind a tee, for writers whose output flor shouldn't
    record (its own tqdm progress bars)."""
    return stream.wrapped if isinstance(stream, _Tee) else stream


def raw_stderr():
    return raw_stream(sys.stderr)


def is_io_channel(name: str) -> bool:
    return name in (STDOUT, STDERR) or name.startswith(LOG_PREFIX)
