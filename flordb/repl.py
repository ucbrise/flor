import ast
import glob
import re
import shutil
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import subprocess
import tempfile
import os

from . import utils
from .cli import IterSpec
from .hlast.visitors import LoggedExpVisitor, WithExpVisitor
from .hlast import backprop

from . import database
from . import versions
from . import orm
from .constants import RUNS_DIR
from .clock import Clock


def dataframe(*args):
    conn, _ = database.conn_and_cursor()
    # Query the distinct value_names
    try:
        df = database.pivot(conn, *(args if args else tuple()))
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        return df
    finally:
        conn.close()


def io(channel: Optional[str] = None):
    """Captured print / logging output, as a dataframe.

    Automatically captured io is kept out of `dataframe()` on purpose -- it is
    text, not metrics -- so this is the way to read it back. Rows carry the
    same loop columns as any other record, plus `channel` and `line`.

        flor.io()                  # everything
        flor.io("io::stdout")      # prints only
        flor.io("io::log")         # every logging level
        flor.io("io::log::error")  # one level

    """
    conn, _ = database.conn_and_cursor()
    try:
        return database.read_io(conn, channel).reset_index(drop=True)
    finally:
        conn.close()


def query(user_query: str):
    conn, cursor = database.conn_and_cursor()
    try:
        df = database.query(cursor, user_query, aspandas=True)
        return df
    finally:
        # Close connection
        conn.commit()
        conn.close()


def _apply_var_is_lineno(v: str) -> bool:
    return v.startswith("@")


def _apply_var_to_lineno(v: str, lev: LoggedExpVisitor) -> int:
    """`@42` -> 42; bare names look up via the AST visitor."""
    if v.startswith("@"):
        return int(v[1:])
    if v not in lev.names:
        raise RuntimeError(
            f"FLOR: --apply {v!r} does not match any flor.log(...) in the current "
            f"script. Known names: {sorted(lev.names)}"
        )
    return lev.names[v]


def _apply_var_to_name(v: str, lev: LoggedExpVisitor) -> str:
    """`@42` -> the log name at line 42; bare names pass through (validated).

    Everything downstream of the AST -- the child's `--apply` projection, the
    Schedule columns, the final dataframe -- is keyed by log *name*, so linenos
    have to be resolved exactly once, here.
    """
    if not _apply_var_is_lineno(v):
        _apply_var_to_lineno(v, lev)  # validate
        return v
    lineno = _apply_var_to_lineno(v, lev)
    name = lev.linenos.get(lineno)
    if name is None:
        raise RuntimeError(
            f"FLOR: --apply {v!r} points at line {lineno}, which is not a "
            f"flor.log(...) call in the current script. Logged lines: "
            f"{sorted(lev.linenos)}"
        )
    return name


def replay(
    apply_vars: List[str],
    narrow_iters: Optional[List[Tuple[str, IterSpec]]] = None,
    where_clause: Optional[str] = None,
    overrides: Optional[List[Tuple[str, str]]] = None,
):
    """
    Re-run historical runs to compute apply_vars that weren't logged originally.

    apply_vars: log names (or `@LINENO`) introduced as hindsight statements in
        the current script. Each historical run is re-executed with these vars
        applied via backprop.
    narrow_iters: optional list of (loop_name, IterSpec) overrides forwarded
        to the inner script as `--iter NAME=SPEC` flags. If None, the
        orchestrator picks defaults based on where the apply_vars sit in the
        loop nesting.
    where_clause: optional SQL-style filter on the schedule (legacy; only used
        for column-name discovery today).
    overrides: optional (key, value) pairs forwarded to the inner script as
        `--override KEY=VALUE`. The child validates them against the historical
        run's flor.arg records (see cli.ENV_OVERRIDE_ALLOWLIST).
    """
    versions.git_commit("Hindsight logging stmts added.")

    jsonl_paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*.jsonl")))
    assert jsonl_paths, f"No runs found in {RUNS_DIR}; cannot replay."
    main_script = orm.read_jsonl(jsonl_paths[-1])[0]["filename"]
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    shutil.copy2(main_script, temp_file.name)
    with open(main_script, "r") as f:
        tree = ast.parse(f.read())
    lev, wev = LoggedExpVisitor(), WithExpVisitor()
    wev.visit(tree)
    lev.visit(tree)

    # Resolve `@LINENO` forms once, up front. `apply_linenos` drives backprop
    # (which is line-oriented); `apply_names` drives everything else: schedule
    # columns, the child's --apply projection, and the result dataframe.
    apply_linenos = [_apply_var_to_lineno(v, lev) for v in apply_vars]
    apply_names = [_apply_var_to_name(v, lev) for v in apply_vars]
    schedule = Schedule(apply_names, where_clause)

    if not wev.found:
        # No flor.loop and no `with flor.checkpointing(...):` in the script --
        # no narrowable scope. Full re-run.
        loglvl = 3
    else:
        loglvl = max(lev.line2level[ln] for ln in apply_linenos)
        # Cap to the number of flor.loops actually present so we don't try to
        # narrow a depth that doesn't exist.
        loglvl = min(loglvl, len(lev.loop_names))

    level_mapper = {0: "run-level (no loop)", 1: "outer loop", 2: "nested loop", 3: "full scan"}

    schedule.estimate_cost(loglvl, lev.loop_names)

    print(f"log level: {level_mapper.get(loglvl, str(loglvl))}")
    if narrow_iters:
        print(f"narrow (user-supplied): {[(n, s) for n, s in narrow_iters]}")
    print()
    print(schedule.df)
    print()

    res = input(
        f"Continue replay estimated to finish in {utils.discretize(sum(schedule.df['composite']))} [y/N]? "
    )
    res = res if res else "n"
    if res.lower().strip() == "n":
        return schedule

    clock = Clock()
    clock.set_start_time()

    active_branch = versions.current_branch()
    try:
        for projid, ts, hexsha, main_script in schedule.iter_dims():
            print("entering", str(ts), hexsha)
            versions.checkout(hexsha)
            for v, lineno in zip(apply_names, apply_linenos):
                print("applying: ", v, lineno)
                try:
                    backprop(lineno, temp_file.name, main_script, main_script)
                except Exception as e:
                    print("Exception raised during `backprop`", e)
                    raise e

            narrow_args = _narrow_args(loglvl, ts, schedule, lev, narrow_iters)
            cmd = [
                "python", main_script,
                "--replay_flor",
                "--apply", ",".join(apply_names),
            ]
            for name, spec in narrow_args:
                cmd += ["--iter", f"{name}={_spec_to_cli(spec)}"]
            for k, v in overrides or []:
                cmd += ["--override", f"{k}={v}"]
            print(*cmd)
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                # The child validates --override and --iter against the
                # historical run; a nonzero exit means it logged nothing, so
                # say so rather than letting the result df come back empty.
                print(
                    f"FLOR: replay of {ts} exited with code {proc.returncode}; "
                    f"no rows recorded for that run."
                )
    except Exception as e:
        print("Exception raised during `schedule.iter_dims()`", e)
        raise e
    finally:
        versions.reset_hard()
        versions.checkout(active_branch)
        os.remove(temp_file.name)

    dt = clock.get_delta()

    # apply_names is already lineno-free, so `@N` replays land in the result
    # dataframe under the name they were logged with.
    filtered_vs = list(apply_names)
    if schedule.vars_in_where is not None:
        filtered_vs += schedule.vars_in_where
    schedule = dataframe(*filtered_vs)

    print()
    print(schedule)
    print()
    print(dt, "seconds")

    return schedule


def _spec_to_cli(spec: IterSpec) -> str:
    if spec.kind in ("all", "last", "none"):
        return spec.kind
    return ",".join(str(i) for i in spec.indices)


def _narrow_args(
    loglvl: int,
    ts,
    schedule: "Schedule",
    lev: LoggedExpVisitor,
    user_narrow: Optional[List[Tuple[str, IterSpec]]],
) -> List[Tuple[str, IterSpec]]:
    """
    Pick the per-loop IterSpecs forwarded to the inner script as --iter flags.

    If the user passed `narrow_iters`, use it verbatim. Otherwise default by
    loglvl:
      loglvl 0 or 3 -> no narrowing (script-wide replay).
      loglvl 1      -> outer loop: every index; inner loops default to `last`
                       implicitly (cli.iter_spec_for falls back).
      loglvl 2      -> outer + every inner index up to (loglvl - 1) depths;
                       deepest loop (where the hindsight log sits) stays at
                       the implicit `last` default.
    """
    if user_narrow:
        return list(user_narrow)

    if loglvl == 0 or loglvl == 3:
        return []

    args: List[Tuple[str, IterSpec]] = []
    n_outer = int(
        schedule.df[schedule.df["tstamp"] == ts]["num_outer"].values[0]
    )
    outer_name = lev.loop_names[0] if lev.loop_names else "epoch"
    args.append((outer_name, IterSpec("indices", tuple(range(n_outer)))))

    # For loglvl >= 2 we also iterate inner loops up to (loglvl - 1) depths.
    # Deeper than that we leave at the implicit `last` default (one iter).
    for depth in range(1, loglvl - 1):
        if depth >= len(lev.loop_names):
            break
        # We don't know per-outer-iter how many inner iters there were; use a
        # wide range and let slice() bound it at runtime via the index filter.
        args.append((lev.loop_names[depth], IterSpec("indices", (0, 1))))
    return args


class Schedule:
    def __init__(self, apply_vars, where_clause) -> None:
        # apply_vars must be resolved log *names* -- `@LINENO` forms are
        # translated by _apply_var_to_name before we get here, because these
        # strings are used directly as dataframe column labels.
        self.apply_vars = apply_vars
        self.where_clause = where_clause
        self.vars_in_where = None
        if where_clause is not None:
            # Regular expression to match column names
            column_pattern = re.compile(r"\b[A-Za-z_]\w*\b")
            columns = set(re.findall(column_pattern, where_clause))

            # Convert to list if needed
            columns_list = list(columns)
            self.vars_in_where = columns_list
            print("columns in where_clause:", columns_list)

    def estimate_cost(self, loglvl: int, loop_names: List[str]):
        """
        Build self.df: one row per historical tstamp with a `composite` column
        giving a wall-time estimate for the narrowed re-run.

        loop_names: outermost-to-innermost flor.loop names from the AST. Used
            to generalize the schedule beyond the legacy "epoch"/"step" pair.
        """
        keys = ["projid", "tstamp", "filename"]
        pvt = dataframe()

        if loglvl == 3:
            # Full scan -- whole-run wall time is the only useful estimate.
            df = dataframe("time::script")
            df["composite"] = pd.to_numeric(df["time::script"])
            self.df = pd.merge(pvt, df, on=keys, how="inner")
            return

        if loglvl == 0:
            # Apply var sits before any flor.loop -- setup time is what dominates.
            df = dataframe("time::setup", "time::teardown")
            df["composite"] = pd.to_numeric(df["time::setup"]) + pd.to_numeric(
                df["time::teardown"]
            )
            self.df = pd.merge(pvt, df, on=keys, how="inner")
            return

        # loglvl in {1, 2}: narrowed loop replay. Estimate = setup + outer-loop
        # wall time + teardown. Outer-loop wall time comes from the time::loop
        # record at the run level (ctx is null). We MAX-aggregate because the
        # DB may have multiple inserts per tstamp from prior replays; the
        # original (slowest) run is the safe upper bound.
        outer_name = loop_names[0] if loop_names else "epoch"
        # Cost estimation is a baseline for "how long would re-running take?"
        # Replay rows reflect a narrowed run (fewer iters, projected vars) and
        # would make the baseline misleadingly fast; filter to forward only.
        outer_loop_wall = query(
            "SELECT projid, tstamp, filename, MAX(CAST(value AS REAL)) AS outer_loop_s "
            "FROM logs WHERE ctx IS NULL AND value_name = 'time::loop' "
            "AND source = 'forward' "
            "GROUP BY projid, tstamp, filename;"
        )

        # num_outer: how many iterations the outermost loop ran. Prefer the
        # inner-loop ctx records (each carries the outer iter index); fall
        # back to the outermost-loop's `time::iter::n` summary when there's
        # no nested loop (ctx IS NULL pinpoints the outermost aggregate).
        loops_df = dataframe("time::loop")
        if outer_name in loops_df.columns:
            num_outer = (
                loops_df.dropna(subset=[outer_name])
                .drop_duplicates(subset=keys + [outer_name])
                .groupby(keys)
                .agg(num_outer=(outer_name, "max"))
                .reset_index()
            )
            # `iteration` is 0-indexed (matches positional narrowing); count
            # of outer iters is therefore max(iteration) + 1.
            num_outer["num_outer"] = pd.to_numeric(num_outer["num_outer"]) + 1
        else:
            iters = query(
                "SELECT projid, tstamp, filename, "
                "MAX(CAST(value AS INTEGER)) AS num_outer "
                "FROM logs WHERE ctx IS NULL AND value_name = 'time::iter::n' "
                "AND source = 'forward' "
                "GROUP BY projid, tstamp, filename;"
            )
            num_outer = iters

        # Edge timings: also MAX-dedupe (one ctx-null time::setup record per
        # tstamp originally, but replays may have added more).
        edges = query(
            "SELECT projid, tstamp, filename, "
            "MAX(CASE WHEN value_name='time::setup'    THEN CAST(value AS REAL) END) AS setup_s, "
            "MAX(CASE WHEN value_name='time::teardown' THEN CAST(value AS REAL) END) AS teardown_s "
            "FROM logs WHERE ctx IS NULL AND value_name IN ('time::setup','time::teardown') "
            "AND source = 'forward' "
            "GROUP BY projid, tstamp, filename;"
        )

        merged = pd.merge(outer_loop_wall, num_outer, on=keys, how="inner")
        merged = pd.merge(merged, edges, on=keys, how="inner")
        merged["composite"] = (
            merged["setup_s"].fillna(0)
            + merged["outer_loop_s"]
            + merged["teardown_s"].fillna(0)
        )
        self.df = pd.merge(pvt, merged, on=keys, how="inner")

    def is_empty(self):
        return self.df.empty

    def iter_dims(self):
        ts2vid = {
            pd.Timestamp(ts): str(vid)
            for ts, vid, _ in versions.get_latest_autocommit()
        }

        prev_row = None

        for row_dict in self.df.to_dict(orient="records"):
            curr_tstamp = row_dict["tstamp"]

            # Compare current timestamp with the previous timestamp
            if prev_row is not None and curr_tstamp != prev_row["tstamp"]:
                yield prev_row["projid"], prev_row["tstamp"], ts2vid[
                    prev_row["tstamp"]
                ], prev_row["filename"]

            # Update prev_row for the next iteration
            prev_row = row_dict

        # Yield the final record if prev_row is populated
        if prev_row is not None:
            yield prev_row["projid"], prev_row["tstamp"], ts2vid[
                prev_row["tstamp"]
            ], prev_row["filename"]

    def __str__(self):
        if self.where_clause is None:
            return self.df[self.df[self.apply_vars].isna().any(axis=1)].__str__()  # type: ignore
        else:
            schedule = self.df.copy()  # appease pandas warning
            schedule[[v for v in self.apply_vars if v not in schedule.columns]] = np.nan
            return schedule.__str__()

    def __repr__(self):
        if self.where_clause is None:
            return self.df[self.df[self.apply_vars].isna().any(axis=1)].__repr__()  # type: ignore
        else:
            schedule = self.df.copy()  # appease pandas warning
            schedule[[v for v in self.apply_vars if v not in schedule.columns]] = np.nan
            return schedule.__repr__()

    def _repr_html_(self):
        if self.where_clause is None:
            return self.df[self.df[self.apply_vars].isna().any(axis=1)]._repr_html_()  # type: ignore
        else:
            schedule = self.df.copy()  # appease pandas warning
            schedule[[v for v in self.apply_vars if v not in schedule.columns]] = np.nan
            return schedule._repr_html_()  # type: ignore
