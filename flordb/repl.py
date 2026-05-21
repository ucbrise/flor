import ast
import glob
import re
import shutil
import numpy as np
import pandas as pd
from typing import List, Optional
import subprocess
import tempfile
import os

from . import utils
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


def query(user_query: str):
    conn, cursor = database.conn_and_cursor()
    try:
        df = database.query(cursor, user_query, aspandas=True)
        return df
    finally:
        # Close connection
        conn.commit()
        conn.close()


def replay(
    apply_vars: List[str],
    narrow: Optional[str] = None,
    where_clause: Optional[str] = None,
):
    """
    Re-run historical runs to compute apply_vars that weren't logged originally.

    apply_vars: log names (or linenos) introduced as hindsight statements in the
        current script. Each historical run is re-executed with these vars
        applied via backprop.
    narrow: optional `--replay_flor`-style narrowing spec, e.g. "epoch=2" or
        "epoch=2 step=". Pass-through to the subprocess. If None, the
        orchestrator picks a default based on where in the loop nesting the
        apply_vars sit.
    where_clause: optional SQL-style filter on the schedule (legacy; only used
        for column-name discovery today).
    """
    versions.git_commit("Hindsight logging stmts added.")
    schedule = Schedule(apply_vars, where_clause)

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

    if not wev.found:
        # No flor.loop and no `with flor.checkpointing(...):` in the script --
        # no narrowable scope. Full re-run.
        loglvl = 3
    else:
        loglvl = max(lev.line2level[lev.names[v]] for v in apply_vars)
        # Cap to the number of flor.loops actually present so we don't try to
        # narrow a depth that doesn't exist.
        loglvl = min(loglvl, len(lev.loop_names))

    level_mapper = {0: "run-level (no loop)", 1: "outer loop", 2: "nested loop", 3: "full scan"}

    schedule.estimate_cost(loglvl, lev.loop_names)

    print(f"log level: {level_mapper.get(loglvl, str(loglvl))}")
    if narrow:
        print(f"narrow (user-supplied): {narrow}")
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
            for v, lineno in zip(
                apply_vars,
                [int(v) if utils.is_integer(v) else lev.names[v] for v in apply_vars],
            ):
                print("applying: ", v, lineno)
                try:
                    backprop(lineno, temp_file.name, main_script, main_script)
                except Exception as e:
                    print("Exception raised during `backprop`", e)
                    raise e

            narrow_args = _narrow_args(loglvl, ts, schedule, lev, narrow)
            cmd = ["python", main_script, "--replay_flor", ",".join(apply_vars)] + (
                narrow_args
            )
            print(*cmd)
            subprocess.run(cmd)
    except Exception as e:
        print("Exception raised during `schedule.iter_dims()`", e)
        raise e
    finally:
        versions.reset_hard()
        versions.checkout(active_branch)
        os.remove(temp_file.name)

    dt = clock.get_delta()

    filtered_vs = [v for v in apply_vars if not utils.is_integer(v)]
    if schedule.vars_in_where is not None:
        filtered_vs += schedule.vars_in_where
    schedule = dataframe(*filtered_vs)

    print()
    print(schedule)
    print()
    print(dt, "seconds")

    return schedule


def _narrow_args(
    loglvl: int,
    ts,
    schedule: "Schedule",
    lev: LoggedExpVisitor,
    user_narrow: Optional[str],
) -> List[str]:
    """
    Build the loop-narrowing key=value list appended to --replay_flor.

    If the user passed a `narrow` string, use it verbatim (space-separated).
    Otherwise pick a default based on loglvl:
      loglvl 0 or 3 -> no narrowing.
      loglvl 1      -> iterate every index of the outermost flor.loop,
                       inner loops default to last-only via slice().
      loglvl 2      -> iterate every outer index AND every inner index up to
                       the depth above loglvl; the deepest one (where the
                       hindsight log sits) defaults to last-only.
    """
    if user_narrow:
        return [s for s in user_narrow.split() if "=" in s]

    if loglvl == 0 or loglvl == 3:
        return []

    args: List[str] = []
    # Resolve how many iterations the outermost loop had in this run.
    n_outer = int(
        schedule.df[schedule.df["tstamp"] == ts]["num_outer"].values[0]
    )
    outer_name = lev.loop_names[0] if lev.loop_names else "epoch"
    tup = ",".join(str(i) for i in range(n_outer)) + ","
    args.append(f"{outer_name}={tup}")

    # For loglvl >= 2 we also iterate inner loops up to (loglvl - 1) depths.
    # Deeper than that we leave at the slice() default (last-only).
    for depth in range(1, loglvl - 1):
        if depth >= len(lev.loop_names):
            break
        # We don't yet know per-outer-iter how many inner iters there were;
        # fall back to a wide range. The slice() default for unknown numeric
        # indices is to pick original[int(i)], which is bounded by the actual
        # length at runtime.
        args.append(f"{lev.loop_names[depth]}=0,1,")
    return args


class Schedule:
    def __init__(self, apply_vars, where_clause) -> None:
        # TODO:
        # case when integer supplied through apply_vars,
        #     you will need to infer var_name from ast
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
        outer_loop_wall = query(
            "SELECT projid, tstamp, filename, MAX(CAST(value AS REAL)) AS outer_loop_s "
            "FROM logs WHERE ctx IS NULL AND value_name = 'time::loop' "
            "GROUP BY projid, tstamp, filename;"
        )

        # num_outer: how many iterations the outermost loop ran. Prefer the
        # inner-loop ctx records (each carries the outer iter index); fall
        # back to counting time::iter records when there's no nested loop.
        loops_df = dataframe("time::loop")
        if outer_name in loops_df.columns:
            num_outer = (
                loops_df.dropna(subset=[outer_name])
                .drop_duplicates(subset=keys + [outer_name])
                .groupby(keys)
                .agg(num_outer=(outer_name, "max"))
                .reset_index()
            )
            num_outer["num_outer"] = pd.to_numeric(num_outer["num_outer"])
        else:
            iters = query(
                "SELECT projid, tstamp, filename, COUNT(DISTINCT ctx) AS num_outer "
                "FROM logs WHERE value_name = 'time::iter' "
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
