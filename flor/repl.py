import ast
import json
import re
import shutil
import numpy as np
import pandas as pd
from typing import List, Optional
import subprocess
import tempfile
import os

from . import utils
from .hlast.visitors import LoggedExpVisitor, NamedColumnVisitor
from .hlast import backprop

from . import database
from . import versions
from .clock import Clock


def dataframe(*args):
    conn, _ = database.conn_and_cursor()
    # Query the distinct value_names
    try:
        df = database.pivot(conn, *(args if args else tuple()))
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


def replay(apply_vars: List[str], where_clause: Optional[str] = None):
    versions.git_commit("Hindsight logging stmts added.")
    schedule = Schedule(apply_vars, where_clause)

    with open(".flor.json", "r") as f:
        main_script = json.load(f)[0]["filename"]
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    shutil.copy2(main_script, temp_file.name)
    with open(main_script, "r") as f:
        tree = ast.parse(f.read())
    lev = LoggedExpVisitor()
    lev.visit(tree)

    loglvl, mark = schedule.get_loglvl(lev)
    schedule.estimate_cost(loglvl, mark)

    print("log level", loglvl, "to", mark)
    print()
    print(schedule.df)
    print()

    res = input(
        f"Continue replay estimated to finish in {utils.discretize(sum(schedule.df['composite']))} [Y/n]? "
    )
    if res.lower().strip() == "n":
        return schedule

    clock = Clock()
    clock.set_start_time()

    # Pick up on versions
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
            if loglvl == 0:
                print("loglvl", loglvl, "no dims")
                cmd = ["python", main_script, "--replay_flor"]
                print(*cmd)
                subprocess.run(cmd)
            elif loglvl == 1:
                tup = ",".join([i for i in range(schedule.df["num_epochs"])]) + ","  # type: ignore
                print("loglvl", loglvl, tup)
                cmd = ["python", main_script, "--replay_flor"] + ["epoch=" + tup]
                print(*cmd)
                subprocess.run(cmd)
            elif loglvl == 2:
                tup = ",".join([i for i in range(schedule.df["num_epochs"])]) + ","  # type: ignore
                print("loglvl", loglvl, tup)
                cmd = ["python", main_script, "--replay_flor"] + [
                    "epoch=" + tup,
                    "step=1",
                ]
                print(*cmd)
                subprocess.run(cmd)
            else:
                raise NotImplementedError(
                    "Please open a Pull Request on GitHub and describe your use-case."
                )
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

    def estimate_cost(self, loglvl: int, mark: str):
        assert mark in ("prefix", "suffix")
        keys = ["projid", "tstamp", "filename"]
        pvt = dataframe()
        if loglvl == 0:
            if mark == "prefix":
                self.df = dataframe("delta::prefix")
                self.df["composite"] = pd.to_numeric(self.df["delta::prefix"])
            else:
                self.df = dataframe("delta::prefix", "delta::suffix")
                self.df["composite"] = (
                    pd.to_numeric(self.df["delta::prefix"])
                    + pd.to_numeric(self.df["delta::suffix"])
                    + 1
                )
            self.df = pd.merge(pvt, self.df, on=keys, how="inner")
        elif loglvl == 1:
            # i - delta::loop where i is the full duration for that iteration
            df = dataframe("delta::loop")
            df["delta::loop"] = pd.to_numeric(df["delta::loop"], errors="coerce")
            df_grouped = (
                df.groupby(keys)
                .agg(
                    num_epochs=("epoch", "max"), sum_nested_loops=("delta::loop", "sum")
                )
                .reset_index()
            )
            temp_df = query(
                "SELECT * FROM logs WHERE ctx_id is null and value_name='delta::loop';"
            )
            temp_df.drop(columns=["ctx_id", "value_name", "value_type"], inplace=True)  # type: ignore
            temp_df = temp_df.rename(columns={"value": "coarse_loop"})  # type: ignore
            temp_df["coarse_loop"] = pd.to_numeric(
                temp_df["coarse_loop"], errors="coerce"
            )

            merged_df = pd.merge(temp_df, df_grouped, on=keys, how="inner")
            merged_df["marginal"] = (
                merged_df["coarse_loop"] - merged_df["sum_nested_loops"]
            )
            merged_df.drop(columns=["coarse_loop", "sum_nested_loops"], inplace=True)
            df = dataframe("delta::prefix", "delta::suffix")
            merged_df = pd.merge(merged_df, df, on=keys, how="inner")
            merged_df["composite"] = (
                pd.to_numeric(merged_df["num_epochs"])
                + merged_df["marginal"]
                + pd.to_numeric(merged_df["delta::prefix"])
                + pd.to_numeric(merged_df["delta::suffix"])
            )
            self.df = pd.merge(pvt, merged_df, on=keys, how="inner")
        elif loglvl == 2:
            base_df = dataframe("delta::prefix", "delta::suffix")
            loop_df = dataframe("delta::loop")
            loop_df["delta::loop"] = pd.to_numeric(
                loop_df["delta::loop"], errors="coerce"
            )
            df_grouped = (
                loop_df.groupby(keys).agg(num_epochs=("epoch", "max")).reset_index()
            )
            temp_df = query(
                "SELECT * FROM logs WHERE ctx_id is null and value_name='delta::loop';"
            )
            temp_df.drop(columns=["ctx_id", "value_name", "value_type"], inplace=True)  # type: ignore
            temp_df = temp_df.rename(columns={"value": "coarse_loop"})  # type: ignore
            temp_df["coarse_loop"] = pd.to_numeric(
                temp_df["coarse_loop"], errors="coerce"
            )
            merged_df = pd.merge(temp_df, base_df, on=keys, how="inner")
            merged_df["composite"] = (
                pd.to_numeric(merged_df["coarse_loop"])
                + pd.to_numeric(merged_df["delta::prefix"])
                + pd.to_numeric(merged_df["delta::suffix"])
            )
            merged_df = pd.merge(pvt, merged_df, on=keys, how="inner")
            merged_df = pd.merge(merged_df, df_grouped, on=keys, how="inner")
            self.df = merged_df
        else:
            raise

    def is_empty(self):
        return self.df.empty

    def get_loglvl(self, lev: LoggedExpVisitor):
        # Get the largest lineno from self.apply_vars
        max_lineno = max(lev.names[v] for v in self.apply_vars)
        loglevels = [lev.line2level[lev.names[v]] for v in self.apply_vars]

        # Check for monotonic growth
        pairs = sorted(
            [
                (line, level)
                for line, level in lev.line2level.items()
                if line <= max_lineno
            ],
            key=lambda x: x[0],
        )
        is_monotonic = all(
            pairs[i][1] <= pairs[i + 1][1] for i in range(len(pairs) - 1)
        )

        return (
            max(loglevels),
            "prefix" if is_monotonic else "suffix",
        )

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
