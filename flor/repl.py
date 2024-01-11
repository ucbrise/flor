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
    database.create_tables(cursor)

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
    if not schedule.is_empty():
        with open(".flor.json", "r") as f:
            main_script = json.load(f)[-1]["FILENAME"]
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        shutil.copy2(main_script, temp_file.name)
        with open(main_script, "r") as f:
            tree = ast.parse(f.read())

        lev = LoggedExpVisitor()
        lev.visit(tree)

        print()
        print(schedule)
        print()

        res = input("Continue [Y/n]? ")
        if res.lower().strip() == "n":
            return schedule

        # Pick up on versions
        active_branch = versions.current_branch()
        loglvl = schedule.get_loglvl()
        try:
            for projid, ts, hexsha, main_script, epochs in schedule.iter_dims():
                print("entering", str(ts), hexsha)
                versions.checkout(hexsha)
                for v, lineno in zip(
                    apply_vars,
                    [
                        int(v) if utils.is_integer(v) else lev.names[v]
                        for v in apply_vars
                    ],
                ):
                    print("applying: ", v, lineno)
                    try:
                        backprop(lineno, temp_file.name, main_script, main_script)
                    except Exception as e:
                        print("Exception raised during `backprop`", e)
                        raise e
                if loglvl == 0:
                    print("loglvl", loglvl, "no dims")
                    subprocess.run(["python", main_script, "--replay_flor"])
                elif loglvl == 1:
                    tup = ",".join(epochs) + ","
                    print("loglvl", loglvl, tup)
                    subprocess.run(
                        ["python", main_script, "--replay_flor"]
                        + [schedule.dims[0] + "=" + tup]
                    )
                elif loglvl == 2:
                    tup = ",".join(epochs) + ","
                    print("loglvl", loglvl, tup)
                    subprocess.run(
                        ["python", main_script, "--replay_flor"]
                        + [schedule.dims[0] + "=" + tup, schedule.dims[1] + "=1"]
                    )
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

        filtered_vs = [v for v in apply_vars if not utils.is_integer(v)]
        if schedule.vars_in_where is not None:
            filtered_vs += schedule.vars_in_where
        schedule = dataframe(*filtered_vs)
        print()
        print(schedule)
        print()
    else:
        print("Nothing to do.")

    return schedule


class Schedule:
    def __init__(self, apply_vars, where_clause) -> None:
        # TODO:
        # case when integer supplied through apply_vars,
        #     you will need to infer var_name from ast
        self.apply_vars = apply_vars
        self.where_clause = where_clause
        self.vars_in_where = None
        df = dataframe()
        if where_clause is None:
            if sub_vars := [v for v in apply_vars if v not in df.columns]:
                # Function to perform the natural join
                ext_df = dataframe(*sub_vars)
                common_columns = set(df.columns) & set(ext_df.columns)
                df = pd.merge(df, ext_df, on=list(common_columns), how="outer")
            self.df = df
        else:
            # Regular expression to match column names
            column_pattern = re.compile(r"\b[A-Za-z_]\w*\b")
            columns = set(re.findall(column_pattern, where_clause))

            # Convert to list if needed
            columns_list = list(columns)
            self.vars_in_where = columns_list
            print("columns in where_clause:", columns_list)
            if columns_list:
                ext_df = dataframe(*columns_list)
                common_columns = set(df.columns) & set(ext_df.columns)
                df = pd.merge(df, ext_df, on=list(common_columns), how="outer")
            self.df = utils.cast_dtypes(df).query(where_clause)

    def is_empty(self):
        return self.df.empty

    def get_loglvl(self):
        self.dims = [
            c.split("_")[0] for c in self.df.columns if str(c).endswith("_iteration")
        ]
        loglvl = len(self.dims)
        return loglvl

    def iter_dims(self):
        ts2vid = {
            pd.Timestamp(ts): str(vid)
            for ts, vid, _ in versions.get_latest_autocommit()
        }

        epochs: List[str] = []
        prev_row = None

        for row_dict in self.df.to_dict(orient="records"):
            curr_tstamp = row_dict["tstamp"]

            # Compare current timestamp with the previous timestamp
            if prev_row is not None and curr_tstamp != prev_row["tstamp"]:
                yield prev_row["projid"], prev_row["tstamp"], ts2vid[
                    prev_row["tstamp"]
                ], prev_row["filename"], epochs
                epochs.clear()

            # Update epochs
            if self.dims:
                epochs.append(str(row_dict[self.dims[0] + "_iteration"]))

            # Update prev_row for the next iteration
            prev_row = row_dict

        # Yield the final record if prev_row is populated
        if prev_row is not None:
            yield prev_row["projid"], prev_row["tstamp"], ts2vid[
                prev_row["tstamp"]
            ], prev_row["filename"], epochs

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
