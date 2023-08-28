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

def pivot(*args):
    conn, cursor = database.conn_and_cursor()
    # Query the distinct value_names
    try:
        df = database.pivot(cursor, *(args if args else tuple()))
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


def replay(apply_vars: List[str], where_clause: Optional[str]=None):
    versions.git_commit("Hindsight logging stmts added.")

    with open(".flor.json", 'r') as f:
        main_script = json.load(f)[-1]["FILENAME"]
    
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    shutil.copy2(main_script, temp_file.name)

    # First, we convert named_vars to linenos
    schedule = Schedule(apply_vars, where_clause, main_script)

    if not schedule.is_empty():
        print()
        print(schedule)
        print()


        # Pick up on versions
        active_branch = versions.current_branch()
        try:
            for projid, ts, hexsha, fname, epoch in schedule.iter_dims():
                if ts in known_tstamps:
                    print("entering", ts, hexsha)
                    versions.checkout(hexsha)
                    with open('.flor.json', 'r') as f:
                        main_script = json.load(f)[-1]['FILENAME']
                    for v,lineno in zip(apply_vars, apply_linenos):
                        print("applying: ", v, lineno)
                        try:
                            backprop(lineno, temp_file.name, main_script, main_script)
                        except Exception as e:
                            print("Exception raised during `backprop`", e)
                            raise e
                    subprocess.run(['python', main_script, '--replay_flor'] + query_op)
        except Exception as e:
            print("Exception raised during outer replay loop", e)
            raise e
        finally:
            versions.reset_hard()
            versions.checkout(active_branch)
            os.remove(temp_file.name)
            schedule = get_schedule(apply_vars, where_clause)

        print()
        print(schedule)
        print()

    else:
        print("Nothing to do.")


        
class Schedule:
    def __init__(self, apply_vars, where_clause, main_script=None) -> None:
        if main_script is None:
            with open(".flor.json", 'r') as f:
                main_script = json.load(f)[-1]["FILENAME"]
        
        with open(main_script, "r") as f:
            tree = ast.parse(f.read())

        self.lev = LoggedExpVisitor()
        self.lev.visit(tree)
        # TODO: 
        # case when integer supplied through apply_vars, 
        #     you will need to infer var_name from ast
        self.apply_vars = apply_vars
        self.apply_linenos = [int(v) if utils.is_integer(v) else self.lev.names[v] for v in apply_vars]
        self.where_clause = where_clause
        df = pivot()
        if where_clause is None:
            if (sub_vars := [v for v in apply_vars if v not in df.columns]):
                # Function to perform the natural join
                ext_df = pivot(*sub_vars)
                common_columns = set(df.columns) & set(ext_df.columns)
                df = pd.merge(df, ext_df, on=list(common_columns), how="outer")
            self.df = df
        else:
            # Regular expression to match column names
            column_pattern = re.compile(r'\b[A-Za-z_]\w*\b')
            columns = set(re.findall(column_pattern, where_clause))

            # Convert to list if needed
            columns_list = list(columns)
            self.vars_in_where = columns_list
            print("columns in where_clause:", columns_list)
            if columns_list:
                ext_df = pivot(*columns_list)
                common_columns = set(df.columns) & set(ext_df.columns)
                df = pd.merge(df, ext_df, on=list(common_columns), how="outer")
            self.df = utils.cast_dtypes(df).query(where_clause)

    def is_empty(self):
        return self.df.empty
    
    def get_loglvl(self):
        self.dims = [c.split('_')[0] for c in self.df.columns if str(c).endswith('_iteration')]
        loglvl = len(self.dims)
        return loglvl
    
    def iter_dims(self):
        ts2vid = {pd.Timestamp(ts):str(vid) for ts, vid, _ in versions.get_latest_autocommit()}

        for row_dict in self.df.to_dict(orient='records'):
            epoch = int(row_dict[self.dims[0] + '_iteration']) if self.dims else None
            yield row_dict['projid'], row_dict['tstamp'], ts2vid[row_dict['tstamp']], row_dict['filename'], epoch

    def __str__(self):
        if self.where_clause is None:
            return self.df[self.df[self.apply_vars].isna().any(axis=1)].__str__()
        else:
            schedule = self.df.copy() # appease pandas warning
            schedule[[v for v in self.apply_vars if v not in schedule.columns]] = np.nan
            return schedule.__str__()
        
    def __repr__(self):
        if self.where_clause is None:
            return self.df[self.df[self.apply_vars].isna().any(axis=1)].__repr__()
        else:
            schedule = self.df.copy() # appease pandas warning
            schedule[[v for v in self.apply_vars if v not in schedule.columns]] = np.nan
            return schedule.__repr__()
        
    def _repr_html_(self):
        if self.where_clause is None:
            return self.df[self.df[self.apply_vars].isna().any(axis=1)]._repr_html_() # type: ignore
        else:
            schedule = self.df.copy() # appease pandas warning
            schedule[[v for v in self.apply_vars if v not in schedule.columns]] = np.nan
            return schedule._repr_html_() # type: ignore
        