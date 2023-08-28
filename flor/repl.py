import ast
import json
import re
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

    with open(main_script, "r") as f:
        anchor_script_buffer = f.read()
        tree = ast.parse(anchor_script_buffer)

    
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(temp_file.name, "w") as f:
        f.write(anchor_script_buffer)

    lev = LoggedExpVisitor()
    lev.visit(tree)

    # First, we convert named_vars to linenos
    # TODO: 
    # case when integer supplied through apply_vars, 
    #     you will need to infer var_name from ast
    apply_linenos = [int(v) if utils.is_integer(v) else lev.names[v] for v in apply_vars]

    # TODO: does schedule bring in dims?
    schedule = get_schedule(apply_vars, where_clause)
    if where_clause is None:
        schedule = schedule[schedule[apply_vars].isna().any(axis=1)]

    # Do a forward pass to determine replay log level
    log_lvl = max([lev.line2level[lineno] for lineno in apply_linenos])
    print("log level:", log_lvl)
    # TODO: case when epoch=1,3,5
    if log_lvl == 0:
        query_op = []
    elif log_lvl == 1:
        query_op = ['epoch=1', 'step=0']
    elif log_lvl == 2:
        query_op = ['epoch=1', 'step=1']
    else:
        raise NotImplementedError("Please open a pull request")


    if not schedule.empty:
        print()
        print(schedule)
        print()
        # Pick up on versions
        active_branch = versions.current_branch()
        try:
            # TODO: iterate the schedule not the git log
            # TODO: use schedule to index the epoch
            known_tstamps = schedule['tstamp'].drop_duplicates().values
            for ts, hexsha, end_ts in versions.get_latest_autocommit():
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

        


    
def get_schedule(apply_vars, where_clause):
    df = pivot()
    if where_clause is None:
        if (sub_vars := [v for v in apply_vars if v not in df.columns]):
            # Function to perform the natural join
            ext_df = pivot(*sub_vars)
            common_columns = set(df.columns) & set(ext_df.columns)
            df = pd.merge(df, ext_df, on=list(common_columns), how="outer")
        return df
    else:
        # Regular expression to match column names
        column_pattern = re.compile(r'\b[A-Za-z_]\w*\b')
        columns = set(re.findall(column_pattern, where_clause))

        # Convert to list if needed
        columns_list = list(columns)
        print("columns in where_clause:", columns_list)
        if columns_list:
            ext_df = pivot(*columns_list)
            common_columns = set(df.columns) & set(ext_df.columns)
            df = pd.merge(df, ext_df, on=list(common_columns), how="outer")
        schedule = df.query(where_clause)
        for var in [v for v in apply_vars if v not in schedule.columns]:
            schedule.loc[:,var] = np.nan
        return schedule