import ast
import json
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
        assert isinstance(df, pd.DataFrame)
        return df
    finally:
        # Close connection
        conn.commit()
        conn.close()

def get_schedule(apply_vars, pd_expression):
    df = pivot()
    if pd_expression is None:
        if (sub_vars := [v for v in apply_vars if v not in df.columns]):
            # Function to perform the natural join
            ext_df = pivot(*sub_vars)
            common_columns = set(df.columns) & set(ext_df.columns)
            df = pd.merge(df, ext_df, on=list(common_columns), how="outer")
        return df
    else:
        # Regular expression to match column names
        ncv = NamedColumnVisitor()
        ncv.visit(ast.parse(pd_expression))
        # Convert to list if needed
        columns_list = list(ncv.names)
        print("columns in pd_expression:", columns_list)
        if columns_list:
            ext_df = pivot(*columns_list)
            common_columns = set(df.columns) & set(ext_df.columns)
            df = pd.merge(df, ext_df, on=list(common_columns), how="outer")
        schedule = eval(pd_expression)
        schedule[[v for v in apply_vars if v not in schedule.columns]] = np.nan
        return schedule


def replay(apply_vars: List[str], pd_expression: Optional[str]=None):
    print("VARS:", apply_vars)
    print("pd_expression:", pd_expression)

    versions.git_commit("Hindsight logging stmts added.")

    with open(".flor.json", 'r') as f:
        main_script = json.load(f)[-1]["FILENAME"]

    print("main script:", main_script)

    with open(main_script, "r") as f:
        anchor_script_buffer = f.read()
        tree = ast.parse(anchor_script_buffer)

    
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(temp_file.name, "w") as f:
        f.write(anchor_script_buffer)

    lev = LoggedExpVisitor()
    lev.visit(tree)

    # First, we convert named_vars to linenos
    apply_linenos = [int(v) if utils.is_integer(v) else lev.names[v] for v in apply_vars]

    # Do a forward pass to determine replay log level
    log_lvl = max([lev.line2level[lineno] for lineno in apply_linenos])
    print("log level:", log_lvl)
    if log_lvl == 0:
        query_op = []
    elif log_lvl == 1:
        query_op = ['epoch=1', 'step=0']
    elif log_lvl == 2:
        query_op = ['epoch=1', 'step=1']
    else:
        raise NotImplementedError("Please open a pull request")


    schedule = get_schedule(apply_vars, pd_expression)
    if pd_expression is None:
        schedule = schedule[schedule[apply_vars].isna().any(axis=1)]

    print()
    print(schedule)
    print()

    # Pick up on versions
    active_branch = versions.current_branch()
    try:
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
                        print("EXCEPTION", e)
                subprocess.run(['python', main_script, '--replay_flor'] + query_op)
    except Exception as e:
        print("EXCEPTION", e)
    finally:
        versions.reset_hard()
        versions.checkout(active_branch)
        os.remove(temp_file.name)
        
    schedule = get_schedule(apply_vars, pd_expression)

    print()
    print(schedule)
    print()

        


    
