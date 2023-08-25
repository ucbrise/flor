import ast
import json
import numpy as np
import pandas as pd
from typing import List, Optional

from . import utils
from .hlast.visitors import LoggedExpVisitor, NamedColumnVisitor

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

def replay(apply_vars: List[str], pd_expression: Optional[str]=None):
    print("VARS:", apply_vars)
    print("pd_expression:", pd_expression)

    with open(".flor.json", 'r') as f:
        main_script = json.load(f)[-1]["FILENAME"]

    print("main script:", main_script)

    with open(main_script, "r") as f:
        tree = ast.parse(f.read())

    lev = LoggedExpVisitor()
    lev.visit(tree)

    # First, we convert named_vars to linenos
    apply_linenos = {int(v) if utils.is_integer(v) else lev.names[v] for v in apply_vars}

    # Do a forward pass to determine replay log level
    log_lvl = max([lev.line2level[lineno] for lineno in apply_linenos])
    print("log level:", log_lvl)


    df = pivot()
    if pd_expression is None:
        if (sub_vars := [v for v in apply_vars if v not in df.columns]):
            # Function to perform the natural join
            ext_df = pivot(*sub_vars)
            common_columns = set(df.columns) & set(ext_df.columns)
            df = pd.merge(df, ext_df, on=list(common_columns), how="outer")
        schedule = df[df[apply_vars].isna().any(axis=1)]
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

    print()
    print(schedule)
    print()

    # Pick up on versions


    
