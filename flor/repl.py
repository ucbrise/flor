import ast
import json
from typing import List, Optional

from . import utils
from .hlast.visitors import LoggedExpVisitor

from . import database

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
        conn.close()

def replay(apply_vars: List[str], where_clause: Optional[str]):
    print("VARS:", apply_vars)
    print("where_clause:", where_clause)

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


    
