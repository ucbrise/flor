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
    print("VARS,", apply_vars)
    print("where_clause", where_clause)


__all__ = ["pivot", "query", "replay"]

# def apply(names: List[str], dst: str, stash=None):
#     """
#     Caller checks out a previous version
#     """

#     fp = Path(dst)
#     facts = q.log_records(skip_unpack=True)

#     # Get latest timestamp for each variable name
#     historical_names = facts[facts["name"].isin(names)][
#         ["name", "tstamp", "vid", "value"]
#     ]
#     historical_names = historical_names[historical_names["value"].notna()]
#     hist_name2tstamp = historical_names[["name", "tstamp", "vid"]].drop_duplicates()

#     stash = q.get_stash()
#     assert stash is not None
#     assert State.repo is not None
#     copyfile(fp, stash / fp)

#     for _, row in hist_name2tstamp.iterrows():
#         if len(State.hls_hits) == len(names):
#             break
#         n = row["name"]
#         v = row["vid"]
#         State.repo.git.checkout(v, "--", fp)
#         lev = LoggedExpVisitor()
#         with open(fp, "r") as f:
#             lev.visit(ast.parse(f.read()))
#         if n in lev.names:
#             State.grouped_names[n] = lev.names[n]
#             State.hls_hits.add(n)
#             copyfile(src=fp, dst=stash / Path(n).with_suffix(".py"))

#     copyfile(stash / fp, fp)

#     for p in os.listdir(stash):
#         State.hls_hits.add(".".join(p.split(".")[0:-1]))

#     assert len(os.listdir(stash)) > len(
#         names
#     ), f"Failed to find log statement for vars {[n for n in names if n not in State.hls_hits]}"

#     # Next, from the stash you will apply each file to our main one
#     for name in names:
#         with open(stash / Path(name).with_suffix(".py"), "r") as f:
#             tree = ast.parse(f.read())

#         if name in State.grouped_names:
#             lineno = int(State.grouped_names[name])
#         else:
#             lev = LoggedExpVisitor()
#             lev.visit(tree)
#             lineno = int(lev.names[name])
#         # lev possibly unbound
#         backprop(
#             lineno,
#             str(stash / Path(name).with_suffix(".py")).replace("\x1b[m", ""),
#             dst,
#         )
#         print(f"Applied {name} to {dst}")
