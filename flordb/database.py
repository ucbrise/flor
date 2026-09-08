import json
from dataclasses import asdict
from functools import reduce
import pandas as pd
import sqlite3

from .constants import *
from . import orm

from . import utils
from typing import Any, Dict, List, Optional, Tuple


def conn_and_cursor():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    return conn, cursor


def _ctx_to_json(ctx) -> Optional[str]:
    if not ctx:
        return None
    if isinstance(ctx, list):
        # list of orm.Segment dataclasses or already-dict segments
        segs = [asdict(s) if hasattr(s, "__dataclass_fields__") else dict(s) for s in ctx]
        return json.dumps(segs)
    raise TypeError(f"ctx must be a list or None, got {type(ctx).__name__}")


def unpack(output_buffer, cursor, source: str = "forward"):
    # `source` tags every row inserted by this call. Forward runs and the
    # `flor unpack` CLI (which rebuilds the cache from JSONL) both insert
    # 'forward'; replay inserts 'replay'. The cache mixes both, read paths
    # default to forward, and `flor unpack` wipes replay state on rebuild.
    if not output_buffer:
        return
    if source not in ("forward", "replay"):
        raise ValueError(f"source must be 'forward' or 'replay', got {source!r}")
    insert_sql = (
        "INSERT INTO logs (projid, tstamp, filename, ctx, value_name, value, "
        "value_type, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    for each in output_buffer:
        if isinstance(each, orm.Log):
            ctx_json = _ctx_to_json(each.ctx)
            cursor.execute(
                insert_sql,
                (
                    each.projid,
                    each.tstamp,
                    each.filename,
                    ctx_json,
                    each.name,
                    str(each.value),
                    each.type,
                    source,
                ),
            )
        else:
            ctx_json = _ctx_to_json(each.get("ctx"))
            cursor.execute(
                insert_sql,
                (
                    each["projid"],
                    each["tstamp"],
                    each["filename"],
                    ctx_json,
                    each["name"],
                    str(each["value"]),
                    each["type"],
                    source,
                ),
            )


def create_tables(cursor):
    # Migrate away from the old (ctx_id-based) schema if present.
    cursor.execute("DROP TABLE IF EXISTS loops")
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='logs'"
    )
    existing = cursor.fetchone() is not None
    if existing:
        cursor.execute("PRAGMA table_info(logs)")
        cols = {row[1] for row in cursor.fetchall()}
        if "ctx" not in cols:
            cursor.execute("DROP TABLE logs")
            existing = False
        elif "source" not in cols:
            # Pre-tag rows existed only because JSONL was unpacked, which is
            # always forward truth -- default backfill matches that history.
            cursor.execute(
                "ALTER TABLE logs ADD COLUMN source TEXT NOT NULL DEFAULT 'forward'"
            )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            projid TEXT,
            tstamp TEXT,
            filename TEXT,
            ctx TEXT,
            value_name TEXT,
            value TEXT,
            value_type INTEGER,
            source TEXT NOT NULL DEFAULT 'forward'
        )
        """
    )


def deduplicate_table(cursor, table_name):
    cursor.execute(
        f"""CREATE TEMPORARY TABLE temp_table AS SELECT DISTINCT * FROM {table_name}"""
    )
    cursor.execute(f"""DELETE FROM {table_name}""")
    cursor.execute(f"""INSERT INTO {table_name} SELECT * FROM temp_table""")
    cursor.execute("DROP TABLE temp_table")


def read_from_logs(cursor, where_clause=None):
    if where_clause is None:
        cursor.execute("SELECT DISTINCT * FROM logs")
    else:
        cursor.execute(f"SELECT DISTINCT * FROM logs WHERE {where_clause}")
    return cursor.fetchall()


def read_known_tstamps(cursor):
    cursor.execute("SELECT DISTINCT tstamp FROM logs")
    return cursor.fetchall()


def query(cursor, user_query, aspandas=False):
    cursor.execute(user_query)
    res = cursor.fetchall()
    if res and aspandas:
        return utils.cast_dtypes(pd.DataFrame(res, columns=get_column_names(cursor)))
    elif res:
        return res


def get_column_names(cursor):
    column_names = [description[0] for description in cursor.description]
    return column_names


def _parse_ctx_cell(s):
    if s is None:
        return []
    if isinstance(s, float) and pd.isna(s):
        return []
    return json.loads(s)


def _value_matches_iteration(iter_col, val_col):
    """Whether `<name>_value` says nothing `<name>` doesn't.

    Looping over `range(n)` makes the iterated value equal to the iteration
    index on every row, so the pair is a duplicate; looping over anything
    else (a list of paths, a dataloader) keeps both. Values arrive as
    strings (`str(value)` at capture time), so compare on the string form.
    """
    def as_text(x):
        # `map` over a nullable Int64 column hands out numpy floats once any
        # row is NA, so 1 would stringify to "1.0" against a value of "1".
        if pd.isna(x):
            return None
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        return str(x)

    left = iter_col.map(as_text)
    right = val_col.map(as_text)
    return bool(((left == right) | (left.isna() & right.isna())).all())


def expand_ctx(logs):
    """Turn the JSON `ctx` column into one column per loop, in place.

    Each `flor.loop` / `flor.iteration` segment contributes `<name>` (the
    iteration index) and, when the iterated value was jsonable and differs
    from that index, `<name>_value`. Walks root to leaf, so outer loops sit
    to the left of the inner loops they enclose.
    """
    parsed = logs["ctx"].apply(_parse_ctx_cell)
    logs = logs.drop(columns=["ctx"])

    max_depth = int(parsed.map(len).max()) if len(parsed) else 0

    for depth in range(max_depth):
        seg = parsed.apply(lambda lst, d=depth: lst[d] if d < len(lst) else None)
        non_null = seg.dropna()
        if non_null.empty:
            continue
        loop_name = non_null.iloc[0]["name"]

        iter_col = seg.apply(
            lambda s: s.get("iteration") if isinstance(s, dict) else None
        )
        val_col = seg.apply(lambda s: s.get("value") if isinstance(s, dict) else None)

        # Surface the iteration column whenever any row carries it. Rows
        # without that ctx depth get NaN. Use Int64 (nullable) so the
        # column survives groupby/max without collapsing to float.
        if iter_col.notna().any():
            # Int64 before the comparison below, so a column pandas read as
            # float doesn't stringify to "1.0" against a value of "1".
            iter_col = iter_col.astype("Int64")
            logs[loop_name] = iter_col

        # Only surface _value column if at least one row carries it, and
        # only when it carries something the iteration index doesn't.
        if val_col.notna().any() and not _value_matches_iteration(iter_col, val_col):
            logs[f"{loop_name}_value"] = val_col

    return logs


def read_io(conn, channel=None):
    """Captured print / logging rows, with loop context expanded to columns."""
    sql = f"SELECT * FROM logs WHERE value_type = {VALUE_TYPE_IO}"
    params: Tuple[Any, ...] = ()
    if channel is not None:
        # LIKE so `io::log` selects every level at once.
        sql += " AND (value_name = ? OR value_name LIKE ?)"
        params = (channel, channel.rstrip(":") + "::%")
    sql += " ORDER BY rowid"
    logs = pd.read_sql(sql, conn, params=params, parse_dates=["tstamp"])
    if logs.empty:
        return logs
    logs = logs[["projid", "tstamp", "filename", "ctx", "source", "value_name", "value"]]
    logs = logs.rename(columns={"value_name": "channel", "value": "line"})
    logs = expand_ctx(logs)
    trailing = ["channel", "line"]
    return logs[[c for c in logs.columns if c not in trailing] + trailing]


def pivot(conn, *args):
    # Pivot surfaces both forward and replay rows. Replay values are not
    # noise -- the user opted in by running `flor replay --apply ...` to see
    # them. The `source` column rides along on every row so they're
    # distinguishable (and joins across variables stay within a source --
    # forward joins to forward, replay to replay -- because `source` ends up
    # in the common-columns set for the per-variable merge).
    def _pivot_star():
        df = pd.read_sql(
            "SELECT DISTINCT value_name FROM logs "
            f"WHERE value_type = {VALUE_TYPE_LOG} AND ctx IS NULL",
            conn,
        )
        value_names = df["value_name"].values
        if len(value_names) == 0:
            print("No default values to pivot on")
            return pd.DataFrame()

        dynamic_sql = ", ".join(
            [
                f"MAX(CASE WHEN value_name = '{value_name}' THEN value ELSE NULL END) AS '{value_name}'"
                for value_name in value_names
            ]
        )

        final_sql = f"""
        SELECT projid,
            tstamp,
            filename,
            source,
            {dynamic_sql}
        FROM logs
        WHERE value_type = {VALUE_TYPE_LOG} AND ctx IS NULL
        GROUP BY projid, tstamp, filename, source;
        """

        return pd.read_sql(
            final_sql,
            conn,
            parse_dates=[
                "tstamp",
            ],
            coerce_float=True,
        )

    if not args:
        return _pivot_star()

    dataframes = []
    for value_name in args:
        logs = pd.read_sql(
            f'SELECT * FROM logs WHERE value_name = "{value_name}"',
            conn,
            parse_dates=["tstamp"],
            coerce_float=True,
        )
        logs = logs[["projid", "tstamp", "filename", "ctx", "source", "value"]]
        logs = logs.rename(columns={"value": value_name})
        logs = expand_ctx(logs)

        dataframes.append(logs)

    def join_on_common_columns(df1, df2):
        common_columns = set(df1.columns) & set(df2.columns)
        return pd.merge(df1, df2, on=list(common_columns), how="outer")

    all_joined = reduce(join_on_common_columns, dataframes)
    cols = [c for c in all_joined.columns if c not in args]
    cols += list(args)
    return all_joined[cols]
