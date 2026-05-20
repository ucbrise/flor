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


def unpack(output_buffer, cursor):
    if not output_buffer:
        return
    for each in output_buffer:
        if isinstance(each, orm.Log):
            ctx_json = _ctx_to_json(each.ctx)
            cursor.execute(
                """INSERT INTO logs (projid, tstamp, filename, ctx, value_name, value, value_type) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    each.projid,
                    each.tstamp,
                    each.filename,
                    ctx_json,
                    each.name,
                    str(each.value),
                    each.type,
                ),
            )
        else:
            ctx_json = _ctx_to_json(each.get("ctx"))
            cursor.execute(
                """INSERT INTO logs (projid, tstamp, filename, ctx, value_name, value, value_type) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    each["projid"],
                    each["tstamp"],
                    each["filename"],
                    ctx_json,
                    each["name"],
                    str(each["value"]),
                    each["type"],
                ),
            )


def create_tables(cursor):
    # Migrate away from the old (ctx_id-based) schema if present.
    cursor.execute("DROP TABLE IF EXISTS loops")
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='logs'"
    )
    if cursor.fetchone() is not None:
        cursor.execute("PRAGMA table_info(logs)")
        cols = {row[1] for row in cursor.fetchall()}
        if "ctx" not in cols:
            cursor.execute("DROP TABLE logs")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            projid TEXT,
            tstamp TEXT,
            filename TEXT,
            ctx TEXT,
            value_name TEXT,
            value TEXT,
            value_type INTEGER
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


def pivot(conn, *args):
    def _pivot_star():
        df = pd.read_sql(
            "SELECT DISTINCT value_name FROM logs WHERE value_type = 1 AND ctx IS NULL",
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
            {dynamic_sql}
        FROM logs
        WHERE value_type = 1 AND ctx IS NULL
        GROUP BY projid, tstamp, filename;
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
        logs = logs[["projid", "tstamp", "filename", "ctx", "value"]]
        logs = logs.rename(columns={"value": value_name})

        parsed = logs["ctx"].apply(_parse_ctx_cell)
        logs = logs.drop(columns=["ctx"])

        max_depth = int(parsed.map(len).max()) if len(parsed) else 0

        # Walk leaf to root, matching the previous JOIN-from-leaf-upward order.
        for depth in range(max_depth - 1, -1, -1):
            seg = parsed.apply(lambda lst, d=depth: lst[d] if d < len(lst) else None)
            non_null = seg.dropna()
            if non_null.empty:
                continue
            loop_name = non_null.iloc[0]["name"]

            iter_col = seg.apply(
                lambda s: s.get("iteration") if isinstance(s, dict) else None
            )
            val_col = seg.apply(
                lambda s: s.get("value") if isinstance(s, dict) else None
            )

            # Match prior behavior: only surface iteration column if every row has one.
            if iter_col.notna().all() and not iter_col.empty:
                logs[loop_name] = iter_col.astype(int)

            # Only surface _value column if at least one row carries it.
            if val_col.notna().any():
                logs[f"{loop_name}_value"] = val_col

        dataframes.append(logs)

    def join_on_common_columns(df1, df2):
        common_columns = set(df1.columns) & set(df2.columns)
        return pd.merge(df1, df2, on=list(common_columns), how="outer")

    all_joined = reduce(join_on_common_columns, dataframes)
    cols = [c for c in all_joined.columns if c not in args]
    cols += list(args)
    return all_joined[cols]
