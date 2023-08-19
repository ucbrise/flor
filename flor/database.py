from typing import Dict, Optional, Tuple
from .constants import *
from . import orm
import pandas as pd


def unpack(output_buffer, cursor):
    if not output_buffer:
        return
    print("Unpacking ", output_buffer[-1])

    # Structure the output buffer
    stem = output_buffer[-1]
    for obj in output_buffer[:-1]:
        orm.parse_entries(obj)
        orm.parse_log(stem, obj, orm.parse_loop(obj))

    # Read existing loop ids to get ctx_ids, know when to create new ones
    for (
        ctx_id,
        parent_ctx_id,
        loop_name,
        loop_entries,
        loop_iteration,
    ) in read_from_loops(cursor):
        orm.loops[
            (
                int(parent_ctx_id)
                if (isinstance(parent_ctx_id, str) or isinstance(parent_ctx_id, int))
                else -1,
                loop_name,
                int(loop_entries),
                int(loop_iteration),
            )
        ] = int(ctx_id)

    def dfs(loop: Optional[orm.Loop]):
        if loop is None:
            return None
        parent_ctx_id = dfs(loop.parent)
        if (
            k := (
                int(parent_ctx_id)
                if (isinstance(parent_ctx_id, str) or isinstance(parent_ctx_id, int))
                else -1,
                loop.name,
                int(loop.entries),
                int(loop.iteration),
            )
        ) in orm.loops:
            return orm.loops[k]
        else:
            v = len(orm.loops)
            orm.loops[k] = v
            insert_data_into_loops(
                v, parent_ctx_id, loop.name, loop.entries, loop.iteration, cursor
            )
            return v

    for log_record in orm.logs:
        insert_data_into_logs(
            log_record.projid,
            log_record.tstamp,
            log_record.filename,
            dfs(log_record.loop),
            log_record.name,
            log_record.value,
            log_record.type.value,
            cursor,
        )


def create_tables(cursor):
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS loops (
        ctx_id INTEGER,
        parent_ctx_id INTEGER,
        loop_name TEXT,
        loop_entries INTEGER,
        loop_iteration INTEGER
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS logs (
        projid TEXT,
        tstamp TEXT,
        filename TEXT,
        ctx_id INTEGER,
        value_name TEXT,
        value TEXT,
        value_type INTEGER
    )
    """
    )


def insert_data_into_loops(
    ctx_id, parent_ctx_id, loop_name, loop_entries, loop_iteration, cursor
):
    cursor.execute(
        """
    INSERT INTO loops (ctx_id, parent_ctx_id, loop_name, loop_entries, loop_iteration)
    VALUES (?, ?, ?, ?, ?)
    """,
        (ctx_id, parent_ctx_id, loop_name, loop_entries, loop_iteration),
    )


def insert_data_into_logs(
    projid, tstamp, filename, ctx_id, value_name, value, value_type, cursor
):
    cursor.execute(
        """
    INSERT INTO logs (projid, tstamp, filename, ctx_id, value_name, value, value_type)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (projid, tstamp, filename, ctx_id, value_name, value, value_type),
    )


def read_from_loops(cursor):
    cursor.execute("SELECT * FROM loops")
    return cursor.fetchall()


def read_from_logs(cursor, where_clause=None):
    if where_clause is None:
        cursor.execute("SELECT DISTINCT * FROM logs")
    else:
        cursor.execute(f"SELECT DISTINCT * FROM logs WHERE {where_clause}")
    return cursor.fetchall()


def read_known_tstamps(cursor):
    cursor.execute("SELECT DISTINCT tstamp FROM logs")
    return cursor.fetchall()


def query(cursor, user_query):
    cursor.execute(user_query)
    return cursor.fetchall()


def get_column_names(cursor):
    column_names = [description[0] for description in cursor.description]
    return column_names


def pivot(cursor, *args):
    print(args)

    def _pivot_star():
        cursor.execute(
            "SELECT DISTINCT value_name FROM logs WHERE value_type = 1 AND ctx_id IS NULL;"
        )
        value_names = cursor.fetchall()

        # Build the dynamic part of the SQL query
        dynamic_sql = ", ".join(
            [
                f"MAX(CASE WHEN value_name = '{value_name[0]}' THEN value ELSE NULL END) AS '{value_name[0]}'"
                for value_name in value_names
            ]
        )

        # Construct the final SQL query
        final_sql = f"""
        SELECT projid,
            tstamp,
            filename,
            {dynamic_sql}
        FROM logs
        WHERE value_type = 1 AND ctx_id IS NULL
        GROUP BY projid, tstamp, filename;
        """

        # Execute the final SQL query
        cursor.execute(final_sql)
        df = pd.DataFrame(cursor.fetchall(), columns=get_column_names(cursor))
        return df

    if not args:
        return _pivot_star()

    dataframes = []
    loops = pd.DataFrame(read_from_loops(cursor), columns=get_column_names(cursor))
    for value_name in args:
        logs = pd.DataFrame(
            read_from_logs(cursor, where_clause=f'value_name = "{value_name}"'), columns=get_column_names(cursor)
        )
        dataframes.append(logs)
    
    return loops, dataframes
