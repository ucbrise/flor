from .constants import *
from .cli import flags
from . import database
from . import versions
from . import utils


import json
import os
from pathlib import Path
import sqlite3
import pandas as pd


def main():
    if flags.args is not None and flags.args.flor_command is not None:
        if flags.args.flor_command == "unpack":
            conn = sqlite3.connect(
                os.path.join(HOMEDIR, Path(PROJID).with_suffix(".db"))
            )
            cursor = conn.cursor()
            database.create_tables(cursor)

            start_branch = versions.current_branch()
            assert start_branch is not None
            known_tstamps = [t for t, in database.read_known_tstamps(cursor)]
            try:
                for triplet in versions.get_latest_autocommit():
                    ts_start, next_commit, _ = triplet
                    if ts_start in known_tstamps:
                        break
                    versions.checkout(next_commit)
                    with open(".flor.json", "r") as f:
                        database.unpack(json.load(f), cursor)
                conn.commit()
                conn.close()
            finally:
                versions.checkout(start_branch.name)
        elif flags.args.flor_command == "query":
            conn = sqlite3.connect(
                os.path.join(HOMEDIR, Path(PROJID).with_suffix(".db"))
            )
            cursor = conn.cursor()
            database.create_tables(cursor)

            user_query = flags.args.q
            try:
                cursor.execute(user_query)
                results = cursor.fetchall()
                parta, partb = utils.split_and_retrieve_elements(results)
                if len(parta) + len(partb) == len(results):
                    for row in parta + partb:
                        print(row)
                else:
                    for row in (
                        parta
                        + [
                            "...",
                        ]
                        + partb
                    ):
                        print(row)
            except sqlite3.Error as e:
                print(f"An error occurred: {e}")

            # Close connection
            conn.close()

        elif flags.args.flor_command == "pivot":
            conn = sqlite3.connect(
                os.path.join(HOMEDIR, Path(PROJID).with_suffix(".db"))
            )
            cursor = conn.cursor()

            # Query the distinct value_names
            try:
                if not flags.columns:
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
                    result = cursor.fetchall()
                    column_names = [
                        description[0] for description in cursor.description
                    ]
                    df = pd.DataFrame(result, columns=column_names)
                    print(df.head())
                else:
                    value_names = flags.columns
                    dataframes = []
                    for value_name in value_names:
                        column_dims = ["projid", "tstamp", "filename", "ctx_id"]
                        cursor.execute(
                            f"SELECT DISTINCT {','.join(column_dims)}, value as '{value_name[0]}' FROM logs WHERE value_name = ?",
                            value_name,
                        )
                        results = cursor.fetchall()
                        df = pd.DataFrame(
                            results,
                            columns=(
                                column_dims
                                + [
                                    value_name[0],
                                ]
                            ),
                        )
                        while any(df["ctx_id"].notnull()):
                            # TODO PIVOT INCOMPLETE
                            ctx_ids = tuple(df["ctx_id"].dropna().values)
                            ctx_ids_str = ", ".join(map(str, ctx_ids))
                            query = f"SELECT DISTINCT loop_name FROM loops WHERE ctx_id IN ({ctx_ids_str})"
                            cursor.execute(query)
                            dim_name = cursor.fetchall().pop()[0]

                            cursor.execute(
                                f"""
                                SELECT * from 
                                """,
                                value_name,
                            )
                            results = cursor.fetchall()
                            print(results)
                            column_names = [
                                description[0] for description in cursor.description
                            ]
                            print(column_names)
                            df = pd.DataFrame(results, columns=column_names)
                            # df = df.drop(columns=["ctx_id", "loop_name"])
                            df = df.rename(
                                {
                                    "parent_ctx_id": "ctx_id",
                                    "loop_entries": dim_name + "_entries",
                                    "loop_iteration": dim_name + "_iteration",
                                }
                            )

                        # df = df.drop(
                        #     columns=[
                        #         "ctx_id",
                        #     ]
                        # )
                        dataframes.append(df)

                    print(dataframes)

            finally:
                conn.close()

        elif flags.args.flor_command == "apply":
            from .hlast import apply

            apply(flags.args.dp_list, flags.args.train_file)


if __name__ == "__main__":
    main()
