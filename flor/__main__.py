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
            conn, cursor = database.conn_and_cursor()
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
            conn, cursor = database.conn_and_cursor()
            database.create_tables(cursor)

            user_query = flags.args.q
            try:
                results = database.query(cursor, user_query)
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
            conn, cursor = database.conn_and_cursor()
            # Query the distinct value_names
            try:
                df = database.pivot(cursor, *[c.pop() for c in flags.args.columns])
                print(df.tail(20))
            finally:
                conn.close()

        elif flags.args.flor_command == "apply":
            from .hlast import apply

            apply(flags.args.dp_list, flags.args.train_file)


if __name__ == "__main__":
    main()
