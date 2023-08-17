from .constants import *
from .cli import flags
from . import database
from . import versions


import json
import os
from pathlib import Path
import sqlite3


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
                for row in results:
                    print(row)
            except sqlite3.Error as e:
                print(f"An error occurred: {e}")

            # Close connection
            conn.close()

        elif flags.args.flor_command == "apply":
            from .hlast import apply

            apply(flags.args.dp_list, flags.args.train_file)


if __name__ == "__main__":
    main()
