from .constants import *
from .cli import flags
from . import database
from . import versions
from . import repl


import json
import platform
import sys
import os
import atexit


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
            user_query = str(flags.args.q)
            df = repl.query(user_query)
            print(df)
        elif flags.args.flor_command == "dataframe":
            df = repl.dataframe(
                *(flags.args.columns if flags.args.columns else tuple())
            )
            print(df)
        elif flags.args.flor_command == "replay":
            if flags.args.where_clause:
                repl.replay(flags.args.VARS, flags.args.where_clause)
            else:
                repl.replay(flags.args.VARS)
        elif flags.args.flor_command == "stat":
            build_context = {
                "architecture": platform.machine(),
                "operating_system": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version(),
                "in_anaconda": "conda" in sys.version
                or "Continuum" in sys.version
                or os.path.exists(os.path.join(sys.prefix, "conda-meta")),
            }
            for k, v in build_context.items():
                print(k + ":", v)


if __name__ == "__main__":
    main()
