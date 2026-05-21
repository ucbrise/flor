from .constants import *
from .cli import flags
from . import database
from . import versions
from . import orm
from . import repl


import glob
import platform
import sys
import os
import atexit


def main():
    if flags.args is not None and flags.args.flor_command is not None:
        if flags.args.flor_command == "unpack":
            conn, cursor = database.conn_and_cursor()
            database.create_tables(cursor)

            known_tstamps = {t for t, in database.read_known_tstamps(cursor)}
            jsonl_paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*.jsonl")))
            for path in jsonl_paths:
                tstamp = os.path.splitext(os.path.basename(path))[0]
                if tstamp in known_tstamps:
                    continue
                records = orm.read_jsonl(path)
                database.unpack(records, cursor)

            conn.commit()
            conn.close()
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
            repl.replay(
                flags.args.VARS,
                narrow=flags.args.narrow,
                where_clause=flags.args.where_clause,
            )
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
