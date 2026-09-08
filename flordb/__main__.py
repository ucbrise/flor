from .constants import *
from . import capture
from . import cli
from .cli import flags
from . import database
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

            # Replay rows are scratch keyed off the historical tstamp; an
            # `unpack` rebuilds from JSONL (forward truth), so wipe replay
            # scratch first to make the rebuild idempotent.
            cursor.execute("DELETE FROM logs WHERE source = 'replay'")

            known_tstamps = {t for t, in database.read_known_tstamps(cursor)}
            jsonl_paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*.jsonl")))
            for path in jsonl_paths:
                tstamp = os.path.splitext(os.path.basename(path))[0]
                if tstamp in known_tstamps:
                    continue
                records = orm.read_jsonl(path)
                database.unpack(records, cursor, source="forward")

            # Extracted metrics are not restored here. They are a local view
            # over the io in runs/, so a rebuild leaves whatever derivation
            # this cache already had and adds none for the runs it just read;
            # `flor capture --extract` is what refreshes them.

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
            apply_vars, where_clause = cli.resolve_replay_args(flags.args)
            repl.replay(
                apply_vars,
                narrow_iters=flags.args.narrow_iters or None,
                where_clause=where_clause,
                overrides=flags.args.replay_overrides or None,
            )
        elif flags.args.flor_command == "capture":
            conn, cursor = database.conn_and_cursor()
            try:
                cursor.execute(
                    f"SELECT COUNT(*) FROM logs WHERE value_type = {VALUE_TYPE_IO}"
                )
                (io_count,) = cursor.fetchone()
                if not io_count:
                    print(
                        "No captured io yet. Run a script that prints or logs, "
                        "or check that capture is on (FLOR_CAPTURE / "
                        "flor.set_capture)."
                    )
                elif flags.args.preview or flags.args.extract:
                    # Preview and extract read the same derivation, so what the
                    # dry run shows is exactly what the write puts in the table.
                    if flags.args.extract:
                        derived = database.extract_metrics(cursor)
                        conn.commit()
                    else:
                        derived = database.derive_extractions(cursor)
                    cli.report_extractions(
                        derived,
                        io_count,
                        limit=flags.args.limit,
                        wrote=flags.args.extract,
                    )
                else:
                    print(repl.io().head(flags.args.limit))
            finally:
                conn.close()
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
