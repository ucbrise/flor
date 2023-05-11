import pandas as pd
import os
import math

from . import database


def batch(table: pd.DataFrame, from_path=os.getcwd()):
    cli_args = []
    records = table.to_dict(orient="records")
    for record in records:
        s = ""
        for k, v in record.items():
            if v is None or math.isnan(v):
                continue
            s += f"--{k} {v} "
        cli_args.append(s)
        print(s)

    db_conn = database.start_db()
    try:
        database.add_jobs(db_conn, from_path, cli_args)
    except:
        pass
    finally:
        db_conn.close()
