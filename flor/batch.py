import pandas as pd
import os
import math

from . import database


def batch(args_table: pd.DataFrame, script="train.py", from_path=os.getcwd()):
    cli_args = []
    records = args_table.to_dict(orient="records")
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
        database.add_jobs(db_conn, from_path, script, cli_args)
    except Exception as e:
        print(e)
    finally:
        db_conn.close()


def cross_prod(**kwargs):
    """
    {'epochs': 2, 'batch_size': 2, 'lr': (1e-5, 1e-4, 1e-3)}
      => [
          {'epochs': 2, 'batch_size': 2, 'lr': 1e-5},
          {'epochs': 2, 'batch_size': 2, 'lr': 1e-4},
          {'epochs': 2, 'batch_size': 2, 'lr': 1e-3}
        ]
    """
    pass
