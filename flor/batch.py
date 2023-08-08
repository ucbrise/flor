from typing import Optional
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


def cross_prod(**kwargs) -> Optional[pd.DataFrame]:
    for k in kwargs:
        if not isinstance(kwargs[k], (tuple, list)):
            print(f"Coercing type {type(kwargs[k])}: {kwargs[k]}")
            kwargs[k] = tuple(kwargs[k])
    dataframes = [pd.DataFrame.from_dict({k: v}) for k, v in kwargs.items()]
    if dataframes:
        rolling_df = dataframes[0]
        for df in dataframes[1:]:
            rolling_df = rolling_df.merge(df, how="cross")
        return rolling_df
