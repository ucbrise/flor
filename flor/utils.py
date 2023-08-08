from typing import Optional
import pandas as pd


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
