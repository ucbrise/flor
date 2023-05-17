from typing import Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np

from .shelf import home_shelf


def plot_confusion_matrix(y_true, y_pred, x_classes, y_classes):
    cf_matrix = confusion_matrix(y_true, y_pred)
    for i in range(len(cf_matrix)):
        cf_matrix[i][i] = 0
    df_cm = pd.DataFrame(cf_matrix, index=y_classes, columns=x_classes)
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=False)

    path = home_shelf.get_ref(".png")
    plt.savefig(path)
    return path


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
