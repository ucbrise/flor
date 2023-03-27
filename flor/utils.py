import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np

from .shelf import home_shelf


def plot_confusion_matrix(y_true, y_pred, x_classes, y_classes):
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=y_classes, columns=x_classes)
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=False)

    path = home_shelf.get_ref(".png")
    plt.savefig(path)
    return path
