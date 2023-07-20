from flor.constants import *


def data_prep_pivot(df, data_prep_names):
    rolling_dataframe = None
    for dpname in data_prep_names:
        if dpname == "tstamp":
            continue  # Avoiding redundant columns
        pivot_value = df[df["name"] == dpname][
            list(DATA_PREP)
            + [
                "value",
            ]
        ]
        pivot_f = pivot_value.rename(columns={"value": dpname})
        if rolling_dataframe is None:
            rolling_dataframe = pivot_f
        else:
            rolling_dataframe = rolling_dataframe.merge(
                pivot_f, how="outer", on=list(DATA_PREP)
            )
    if rolling_dataframe is not None:
        rolling_dataframe = rolling_dataframe.drop_duplicates(subset=list(DATA_PREP))
        return rolling_dataframe


def outer_loop_pivot(df, outer_loop_names):
    rolling_dataframe = None
    for ol_name in outer_loop_names:
        if ol_name == "tstamp":
            continue  # Avoiding redundant columns
        pivot_value = df[df["name"] == ol_name][
            list(OUTR_LOOP)
            + [
                "value",
            ]
        ]
        pivot_f = pivot_value.rename(columns={"value": ol_name})
        if rolling_dataframe is None:
            rolling_dataframe = pivot_f
        else:
            rolling_dataframe = rolling_dataframe.merge(
                pivot_f, how="outer", on=list(OUTR_LOOP)
            )
    if rolling_dataframe is not None:
        rolling_dataframe = rolling_dataframe.drop_duplicates(subset=list(OUTR_LOOP))
        return rolling_dataframe


def inner_loop_pivot(df, inner_loop_names):
    rolling_dataframe = None
    for il_name in inner_loop_names:
        if il_name == "tstamp":
            continue  # Avoiding redundant columns
        pivot_value = df[df["name"] == il_name][
            list(INNR_LOOP)
            + [
                "value",
            ]
        ]
        pivot_value["step"] = pivot_value["step"].map(
            lambda x: max(int(x), 0), na_action="ignore"
        )
        pivot_f = pivot_value.rename(columns={"value": il_name})
        if rolling_dataframe is None:
            rolling_dataframe = pivot_f
        else:
            rolling_dataframe = rolling_dataframe.merge(
                pivot_f, how="outer", on=list(INNR_LOOP)
            )
    if rolling_dataframe is not None:
        rolling_dataframe = rolling_dataframe.drop_duplicates(subset=list(INNR_LOOP))
        return rolling_dataframe
