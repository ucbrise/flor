def data_prep_pivot(df, data_prep_names):
    rolling_dataframe = None
    for dpname in data_prep_names:
        if dpname == "tstamp":
            continue  # Avoiding redundant columns
        pivot_value = df[df["name"] == dpname][
            ["projid", "tstamp", "vid", "epoch", "step", "value"]
        ]
        pivot_f = pivot_value.rename(columns={"value": dpname})
        pivot_f["epoch"] = -1
        pivot_f["step"] = -1
        if rolling_dataframe is None:
            rolling_dataframe = pivot_f
        else:
            rolling_dataframe = rolling_dataframe.merge(
                pivot_f, how="outer", on=["projid", "tstamp", "vid", "epoch", "step"]
            )
    if rolling_dataframe is not None:
        rolling_dataframe = rolling_dataframe.drop_duplicates(
            subset=["projid", "tstamp", "vid"]
        )
        return rolling_dataframe


def outer_loop_pivot(df, outer_loop_names):
    rolling_dataframe = None
    for ol_name in outer_loop_names:
        if ol_name == "tstamp":
            continue  # Avoiding redundant columns
        pivot_value = df[df["name"] == ol_name][
            ["projid", "tstamp", "vid", "epoch", "step", "value"]
        ]
        pivot_f = pivot_value.rename(columns={"value": ol_name})
        pivot_f["step"] = -1
        if rolling_dataframe is None:
            rolling_dataframe = pivot_f
        else:
            rolling_dataframe = rolling_dataframe.merge(
                pivot_f, how="outer", on=["projid", "tstamp", "vid", "epoch", "step"]
            )
    if rolling_dataframe is not None:
        rolling_dataframe = rolling_dataframe.drop_duplicates(
            subset=["projid", "tstamp", "vid", "epoch"]
        )
        return rolling_dataframe


def inner_loop_pivot(df, inner_loop_names):
    rolling_dataframe = None
    for il_name in inner_loop_names:
        if il_name == "tstamp":
            continue  # Avoiding redundant columns
        pivot_value = df[df["name"] == il_name][
            ["projid", "tstamp", "vid", "epoch", "step", "value"]
        ]
        pivot_value["step"] = pivot_value["step"].map(
            lambda x: max(int(x), 0), na_action="ignore"
        )
        pivot_f = pivot_value.rename(columns={"value": il_name})
        if rolling_dataframe is None:
            rolling_dataframe = pivot_f
        else:
            rolling_dataframe = rolling_dataframe.merge(
                pivot_f, how="outer", on=["projid", "tstamp", "vid", "epoch", "step"]
            )
    if rolling_dataframe is not None:
        rolling_dataframe = rolling_dataframe.drop_duplicates(
            subset=["projid", "tstamp", "vid", "epoch", "step"]
        )
        return rolling_dataframe
