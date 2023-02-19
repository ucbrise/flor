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
                pivot_f, how="outer", on=["projid", "tstamp"]
            )
    print(rolling_dataframe)


def outer_loop_pivot(df, outer_loop_names):
    ...


def inner_loop_pivot(df, inner_loop_names):
    ...
