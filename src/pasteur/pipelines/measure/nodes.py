from math import isnan
import pandas as pd


def sdv_compat(table: pd.DataFrame) -> pd.DataFrame:
    # Move primary key from index to column
    table = table.reset_index(drop=not table.index.name)

    # Convert Categorical columns to objects
    categorical = {
        field: "O"
        for field, type in table.dtypes.items()
        if isinstance(type, pd.CategoricalDtype)
    }
    table = table.astype(categorical)

    return table


def measure_sdmetrics_single_table(metadata, real, synthetic):
    import sdmetrics

    metrics = sdmetrics.single_table.SingleTableMetric.get_subclasses()

    return sdmetrics.compute_metrics(
        metrics,
        sdv_compat(real),
        sdv_compat(synthetic),
        metadata=metadata,
    )


def measure_sdmetrics_multi_table(**kwargs):
    import sdmetrics

    metadata = kwargs.pop("metadata")
    real = {}
    synth = {}

    for name, table in kwargs.items():
        table = sdv_compat(table)

        if "real" in name:
            name = name.replace("real.", "")
            real[name] = table
        if "synth" in name:
            name = name.replace("synth.", "")
            synth[name] = table

    metrics = sdmetrics.multi_table.MultiTableMetric.get_subclasses()

    return sdmetrics.compute_metrics(
        metrics,
        real,
        synth,
        metadata=metadata,
    )
