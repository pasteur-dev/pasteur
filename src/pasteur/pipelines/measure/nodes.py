from math import isnan
import sdmetrics


def measure_sdmetrics_single_table(metadata, real, synthetic):
    metrics = sdmetrics.single_table.SingleTableMetric.get_subclasses()

    return sdmetrics.compute_metrics(
        metrics,
        real.reset_index(drop=not real.index.name),
        synthetic.reset_index(drop=not synthetic.index.name),
        metadata=metadata,
    )


def measure_sdmetrics_multi_table(**kwargs):
    metadata = kwargs.pop("metadata")
    real = {}
    synth = {}

    for name, table in kwargs.items():
        table = table.reset_index(drop=not table.index.name)

        if "real" in name:
            real[name.replace("real.", "")] = table
        if "synth" in name:
            synth[name.replace("synth.", "")] = table

    metrics = sdmetrics.multi_table.MultiTableMetric.get_subclasses()

    return sdmetrics.compute_metrics(
        metrics,
        real,
        synth,
        metadata=metadata,
    )
