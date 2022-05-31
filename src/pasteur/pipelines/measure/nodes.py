from math import isnan
import sdmetrics


def measure_sdmetrics_single_table(metadata, real, synthetic):
    metrics = sdmetrics.single_table.SingleTableMetric.get_subclasses()

    results = sdmetrics.compute_metrics(
        metrics,
        real.reset_index(drop=not real.index.name),
        synthetic.reset_index(drop=not synthetic.index.name),
        metadata=metadata,
    )
    return {
        m["metric"]: {"value": m["raw_score"], "step": 0}
        for _, m in results.iterrows()
        if m["raw_score"] and not isnan(m["raw_score"])
    }
