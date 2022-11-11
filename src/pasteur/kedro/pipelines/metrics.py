from kedro.pipeline import node, pipeline

from ...metadata import Metadata
from ...metric import ColumnMetric, ColumnMetricFactory
from ...view import View
from .meta import DatasetMeta as D
from .meta import PipelineMeta
from .utils import gen_closure


def _create_column_metric(table: str, col: str, fs: ColumnMetricFactory, meta: Metadata):
    metric = fs.build()
    metric.fit(meta)


def _create_ingest_column_pipelines(
    view: View,
    metrics: dict[str, list[ColumnMetricFactory]],
    wrk_split: str,
    ref_split: str,
) -> PipelineMeta:
    nodes = []
    for table in view.tables:
        for col, col_metrics in metrics.items():
            for metric in metrics:
                nodes += [node()]

    return PipelineMeta()
