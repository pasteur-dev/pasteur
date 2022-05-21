from typing import Collection
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from pasteur.pipelines.synth.nodes import (
    fit_table,
    transform_table,
    reverse_transform_table,
)


def create_transform_pipeline(tables):
    tables = [t.split(".")[-1] for t in tables]

    table_nodes = []
    for t in tables:
        table_nodes += [
            node(
                func=fit_table,
                inputs=["primary.%s" % t, "params:metadata.tables.%s" % t],
                outputs="transformer.%s" % t,
                namespace="transformer",
            ),
            node(
                func=transform_table,
                inputs=["primary.%s" % t, "transformer.%s" % t],
                outputs="encoded.%s" % t,
                namespace="encoded",
            ),
            node(
                func=reverse_transform_table,
                inputs=["encoded.%s" % t, "transformer.%s" % t],
                outputs="decoded.%s" % t,
                namespace="decoded",
            ),
        ]

    return pipeline(table_nodes)


def create_pipeline(dataset: str, tables: Collection[str]) -> Pipeline:
    table_mapping = {"primary.%s" % t.split(".")[-1]: t for t in tables}
    parameters = {
        "params:metadata.tables.%s"
        % t.split(".")[-1]: "params:%s.metadata.tables.%s"
        % (dataset, t.split(".")[-1])
        for t in tables
    }

    return modular_pipeline(
        pipe=create_transform_pipeline(tables),
        namespace="view",
        inputs=table_mapping,
        parameters=parameters,
    )
