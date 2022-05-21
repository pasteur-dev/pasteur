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
                inputs=[t, "params:metadata.tables.%s" % t],
                outputs="transformer_%s" % t,
            ),
            node(
                func=transform_table,
                inputs=[t, "transformer_%s" % t],
                outputs="encoded_%s" % t,
            ),
            node(
                func=reverse_transform_table,
                inputs=["encoded_%s" % t, "transformer_%s" % t],
                outputs="decoded_%s" % t,
            ),
        ]

    return pipeline(table_nodes)


def create_pipeline(dataset: str, tables: Collection[str]) -> Pipeline:
    parameters = {
        "params:metadata.tables.%s"
        % t.split(".")[-1]: "params:%s.metadata.tables.%s"
        % (dataset, t.split(".")[-1])
        for t in tables
    }

    return modular_pipeline(
        pipe=create_transform_pipeline(tables),
        namespace=dataset,
        parameters=parameters,
    )
