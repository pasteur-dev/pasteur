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
    orig_tables = {t: "orig:primary:%s" % t for t in tables}
    encoded_tables = {t: "orig:encoded:%s" % t for t in tables}
    decoded_tables = {t: "orig:decoded:%s" % t for t in tables}

    table_nodes = []
    for t in tables:
        table_nodes += [
            node(
                func=fit_table,
                inputs=["view:primary:%s" % t, "params:metadata.tables.%s" % t],
                outputs="view:transformer:%s" % t,
            ),
            node(
                func=transform_table,
                inputs=["view:primary:%s" % t, "view:transformer:%s" % t],
                outputs="view:encoded:%s" % t,
            ),
            node(
                func=reverse_transform_table,
                inputs=["view:encoded:%s" % t, "view:transformer:%s" % t],
                outputs="view:decoded:%s" % t,
            ),
        ]

    return pipeline(table_nodes)


def create_pipeline(dataset: str, tables: Collection[str]) -> Pipeline:
    table_mapping = {"view:primary:%s" % t.split(".")[-1]: t for t in tables}
    parameters = {
        "params:metadata.tables.%s"
        % t.split(".")[-1]: "params:%s.metadata.tables.%s"
        % (dataset, t.split(".")[-1])
        for t in tables
    }

    return modular_pipeline(
        pipe=create_transform_pipeline(tables),
        namespace="synth",
        inputs=table_mapping,
        parameters=parameters,
    )
