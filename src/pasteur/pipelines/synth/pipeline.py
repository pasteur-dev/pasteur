from importlib.metadata import metadata
from typing import Collection
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from pasteur.pipelines.synth.nodes import reverse_transform_tables, transform_tables


def create_transform_pipeline(tables):
    tables = [t.split(".")[-1] for t in tables]
    orig_tables = {t: "orig:primary:%s" % t for t in tables}
    encoded_tables = {t: "orig:encoded:%s" % t for t in tables}
    decoded_tables = {t: "orig:decoded:%s" % t for t in tables}

    return pipeline(
        [
            node(
                func=transform_tables,
                inputs={"metadata": "params:metadata", **orig_tables},
                outputs={"transformers": "transformers", **encoded_tables},
            ),
            node(
                func=reverse_transform_tables,
                inputs={"transformers": "transformers", **encoded_tables},
                outputs=decoded_tables,
            ),
        ]
    )


def create_pipeline(dataset: str, tables: Collection[str]) -> Pipeline:
    table_mapping = {"orig:primary:%s" % t.split(".")[-1]: t for t in tables}
    parameters = {"params:metadata": "params:%s.metadata" % dataset}

    return modular_pipeline(
        pipe=create_transform_pipeline(tables),
        namespace="synth",
        inputs=table_mapping,
        parameters=parameters,
    )
