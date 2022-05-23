from typing import Collection
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from pasteur.pipelines.general.nodes import filter_by_keys

from pasteur.pipelines.synth.nodes import (
    fit_table,
    sdv_fit,
    sdv_sample,
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
                inputs=["wrk_%s" % t, "params:metadata.tables.%s" % t],
                outputs="transformer_%s" % t,
            ),
            node(
                func=transform_table,
                inputs=["wrk_%s" % t, "transformer_%s" % t],
                outputs="encoded_%s" % t,
            ),
            node(
                func=reverse_transform_table,
                inputs=["encoded_%s" % t, "transformer_%s" % t],
                outputs="decoded_%s" % t,
            ),
        ]

    return pipeline(table_nodes)


def create_synth_pipeline(tables):
    tables = [t.split(".")[-1] for t in tables]
    alg = "hma1"

    synth_pipe = pipeline(
        [
            node(
                func=sdv_fit,
                inputs={
                    "metadata": "params:metadata",
                    **{t: "encoded_%s" % t for t in tables},
                },
                outputs="%s_model" % alg,
            ),
            node(
                func=sdv_sample,
                inputs="%s_model" % alg,
                outputs=["%s_encoded_%s" % (alg, t) for t in tables],
            ),
        ]
    )

    decode_pipe = pipeline(
        [
            node(
                func=reverse_transform_table,
                inputs=["%s_encoded_%s" % (alg, t), "transformer_%s" % t],
                outputs="%s_%s" % (alg, t),
            )
            for t in tables
        ]
    )

    return synth_pipe + decode_pipe


def create_split_pipeline(tables):
    tables = [t.split(".")[-1] for t in tables]

    return pipeline(
        [
            node(
                func=filter_by_keys,
                inputs=["%s" % t, "keys_wrk"],
                outputs="wrk_%s" % t,
            )
            for t in tables
        ]
    )


def create_pipeline(dataset: str, view: str, tables: Collection[str]) -> Pipeline:
    return modular_pipeline(
        pipe=create_transform_pipeline(tables)
        + create_synth_pipeline(tables)
        + create_split_pipeline(tables),
        namespace=view,
        inputs={"keys_wrk": "%s.keys_wrk" % dataset},
    )
