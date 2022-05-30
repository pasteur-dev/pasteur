from typing import Collection, List
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from pasteur.pipelines.general.pipeline import create_split_pipeline

from .nodes import (
    fit_table,
    synth_fit_closure,
    synth_sample_closure,
    transform_table,
    reverse_transform_table,
)


def create_transform_pipeline(tables):
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


def create_synth_pipeline(alg: str, tables: List[str]):

    synth_pipe = pipeline(
        [
            node(
                func=synth_fit_closure(alg),
                inputs={
                    "metadata": "params:metadata",
                    **{t: "in_%s" % t for t in tables},
                },
                outputs="model",
            ),
            node(
                func=synth_sample_closure(alg),
                inputs="model",
                outputs={t: "encoded_%s" % t for t in tables},
            ),
        ]
    )

    decode_pipe = pipeline(
        [
            node(
                func=reverse_transform_table,
                inputs=["encoded_%s" % t, "transformer_%s" % t],
                outputs=t,
            )
            for t in tables
        ]
    )

    return synth_pipe + decode_pipe


def create_pipeline(
    dataset: str, view: str, alg: str, tables: Collection[str]
) -> Pipeline:
    tables = [t.split(".")[-1] for t in tables]

    split_mpipe = create_split_pipeline("wrk", dataset, view, tables)

    transform_mpipe = modular_pipeline(
        pipe=create_transform_pipeline(tables),
        namespace="%s.wrk" % view,
        parameters={
            **{
                "metadata.tables.%s" % t: "%s.metadata.tables.%s" % (view, t)
                for t in tables
            },
        },
    )

    synth_pipe = create_synth_pipeline(alg, tables)
    synth_mpipe = modular_pipeline(
        pipe=synth_pipe,
        namespace="%s.%s" % (view, alg),
        inputs={
            **{"in_%s" % t: "%s.wrk.encoded_%s" % (view, t) for t in tables},
            **{
                "transformer_%s" % t: "%s.wrk.transformer_%s" % (view, t)
                for t in tables
            },
        },
        parameters={"metadata": "%s.metadata" % view},
    )

    return split_mpipe + transform_mpipe + synth_mpipe


def get_algs():
    return ["hma1"]
