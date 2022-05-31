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
                inputs=[t, f"params:metadata.tables.{t}"],
                outputs=f"transformer_{t}",
            ),
            node(
                func=transform_table,
                inputs=[t, f"transformer_{t}"],
                outputs=f"encoded_{t}",
            ),
            node(
                func=reverse_transform_table,
                inputs=[f"encoded_{t}", f"transformer_{t}"],
                outputs=f"decoded_{t}",
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
                    **{t: f"in_{t}" for t in tables},
                },
                outputs="model",
            ),
            node(
                func=synth_sample_closure(alg),
                inputs="model",
                outputs={t: f"encoded_{t}" for t in tables},
            ),
        ]
    )

    decode_pipe = pipeline(
        [
            node(
                func=reverse_transform_table,
                inputs=[f"encoded_{t}", f"transformer_{t}"],
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
        namespace=f"{view}.wrk",
        parameters={
            **{f"metadata.tables.{t}": f"{view}.metadata.tables.{t}" for t in tables},
        },
    )

    synth_pipe = create_synth_pipeline(alg, tables)
    synth_mpipe = modular_pipeline(
        pipe=synth_pipe,
        namespace=f"{view}.{alg}",
        inputs={
            **{f"in_{t}": f"{view}.wrk.encoded_{t}" for t in tables},
            **{f"transformer_{t}": f"{view}.wrk.transformer_{t}" for t in tables},
        },
        parameters={"metadata": f"{view}.metadata"},
    )

    return split_mpipe + transform_mpipe + synth_mpipe


def get_algs():
    return ["hma1"]
