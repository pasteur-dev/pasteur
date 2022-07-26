from typing import Collection, Dict, List
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from .synth import get_algs as synth_get_algs

from .transform import (
    fit_table_closure,
    transform_table,
    reverse_table,
)


def create_transform_pipeline(
    tables, type, requirements: Dict[str, Collection[str]] | None = None
):
    table_nodes = []
    if requirements is None:
        requirements = {}

    for t in tables:
        table_nodes += [
            node(
                func=fit_table_closure(t, type),
                inputs={"metadata": "params:metadata", **{t: t for t in tables}},
                outputs=f"trn_{t}",
            ),
            node(
                func=transform_table,
                inputs={"transformer": f"trn_{t}", **{t: t for t in tables}},
                outputs=[f"enc_{t}", f"ids_{t}"],
            ),
            node(
                func=reverse_table,
                inputs={
                    "transformer": f"trn_{t}",
                    "ids": f"ids_{t}",
                    "table": f"enc_{t}",
                    **{req: f"dec_{req}" for req in requirements.get(t, [])},
                },
                outputs=f"dec_{t}",
            ),
        ]

    return pipeline(table_nodes)


def create_synth_pipeline(
    alg: str, tables: List[str], requirements: Dict[str, Collection[str]] | None = None
):
    model = synth_get_algs()[alg]
    if requirements is None:
        requirements = {}

    synth_pipe = pipeline(
        [
            node(
                func=model.fit,
                inputs={
                    "metadata": "params:metadata",
                    **{f"ids_{t}": f"in_ids_{t}" for t in tables},
                    **{f"enc_{t}": f"in_enc_{t}" for t in tables},
                },
                outputs="model",
            ),
            node(
                func=model.sample,
                inputs="model",
                outputs={
                    **{f"ids_{t}": f"ids_{t}" for t in tables},
                    **{f"enc_{t}": f"enc_{t}" for t in tables},
                },
            ),
        ]
    )

    decode_pipe = pipeline(
        [
            node(
                func=reverse_table,
                inputs={
                    "transformer": f"trn_{t}",
                    "ids": f"ids_{t}",
                    "table": f"enc_{t}",
                    **{req: req for req in requirements.get(t, [])},
                },
                outputs=t,
            )
            for t in tables
        ]
    )

    return synth_pipe + decode_pipe


def create_pipeline(
    view: str,
    split: str,
    alg: str,
    tables: Collection[str],
    requirements: Dict[str, Collection[str]] | None = None,
) -> Pipeline:
    tables = [t.split(".")[-1] for t in tables]

    type = "idx"
    transform_mpipe = modular_pipeline(
        pipe=create_transform_pipeline(tables, type, requirements),
        namespace=f"{view}.{split}",
        parameters={
            "metadata": f"{view}.metadata",
        },
    )

    synth_pipe = create_synth_pipeline(alg, tables, requirements)
    synth_mpipe = modular_pipeline(
        pipe=synth_pipe,
        namespace=f"{view}.{alg}",
        inputs={
            **{f"in_enc_{t}": f"{view}.{split}.enc_{t}" for t in tables},
            **{f"in_ids_{t}": f"{view}.{split}.ids_{t}" for t in tables},
            **{f"trn_{t}": f"{view}.{split}.trn_{t}" for t in tables},
        },
        parameters={"metadata": f"{view}.metadata"},
    )

    return transform_mpipe + synth_mpipe


def get_algs():
    return list(synth_get_algs().keys())
