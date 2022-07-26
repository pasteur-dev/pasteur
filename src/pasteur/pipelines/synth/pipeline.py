from typing import Collection, Dict
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from .synth import get_algs as synth_get_algs

from .transform import (
    fit_table_closure,
    find_ids,
    transform_table_closure,
    reverse_table_closure,
)


def create_transform_pipeline(
    view: str, split: str, tables: Collection[str], types: Collection[str] | str = []
):
    if isinstance(types, str):
        types = [types]
    tables = [t.split(".")[-1] for t in tables]
    table_nodes = []

    for t in tables:
        table_nodes += [
            node(
                func=fit_table_closure(t, type),
                inputs={"metadata": "params:metadata", **{t: t for t in tables}},
                outputs=f"trn_{t}",
            ),
            node(
                func=find_ids,
                inputs={"transformer": f"trn_{t}", **{t: t for t in tables}},
                outputs=f"ids_{t}",
            ),
        ]

        table_nodes += [
            node(
                func=transform_table_closure(type),
                inputs={
                    "transformer": f"trn_{t}",
                    "ids": f"ids_{t}",
                    **{t: t for t in tables},
                },
                outputs=f"{type}_{t}",
            )
            for type in types
        ]

    return modular_pipeline(
        pipe=pipeline(table_nodes),
        namespace=f"{view}.{split}",
        parameters={
            "metadata": f"{view}.metadata",
        },
    )


def create_synth_pipeline(
    view: str,
    split: str,
    alg: str,
    tables: Collection[str],
    requirements: Dict[str, Collection[str]] | None = None,
):
    model = synth_get_algs()[alg]
    type = model.type
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
                func=reverse_table_closure(type),
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

    return modular_pipeline(
        pipe=synth_pipe + decode_pipe,
        namespace=f"{view}.{alg}",
        inputs={
            **{f"in_enc_{t}": f"{view}.{split}.{type}_{t}" for t in tables},
            **{f"in_ids_{t}": f"{view}.{split}.ids_{t}" for t in tables},
            **{f"trn_{t}": f"{view}.{split}.trn_{t}" for t in tables},
        },
        parameters={"metadata": f"{view}.metadata"},
    )


def get_algs():
    return synth_get_algs()
