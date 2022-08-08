from typing import Collection, Dict

from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...synth import synth_fit_closure, synth_sample
from .transform import (
    find_ids,
    fit_table_closure,
    reverse_table_closure,
    transform_table_closure,
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
                func=fit_table_closure(t, types),
                inputs={
                    "metadata": "params:metadata",
                    "params": "parameters",
                    **{t: t for t in tables},
                },
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
            "metadata": f"{view}",
            "parameters": f"parameters",
        },
    )


def create_synth_pipeline(
    view: str,
    split: str,
    cls: str,
    tables: Collection[str],
    requirements: Dict[str, Collection[str]] | None = None,
):
    alg = cls.name
    type = cls.type
    if requirements is None:
        requirements = {}

    synth_pipe = pipeline(
        [
            node(
                func=synth_fit_closure(cls),
                inputs={
                    **{f"trn_{t}": f"trn_{t}" for t in tables},
                    **{f"ids_{t}": f"in_ids_{t}" for t in tables},
                    **{f"enc_{t}": f"in_enc_{t}" for t in tables},
                },
                outputs="model",
            ),
            node(
                func=synth_sample,
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
    )
